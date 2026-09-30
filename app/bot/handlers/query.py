import asyncio
import contextlib
import logging
import re
import time
from collections import defaultdict
from collections.abc import AsyncGenerator

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.formatting import esc, llm_to_html
from app.config import settings
from app.core.classifier import classify_intent
from app.core.conversation import get_context
from app.core.embedder import EmbeddingServiceError
from app.core.rag_pipeline import get_pipeline
from app.db.models import QueryHistory, User
from app.db.repositories.document import DocumentRepository
from app.db.repositories.user import UserRepository
from app.llm.provider import LLMError
from app.presets import get_preset

logger = logging.getLogger(__name__)
router = Router()

_STREAM_CURSOR = " \u258d"

# ── Chat by default: детектируем RAG-запросы, остальное — чат ──

# Паттерны, которые ОДНОЗНАЧНО означают «ищи в документах».
_RAG_PATTERNS: list[re.Pattern[str]] = [
    # Прямые команды поиска с указанием источника
    re.compile(
        r"(найди|поищи|отыщи|покажи|выведи|дай)"
        r"\s.{0,30}"
        r"(в документ|в файл|по документ|в загруженн|из документ|из файл)",
    ),
    # «Что написано / говорится / указано в …»
    re.compile(r"что (написано|говорится|сказано|указано) в"),
    # «Согласно документу / по документу / из документа X»
    re.compile(r"(согласно|из)\s+(документ|файл)\w*\s+\S"),
    # Прямая ссылка на конкретный файл
    re.compile(r"\S+\.(md|pdf|docx|txt)\b"),
    # «Цитируй / процитируй»
    re.compile(r"(про)?цитируй"),
    # «Расскажи/объясни из документов»
    re.compile(r"(расскажи|объясни|опиши).{0,20}(из документ|из файл|по документ)"),
]

# Быстрые паттерны очевидного чата (приветствия, эмоции и т.д.).
# Экономят вызов LLM-классификатора на тривиальных сообщениях.
_GREETING_PATTERNS: list[re.Pattern[str]] = [
    # Приветствия
    re.compile(
        r"^(привет|здравствуй(те)?|хай|хеллоу|салют|йо|здарова|здоров"
        r"|дарова|даров|дароу|хола|ку|шалом|алоха|хэй|хей"
        r"|добр(ый|ое|ого)\s+(день|утро|утра|вечер|вечера)"
        r"|доброго\s+времени\s+суток)"
        r"[!.,?]*$",
    ),
    # Прощания
    re.compile(
        r"^(пока|до свидания|бай|прощай|увидимся|спокойной ночи|бб|досвидос|чао)"
        r"[!.,?]*$",
    ),
    # Благодарности
    re.compile(
        r"^(спасибо|благодарю|мерси|сенкс|thanks|спс|пасиб|пасибо)"
        r"[!.,?]*$",
    ),
    # Casual-вопросы / small talk
    re.compile(
        r"^(как дела|как жизнь|как ты|как сам|как оно|как настроение"
        r"|что нового|что делаешь|чем занят|чем заним(аешься)?)"
        r"[!.,?]*$",
    ),
    # Эмоциональные реакции / короткие реплики
    re.compile(
        r"^(грустно|весело|круто|класс|супер|ого|вау|блин|жаль"
        r"|ок|окей|ладно|понятно|хорошо|отлично|ясно|норм|нормально"
        r"|да|нет|конечно|точно|согласен|верно|неа|ага|угу"
        r"|лол|хаха|ахах|кек|ржу|ржунимагу|смешно|забавно)"
        r"[!.,?)]*$",
    ),
    # Проверка бота
    re.compile(
        r"^(ты (жив|тут|здесь|умер|сдох|работаешь|спишь|живой|мёртв|мертв)"
        r"|алло|ау|эй|есть кто)"
        r"[!.,?]*$",
    ),
]

# Слова-индикаторы содержательного запроса.
# Если сообщение из 1-2 слов содержит такое слово — НЕ считаем quick-chat.
def _get_rag_indicator_words() -> frozenset[str]:
    """RAG-индикаторные слова из пресета или дефолтные."""
    preset = get_preset()
    if preset.rag_keywords:
        return frozenset(preset.rag_keywords)
    return frozenset({
        "закон", "статья", "пункт", "документ", "договор", "кодекс", "конституция",
        "указ", "постановление", "приказ", "регламент", "инструкция", "акт",
        "налог", "штраф", "суд", "иск", "жалоба", "апелляция", "кассация",
        "api", "pdf", "steam", "ключ", "лицензия", "доступ", "сервис",
    })


# Кэшируем при первом вызове
_rag_indicator_cache: frozenset[str] | None = None


def _get_cached_rag_indicators() -> frozenset[str]:
    global _rag_indicator_cache
    if _rag_indicator_cache is None:
        _rag_indicator_cache = _get_rag_indicator_words()
    return _rag_indicator_cache

# Маркеры начала уточняющего/follow-up вопроса.
# Короткое сообщение (1-2 слова) с таким маркером — вероятный follow-up, не quick-chat.
_FOLLOWUP_STARTERS = frozenset({
    "а", "и", "но", "ещё", "еще", "ещe", "тогда", "плюс",
    "зачем", "почему", "когда", "как", "сколько", "откуда",
    # follow-up слова, которые НЕ должны попадать в quick_chat
    "подробнее", "детальнее", "продолжай", "дальше", "поподробнее",
    "конкретнее", "уточни", "разверни", "поясни", "объясни",
})


def _needs_rag(text: str) -> bool:
    """Определить, является ли сообщение явным запросом к документам."""
    normalized = text.lower().strip()
    return any(p.search(normalized) for p in _RAG_PATTERNS)


def _is_quick_chat(text: str) -> bool:
    """Быстрый детектор очевидного чата (приветствия, 1-2 слова).

    Для экономии вызовов LLM-классификатора на тривиальных сообщениях.
    """
    normalized = text.lower().strip()
    words = normalized.split()

    # 1-2 слова без RAG-индикаторов → чат, НО:
    # если первое слово — дискурсивный маркер (А/И/Но/Тогда...) →
    # вероятный follow-up, отправляем к классификатору
    if len(words) <= 2:
        has_indicator = any(w.rstrip("?!.,") in _get_cached_rag_indicators() for w in words)
        if not has_indicator:
            # Маркер follow-up в начале — пусть LLM-классификатор разберётся
            return words[0].rstrip("?!.,") not in _FOLLOWUP_STARTERS

    # Чистые приветствия, прощания, благодарности, эмоции
    return any(p.search(normalized) for p in _GREETING_PATTERNS)


def _build_user_state(user: "User", docs: list) -> str:
    """Компактная строка состояния пользователя для CHAT_PROMPT (~200 символов max)."""
    tier_names = {"free": "Free", "pro": "Pro", "admin": "Admin"}
    tier = tier_names.get(user.tier, user.tier)

    ready_docs = [d for d in docs if d.status == "ready"]
    is_unlimited = user.documents_limit >= 999_000

    # Список имён файлов (max 5, обрезка длинных имён)
    doc_names = []
    for d in ready_docs[:5]:
        name = d.filename if len(d.filename) <= 30 else d.filename[:27] + "..."
        doc_names.append(name)
    if len(ready_docs) > 5:
        doc_names.append(f"и ещё {len(ready_docs) - 5}")

    if is_unlimited:
        docs_str = f"{len(ready_docs)} ({', '.join(doc_names)})" if doc_names else "0"
        queries_str = str(user.queries_today)
    else:
        docs_str = f"{len(ready_docs)}/{user.documents_limit}"
        if doc_names:
            docs_str += f" ({', '.join(doc_names)})"
        queries_str = f"{user.queries_today}/{user.queries_limit}"

    preset = get_preset()
    base = f"Тариф: {tier} | Документы: {docs_str} | Запросы сегодня: {queries_str}"
    if preset.features.law_search:
        law_str = "вкл" if user.law_search_enabled else "выкл"
        base += f" | Законодательство: {law_str}"
    return base


# ── Очистка LLM-сгенерированных ссылок на источники ──

_SOURCE_LINE_RE = re.compile(
    r"^\s*[Ии]сточник[и]?\s*[:—–\-].*$",
    re.MULTILINE,
)
_SOURCE_INLINE_RE = re.compile(r"\((?:см\.|источник:?\s*)\s*[^)]+\)", re.IGNORECASE)

# Qwen3 thinking blocks — strip <think>...</think> from output
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_llm_sources(text: str) -> str:
    """Удалить из ответа LLM строки вида 'Источник: ...' и <think> блоки Qwen3."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _SOURCE_LINE_RE.sub("", text)
    text = _SOURCE_INLINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip()


def _format_sources(sources: list[dict]) -> str:
    """Компактное форматирование источников: группировка по файлу, диапазон страниц."""
    by_file: dict[str, set[int]] = defaultdict(set)
    law_sources: list[dict] = []

    for s in sources:
        if s.get("source_type") == "law":
            law_sources.append(s)
        else:
            fn = s["filename"]
            if s.get("page_number"):
                by_file[fn].add(s["page_number"])
            else:
                by_file.setdefault(fn, set())

    lines = []
    for raw_fn, pages in by_file.items():
        fn = esc(raw_fn)
        if pages:
            sorted_pages = sorted(pages)
            if len(sorted_pages) == 1:
                lines.append(f"  📄 {fn} — стр. {sorted_pages[0]}")
            else:
                lines.append(f"  📄 {fn} — стр. {sorted_pages[0]}–{sorted_pages[-1]}")
        else:
            lines.append(f"  📄 {fn}")

    # Правовые акты (дедупликация по pravo_nd)
    seen_law: set[str] = set()
    for s in law_sources:
        nd = s.get("pravo_nd", "")
        if nd in seen_law:
            continue
        seen_law.add(nd)
        title = (s.get("heading") or "Правовой акт")[:60]
        info_parts = []
        if s.get("doc_type"):
            info_parts.append(s["doc_type"])
        if s.get("doc_date"):
            info_parts.append(f"от {s['doc_date']}")
        if s.get("doc_number"):
            info_parts.append(f"N{s['doc_number']}")
        info = " ".join(info_parts)
        lines.append(f"  ⚖️ {esc(title)}")
        if info:
            lines.append(f"      {esc(info)}")

    return "\n".join(lines)


def _build_final_response(
    answer: str,
    sources: list[dict],
    model: str,
    law_search_failed: bool,
) -> str:
    """Собрать финальное HTML-сообщение с источниками и моделью."""
    answer = _strip_llm_sources(answer)
    response = llm_to_html(answer)

    # Если ответ пустой после обработки — заменяем на "не найдено"
    if not response.strip():
        response = "В загруженных документах ответа на этот вопрос не найдено."

    # Не показываем источники, если бот сам сказал "не найдено"
    _NOT_FOUND_MARKERS = ("не найдено", "не нашёл", "отсутствует информация", "нет информации")
    answer_lower = answer.lower()
    has_answer = not any(marker in answer_lower for marker in _NOT_FOUND_MARKERS)

    if sources and has_answer:
        sources_text = _format_sources(sources)
        response = f"{response}\n\n<b>Источники:</b>\n{sources_text}"
    if law_search_failed:
        response += "\n\n<i>⚠️ Сервис законодательства временно недоступен.</i>"

    return response


async def _safe_edit_text(
    msg: Message,
    text: str,
    parse_mode: str | None = "HTML",
    max_retries: int = 2,
) -> bool:
    """Обёртка edit_text с обработкой Telegram Flood Control.

    При TelegramRetryAfter — ждёт указанное время и повторяет.
    При TelegramBadRequest — пропускает (сообщение не изменилось и т.п.).
    Возвращает True если edit удался, False если не удалось.
    """
    # Telegram HTML не поддерживает <br> — заменяем на перенос строки
    text = re.sub(r"<br\s*/?>", "\n", text)
    # LLM иногда использует Markdown вместо HTML — конвертируем
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    for attempt in range(max_retries + 1):
        try:
            await msg.edit_text(text[:4096], parse_mode=parse_mode)
            return True
        except TelegramRetryAfter as e:
            if attempt < max_retries:
                wait = e.retry_after + 1
                logger.warning(
                    "Flood control: edit_text заблокирован на %dс (попытка %d/%d)",
                    e.retry_after, attempt + 1, max_retries + 1,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "Flood control: не удалось отправить edit_text "
                    "после %d попыток (retry_after=%d)",
                    max_retries + 1, e.retry_after,
                )
                return False
        except TelegramBadRequest as e:
            logger.warning("edit_text TelegramBadRequest: %s (parse_mode=%s)", e, parse_mode)
            if parse_mode == "HTML":
                plain = re.sub(r"<[^>]+>", "", text)
                try:
                    await msg.edit_text(plain[:4096], parse_mode=None)
                    return True
                except Exception:
                    logger.debug("edit_text plain fallback failed", exc_info=True)
            return False
    return False


async def _wrap_chat_stream(
    gen: AsyncGenerator[str, None], model: str,
) -> AsyncGenerator[dict, None]:
    """Обернуть chat-дельты (str) в формат RAG-событий (dict)."""
    yield {"type": "meta", "sources": [], "model": model, "law_search_failed": False}
    async for delta in gen:
        yield {"type": "delta", "text": delta}


async def _stream_to_telegram(
    status_msg: Message,
    event_stream: AsyncGenerator[dict, None],
) -> tuple[str, list[dict], str, bool]:
    """Буферизированный стриминг: LLM → буфер → Telegram по таймеру.

    LLM-дельты складываются в буфер мгновенно.
    Параллельный цикл выдаёт порции текста с контролируемой скоростью.
    """
    buffer = ""
    displayed = 0
    sources: list[dict] = []
    model = ""
    law_search_failed = False
    done = False
    edit_count = 0
    collector_error: BaseException | None = None
    stream_start = time.monotonic()

    tick_interval = settings.stream_tick_interval
    chars_per_tick = settings.stream_chars_per_tick
    show_cursor = settings.stream_show_cursor

    async def _collect() -> None:
        nonlocal buffer, sources, model, law_search_failed, done, collector_error
        try:
            async for event in event_stream:
                if isinstance(event, dict) and event.get("type") == "meta":
                    sources = event["sources"]
                    model = event["model"]
                    law_search_failed = event.get("law_search_failed", False)
                    # Статус «поиск завершён» необязателен: сбой не должен ронять ответ
                    with contextlib.suppress(Exception):
                        await status_msg.edit_text("Генерирую ответ...", parse_mode=None)
                    continue
                text = event["text"] if isinstance(event, dict) else event
                buffer += text
        except BaseException as e:
            collector_error = e
        finally:
            done = True

    collector = asyncio.create_task(_collect())

    flood_paused_until: float = 0  # monotonic timestamp

    try:
        while not done or displayed < len(buffer):
            await asyncio.sleep(tick_interval)

            if collector_error is not None:
                raise collector_error

            if len(buffer) <= displayed:
                continue

            next_pos = min(displayed + chars_per_tick, len(buffer))
            displayed = next_pos
            display_text = buffer[:displayed]

            # Под Flood Control — пропускаем edit до истечения паузы
            if time.monotonic() < flood_paused_until:
                continue

            cursor = _STREAM_CURSOR if show_cursor and (not done or displayed < len(buffer)) else ""
            try:
                await status_msg.edit_text(
                    (display_text + cursor)[:4096],
                    parse_mode=None,
                )
                edit_count += 1
            except TelegramRetryAfter as e:
                flood_paused_until = time.monotonic() + e.retry_after + 1
                logger.warning(
                    "Flood control в стриминге: пауза %dс, буфер=%d симв., отображено=%d",
                    e.retry_after, len(buffer), displayed,
                )
            except TelegramBadRequest:
                pass

        await collector
    except BaseException:
        collector.cancel()
        raise

    elapsed = time.monotonic() - stream_start
    logger.info(
        "Streaming query OK: %d chars, %d edits, %.1fс",
        len(buffer), edit_count, elapsed,
    )

    return buffer, sources, model, law_search_failed


async def process_query_text(
    question: str,
    status_msg: Message,
    user: User,
    session: AsyncSession,
    skip_retrieval: bool = False,
    user_state: str | None = None,
    mode: str = "rag",
) -> bool:
    """RAG-запрос со стримингом — переиспользуется из text и voice хендлеров.

    Returns:
        True — ответ сгенерирован полностью; False — обработка прервалась ошибкой
        (пользователь уже получил сообщение об ошибке, запрос можно вернуть в лимит).
    """
    full_text = ""

    try:
        conv_ctx = get_context(user.telegram_id)
        conversation_history = conv_ctx.get_history_messages() or None

        pipeline = get_pipeline()

        if skip_retrieval:
            model = pipeline.generator.provider.model
            chat_gen = pipeline.generator.generate_stream(
                question, [],
                conversation_history=conversation_history,
                user_state=user_state,
                mode=mode,
            )
            event_stream = _wrap_chat_stream(chat_gen, model)
        else:
            event_stream = pipeline.query_stream(
                question=question,
                user_telegram_id=user.telegram_id,
                conversation_history=conversation_history,
                law_search_enabled=user.law_search_enabled,
                user_state=user_state,
            )

        full_text, sources, model, law_search_failed = await _stream_to_telegram(
            status_msg, event_stream,
        )

        # --- Финальное сообщение (ждём при flood control) ---
        response = _build_final_response(full_text, sources, model, law_search_failed)
        if not await _safe_edit_text(status_msg, response):
            logger.error("Не удалось отправить финальное сообщение (flood control)")

        # Сохраняем историю запроса
        try:
            history = QueryHistory(
                user_id=user.id,
                question=question,
                answer=full_text,
                sources=sources or None,
                tokens_used=None,
                model_used=model,
            )
            session.add(history)
            await session.commit()
        except Exception:
            logger.warning("Не удалось сохранить историю запроса", exc_info=True)

        # Обновляем контекст диалога (raw answer, до HTML-конвертации)
        conv_ctx.add_pair(question, full_text)
        return True

    except LLMError as e:
        logger.exception("Ошибка LLM при обработке запроса")
        if full_text:
            partial = llm_to_html(full_text)
            partial += "\n\n<i>\u26a0\ufe0f Генерация прервана: " + esc(e) + "</i>"
            await _safe_edit_text(status_msg, partial)
        else:
            await _safe_edit_text(status_msg, str(e), parse_mode=None)
    except EmbeddingServiceError as e:
        logger.warning("Сервис эмбеддингов недоступен: %s", e)
        await _safe_edit_text(
            status_msg, "Сервис эмбеддингов временно недоступен. Попробуйте позже.",
            parse_mode=None,
        )
    except Exception:
        logger.exception("Ошибка при обработке запроса")
        if full_text:
            partial = llm_to_html(full_text)
            partial += "\n\n<i>\u26a0\ufe0f Генерация прервана из-за внутренней ошибки.</i>"
            await _safe_edit_text(status_msg, partial)
        else:
            await _safe_edit_text(
                status_msg, "Произошла внутренняя ошибка. Попробуйте позже.",
                parse_mode=None,
            )
    return False


def _rag_status_text(ready_docs: list, user: "User") -> str:
    """Текст статуса при запуске RAG-поиска."""
    if user.law_search_enabled and ready_docs:
        return "Ищу ответ в документах и базе законодательства..."
    if user.law_search_enabled:
        return "Ищу ответ в базе законодательства..."
    if ready_docs:
        return "Ищу ответ в документах..."
    return "Думаю..."


async def route_and_process(
    question: str,
    status_msg: Message,
    user: User,
    session: AsyncSession,
) -> bool:
    """Единая маршрутизация: chat by default, RAG when needed.

    Используется из text- и voice-хендлеров.

    Returns:
        True — ответ отдан полностью (см. process_query_text).
    """
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)
    ready_docs = [d for d in docs if d.status == "ready"]
    state = _build_user_state(user, docs)

    # 1. Нечего искать → всегда чат
    if not ready_docs and not user.law_search_enabled:
        await _safe_edit_text(status_msg, "Думаю...", parse_mode=None)
        return await process_query_text(
            question, status_msg, user, session,
            skip_retrieval=True, user_state=state, mode="chat",
        )

    # 2. Явный RAG-запрос (быстрый regex)
    if _needs_rag(question):
        await _safe_edit_text(
            status_msg, _rag_status_text(ready_docs, user), parse_mode=None,
        )
        return await process_query_text(question, status_msg, user, session, user_state=state)

    # 3. Очевидный чат (приветствия, 1-2 слова) → экономим на классификаторе
    if _is_quick_chat(question):
        await _safe_edit_text(status_msg, "Думаю...", parse_mode=None)
        return await process_query_text(
            question, status_msg, user, session,
            skip_retrieval=True, user_state=state, mode="chat",
        )

    # 4. Неоднозначный случай → LLM-классификатор (default: chat)
    doc_names = [d.filename for d in ready_docs]
    conv_ctx = get_context(user.telegram_id)
    history = conv_ctx.get_history_messages()

    intent = await classify_intent(
        question, history, doc_names, user.law_search_enabled,
    )

    if intent == "rag":
        conv_ctx_r = get_context(user.telegram_id)
        conv_ctx_r.last_rag_query = question
        await _safe_edit_text(
            status_msg, _rag_status_text(ready_docs, user), parse_mode=None,
        )
        return await process_query_text(question, status_msg, user, session, user_state=state)
    elif intent == "followup":
        conv_ctx_fw = get_context(user.telegram_id)
        if conv_ctx_fw.last_rag_query and ready_docs:
            # Последний вопрос был RAG — перенаправляем followup в RAG
            conv_ctx_fw.last_rag_query = question
            await _safe_edit_text(
                status_msg, _rag_status_text(ready_docs, user), parse_mode=None,
            )
            return await process_query_text(question, status_msg, user, session, user_state=state)
        else:
            await _safe_edit_text(status_msg, "Думаю...", parse_mode=None)
            return await process_query_text(
                question, status_msg, user, session,
                skip_retrieval=True, user_state=state, mode="followup",
            )
    else:
        # chat
        conv_ctx_ch = get_context(user.telegram_id)
        conv_ctx_ch.last_rag_query = None
        await _safe_edit_text(status_msg, "Думаю...", parse_mode=None)
        return await process_query_text(
            question, status_msg, user, session,
            skip_retrieval=True, user_state=state, mode=intent,
        )


@router.message(F.text)
async def handle_query(message: Message, user: User, session: AsyncSession) -> None:
    question = message.text
    if not question or question.startswith("/"):
        return

    status_msg = await message.answer("Думаю...")
    await answer_with_quota(question, status_msg, user, session)


async def answer_with_quota(
    question: str,
    status_msg: Message,
    user: User,
    session: AsyncSession,
    limit_prefix: str = "",
) -> None:
    """Списать запрос из дневного лимита, ответить, вернуть запрос при ошибке.

    Запрос списывается ДО обработки одним атомарным UPDATE — параллельные
    сообщения не проходят лимит. Если ответ не удался, запрос возвращается:
    как и раньше, в лимит идут только успешные ответы.
    """
    # Идентификаторы читаем заранее: после rollback атрибуты ORM-объекта
    # истекают, а ленивая подгрузка в async-сессии недоступна
    user_id, telegram_id = user.id, user.telegram_id

    user_repo = UserRepository(session)
    if not await user_repo.try_consume_query(user):
        await _safe_edit_text(
            status_msg,
            f"{limit_prefix}Достигнут дневной лимит запросов ({user.queries_limit}).\n"
            "Попробуйте завтра.",
        )
        return

    answered = False
    try:
        answered = await route_and_process(question, status_msg, user, session)
    finally:
        if not answered:
            try:
                # Сбой мог оставить транзакцию в ошибочном состоянии
                await session.rollback()
                await user_repo.refund_query(user_id)
            except Exception:
                logger.exception("Не удалось вернуть запрос в лимит (user=%s)", telegram_id)
