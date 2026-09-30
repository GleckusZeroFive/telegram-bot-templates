"""Хендлеры для работы с базой законодательства РФ."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.formatting import esc
from app.config import settings
from app.db.models import User

logger = logging.getLogger(__name__)
router = Router()


@router.message(Command("law"))
async def cmd_law(message: Message, user: User, session: AsyncSession) -> None:
    """
    /law — переключить поиск по законодательству (вкл/выкл).
    /law <запрос> — одноразовый поиск по законодательству.
    """
    if not settings.law_corpus_enabled:
        await message.answer("Поиск по законодательству не настроен на сервере.")
        return

    # Проверяем, есть ли текст после команды
    args = message.text.strip().removeprefix("/law").strip() if message.text else ""

    if args:
        # Одноразовый поиск
        from app.core.rag_pipeline import get_pipeline
        status_msg = await message.answer("Ищу в базе законодательства...")

        pipeline = get_pipeline()
        try:
            law_client = pipeline._law_client
            if not law_client:
                await status_msg.edit_text("Модуль законодательства не инициализирован.")
                return
            query_vector = await pipeline.embedder.embed_query(args)
            results = await law_client.search(args, query_vector, settings.law_corpus_top_k)

            if not results:
                await status_msg.edit_text("По запросу ничего не найдено в базе законодательства.")
                return

            # Форматируем результаты как краткий список
            # Запрос пользователя и тексты из внешнего API экранируются перед HTML
            lines = [f"<b>Результаты по запросу:</b> <i>{esc(args)}</i>\n"]
            for i, r in enumerate(results[:5], 1):
                title = (r.get("heading") or "Без заголовка")[:80]
                info_parts = []
                if r.get("doc_type"):
                    info_parts.append(r["doc_type"])
                if r.get("doc_date"):
                    info_parts.append(f"от {r['doc_date']}")
                if r.get("doc_number"):
                    info_parts.append(f"N{r['doc_number']}")
                info = " ".join(info_parts)
                status = r.get("status", "")

                lines.append(f"<b>{i}.</b> {esc(title)}")
                if info:
                    lines.append(f"   {esc(info)}")
                if status:
                    lines.append(f"   <i>{esc(status)}</i>")
                # Показываем начало текста
                text_preview = (r.get("text") or "")[:200].strip()
                if text_preview:
                    lines.append(f"   {esc(text_preview)}...")
                lines.append("")

            await status_msg.edit_text("\n".join(lines))
        except Exception:
            logger.exception("Ошибка поиска по законодательству")
            await status_msg.edit_text("Не удалось выполнить поиск. Сервис законодательства недоступен.")
        return

    # Toggle — переключение поиска по законодательству
    user.law_search_enabled = not user.law_search_enabled
    await session.commit()

    if user.law_search_enabled:
        await message.answer(
            "Поиск по законодательству РФ <b>включён</b> ✅\n\n"
            "Теперь при каждом вопросе я буду искать ответ и в базе "
            "нормативно-правовых актов.\n"
            "Для отключения — /law"
        )
    else:
        await message.answer(
            "Поиск по законодательству РФ <b>выключен</b> ❌\n"
            "Для включения — /law"
        )


@router.message(Command("lawstats"))
async def cmd_lawstats(message: Message) -> None:
    """Статистика базы законодательства."""
    if not settings.law_corpus_enabled:
        await message.answer("Поиск по законодательству не настроен на сервере.")
        return

    try:
        from app.core.rag_pipeline import get_pipeline
        pipeline = get_pipeline()
        client = pipeline._law_client
        if not client:
            await message.answer("Модуль законодательства не инициализирован.")
            return

        is_healthy = await client.health()
        if not is_healthy:
            await message.answer("Сервис законодательства недоступен.")
            return

        stats = await client.stats()
        text = (
            "<b>База законодательства РФ</b>\n\n"
            f"Документов всего: <b>{stats.get('total_documents', '?')}</b>\n"
            f"Проиндексировано: <b>{stats.get('indexed_documents', '?')}</b>\n"
            f"В очереди: {stats.get('pending_documents', '?')}\n"
            f"Ошибки: {stats.get('failed_documents', '?')}\n"
            f"Чанков в Qdrant: <b>{stats.get('qdrant_points', '?')}</b>\n"
        )
        await message.answer(text)
    except Exception:
        logger.exception("Ошибка получения статистики законодательства")
        await message.answer("Не удалось получить статистику. Сервис недоступен.")
