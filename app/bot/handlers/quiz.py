import asyncio
import logging
import random
import re

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.formatting import esc
from app.core.rag_pipeline import get_pipeline
from app.db.models import User
from app.db.repositories.document import DocumentRepository
from app.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)
router = Router()

# In-memory quiz state: telegram_id -> state dict
_quiz_state: dict[int, dict] = {}

_QUIZ_PROMPT = (
    "На основе этого фрагмента учебного материала, составь 1 вопрос с 4 вариантами ответа "
    "(один правильный). Формат строго:\n"
    "Вопрос: <текст вопроса>\n"
    "А) <вариант>\n"
    "Б) <вариант>\n"
    "В) <вариант>\n"
    "Г) <вариант>\n"
    "Правильный: <буква>\n\n"
    "Фрагмент:\n{chunk_text}"
)

_ANSWER_MAP = {"А": "A", "Б": "B", "В": "C", "Г": "D"}
_REVERSE_MAP = {"A": "А", "B": "Б", "C": "В", "D": "Г"}


def _parse_quiz_response(text: str) -> dict | None:
    """Parse LLM quiz response into structured data."""
    lines = text.strip().split("\n")
    question = None
    options = {}
    correct = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("вопрос:"):
            question = line.split(":", 1)[1].strip()
        elif re.match(r"^А\)", line):
            options["А"] = re.sub(r"^А\)\s*", "", line)
        elif re.match(r"^Б\)", line):
            options["Б"] = re.sub(r"^Б\)\s*", "", line)
        elif re.match(r"^В\)", line):
            options["В"] = re.sub(r"^В\)\s*", "", line)
        elif re.match(r"^Г\)", line):
            options["Г"] = re.sub(r"^Г\)\s*", "", line)
        elif lower.startswith("правильный:"):
            ans = line.split(":", 1)[1].strip()
            for ru in _ANSWER_MAP:
                if ans.startswith(ru):
                    correct = ru
                    break

    if question and len(options) == 4 and correct and correct in options:
        return {
            "question": question,
            "options": options,
            "correct": correct,
        }
    return None


def _build_question_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="А", callback_data="quiz:A"),
            InlineKeyboardButton(text="Б", callback_data="quiz:B"),
            InlineKeyboardButton(text="В", callback_data="quiz:C"),
            InlineKeyboardButton(text="Г", callback_data="quiz:D"),
        ]
    ])


def _build_next_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Следующий вопрос", callback_data="quiz:next"),
            InlineKeyboardButton(text="Завершить", callback_data="quiz:stop"),
        ]
    ])


async def _get_random_chunks(user_telegram_id: int, count: int = 5) -> list[dict]:
    """Fetch random chunks from user's Qdrant collection."""
    pipeline = get_pipeline()
    client = pipeline.retriever.client
    collection_name = f"user_{user_telegram_id}"

    try:
        # Синхронный клиент Qdrant — вызовы уводим из event loop
        if not await asyncio.to_thread(client.collection_exists, collection_name):
            return []

        info = await asyncio.to_thread(client.get_collection, collection_name)
        total = info.points_count
        if total == 0:
            return []

        all_points, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=collection_name,
            limit=min(total, 500),
            with_payload=True,
            with_vectors=False,
        )

        if not all_points:
            return []

        selected = random.sample(all_points, min(count, len(all_points)))

        return [
            {
                "text": p.payload.get("text", ""),
                "filename": p.payload.get("filename", ""),
            }
            for p in selected
            if p.payload.get("text", "").strip()
        ]

    except Exception as e:
        logger.error("Failed to get random chunks for quiz: %s", e)
        return []


async def _generate_question(chunk_text: str) -> dict | None:
    """Generate a quiz question from a chunk using LLM."""
    provider = get_llm_provider()
    prompt = _QUIZ_PROMPT.format(chunk_text=chunk_text[:2000])

    messages = [
        {"role": "system", "content": "Ты — генератор тестовых вопросов по учебным материалам."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await provider.generate(messages, temperature=0.7, max_tokens=500)
        return _parse_quiz_response(response)
    except Exception as e:
        logger.error("Failed to generate quiz question: %s", e)
        return None


async def _send_question(
    message_or_callback,
    user_telegram_id: int,
) -> None:
    """Generate and send the next quiz question."""
    state = _quiz_state.get(user_telegram_id)
    if not state or not state["chunks"]:
        score = state["score"] if state else 0
        total = state["total"] if state else 0
        text = f"Квиз завершён!\n\nРезультат: {score}/{total}"
        if hasattr(message_or_callback, "message"):
            await message_or_callback.message.answer(text)
        else:
            await message_or_callback.answer(text)
        _quiz_state.pop(user_telegram_id, None)
        return

    chunk = state["chunks"].pop(0)

    if hasattr(message_or_callback, "message"):
        status_msg = await message_or_callback.message.answer("Генерирую вопрос...")
    else:
        status_msg = await message_or_callback.answer("Генерирую вопрос...")

    quiz_data = await _generate_question(chunk["text"])

    if not quiz_data:
        await status_msg.edit_text("Не удалось сгенерировать вопрос, пробую другой фрагмент...")
        await _send_question(message_or_callback, user_telegram_id)
        return

    state["current"] = quiz_data
    state["total"] += 1
    state["source_filename"] = chunk.get("filename", "")

    q_num = state["total"]
    text = (
        f"<b>Вопрос {q_num}:</b>\n\n"
        f"{esc(quiz_data['question'])}\n\n"
        f"А) {esc(quiz_data['options']['А'])}\n"
        f"Б) {esc(quiz_data['options']['Б'])}\n"
        f"В) {esc(quiz_data['options']['В'])}\n"
        f"Г) {esc(quiz_data['options']['Г'])}"
    )

    await status_msg.edit_text(text, reply_markup=_build_question_keyboard(), parse_mode="HTML")


@router.message(Command("quiz"))
async def cmd_quiz(message: Message, user: User, session: AsyncSession) -> None:
    """Start quiz mode — generate questions from uploaded documents."""
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)
    ready_docs = [d for d in docs if d.status == "ready"]

    if not ready_docs:
        await message.answer(
            "У вас нет загруженных документов.\n"
            "Сначала загрузите учебные материалы, а потом запустите квиз."
        )
        return

    status_msg = await message.answer("Подготавливаю квиз по вашим материалам...")

    chunks = await _get_random_chunks(user.telegram_id, count=5)
    if not chunks:
        await status_msg.edit_text(
            "Не удалось получить фрагменты из документов. "
            "Попробуйте позже или загрузите документы заново."
        )
        return

    _quiz_state[user.telegram_id] = {
        "chunks": chunks,
        "current": None,
        "score": 0,
        "total": 0,
        "source_filename": "",
    }

    await status_msg.delete()
    await _send_question(message, user.telegram_id)


@router.callback_query(F.data.startswith("quiz:"))
async def handle_quiz_callback(callback: CallbackQuery, user: User) -> None:
    """Handle quiz answer buttons and navigation."""
    action = callback.data.split(":", 1)[1]

    state = _quiz_state.get(user.telegram_id)
    if not state:
        await callback.answer("Квиз не активен. Запустите /quiz", show_alert=True)
        return

    if action in ("A", "B", "C", "D"):
        current = state.get("current")
        if not current:
            await callback.answer("Вопрос не найден", show_alert=True)
            return

        chosen_ru = _REVERSE_MAP[action]
        correct_ru = current["correct"]

        if chosen_ru == correct_ru:
            state["score"] += 1
            result_text = f"Правильно!\n\nОтвет: {correct_ru}) {esc(current['options'][correct_ru])}"
        else:
            result_text = (
                f"Неправильно.\n\n"
                f"Ваш ответ: {chosen_ru}) {esc(current['options'][chosen_ru])}\n"
                f"Правильный ответ: {correct_ru}) {esc(current['options'][correct_ru])}"
            )

        score_text = f"\n\nСчёт: {state['score']}/{state['total']}"
        result_text += score_text

        has_more = bool(state["chunks"])
        if has_more:
            await callback.message.edit_text(
                result_text,
                reply_markup=_build_next_keyboard(),
            )
        else:
            result_text += f"\n\nКвиз завершён! Итого: {state['score']}/{state['total']}"
            await callback.message.edit_text(result_text)
            _quiz_state.pop(user.telegram_id, None)

        await callback.answer()

    elif action == "next":
        await callback.answer()
        await _send_question(callback, user.telegram_id)

    elif action == "stop":
        score = state["score"]
        total = state["total"]
        await callback.message.edit_text(
            f"Квиз завершён!\n\nРезультат: {score}/{total}"
        )
        _quiz_state.pop(user.telegram_id, None)
        await callback.answer()
