import logging

from aiogram import Bot, F, Router
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.formatting import esc
from app.bot.handlers.query import answer_with_quota
from app.config import settings
from app.core.embedder import EmbeddingServiceError
from app.core.transcriber import TranscriptionError, transcribe
from app.db.models import User

logger = logging.getLogger(__name__)
router = Router()

# Маппинг голосовых команд на команды бота
_VOICE_COMMANDS: dict[str, str] = {
    "помощь": "/help",
    "хелп": "/help",
    "старт": "/start",
    "начать": "/start",
    "документы": "/docs",
    "мои документы": "/docs",
    "статистика": "/stats",
    "стата": "/stats",
    "удалить": "/delete",
    "сброс": "/reset",
    "новый диалог": "/new",
    "новый": "/new",
}


def _detect_command(text: str) -> str | None:
    """Проверить, является ли распознанный текст командой бота."""
    normalized = text.strip().lower().rstrip(".")
    if normalized.startswith("/"):
        return normalized.split()[0]
    return _VOICE_COMMANDS.get(normalized)


@router.message(F.voice)
async def handle_voice(
    message: Message, bot: Bot, user: User, session: AsyncSession,
) -> None:
    if not settings.stt_enabled:
        await message.answer("Распознавание голоса отключено.")
        return

    if message.voice.duration > settings.whisper_max_duration:
        await message.answer(
            f"Голосовое слишком длинное (макс. {settings.whisper_max_duration} сек)."
        )
        return

    status_msg = await message.answer("Распознаю голосовое сообщение...")

    try:
        # Скачиваем OGG из Telegram
        file = await bot.get_file(message.voice.file_id)
        bio = await bot.download_file(file.file_path)
        ogg_bytes = bio.read()

        # Транскрипция
        text = await transcribe(ogg_bytes)

        if not text:
            await status_msg.edit_text(
                "Не удалось распознать речь. Попробуйте ещё раз."
            )
            return

        recognized = f"Распознано: <i>{esc(text)}</i>"

        # Детекция команд
        command = _detect_command(text)
        if command:
            await status_msg.edit_text(
                f"{recognized}\n\n"
                f"Похоже, вы хотите выполнить <b>{esc(command)}</b>.\n"
                "Отправьте команду текстом для подтверждения."
            )
            return

        # Показываем распознанный текст
        await status_msg.edit_text(recognized)

        # Лимит запросов + единая маршрутизация (chat by default, RAG when needed)
        await answer_with_quota(
            text, status_msg, user, session, limit_prefix=f"{recognized}\n\n",
        )

    except TranscriptionError as e:
        logger.warning("Ошибка транскрипции: %s", e)
        await status_msg.edit_text("Не удалось распознать голосовое сообщение. Попробуйте ещё раз.")
    except EmbeddingServiceError as e:
        logger.warning("Сервис эмбеддингов недоступен: %s", e)
        await status_msg.edit_text(
            "Сервис эмбеддингов временно недоступен. Попробуйте позже."
        )
    except Exception:
        logger.exception("Ошибка при обработке голосового сообщения")
        await status_msg.edit_text(
            "Произошла ошибка при обработке голосового сообщения. Попробуйте позже."
        )
