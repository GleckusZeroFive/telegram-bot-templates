from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.commands import format_commands_for_help
from app.bot.formatting import esc
from app.core.conversation import clear_context
from app.db.models import User
from app.db.repositories.document import DocumentRepository
from app.presets import get_preset

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, user: User) -> None:
    preset = get_preset()
    # Имя из профиля Telegram попадает в HTML-шаблон приветствия
    name = esc(user.first_name or user.username or "пользователь")

    if preset.messages.start:
        text = preset.messages.start.format(
            user_name=name,
            bot_name=preset.name,
        )
    else:
        text = (
            f"Привет, <b>{name}</b>! Я — <b>{preset.name}</b>, AI-ассистент по документам.\n\n"
            "Отправь мне файл (PDF, DOCX, TXT, MD) — я проиндексирую его "
            "и смогу отвечать на вопросы по содержимому.\n\n"
            "/help — все команды\n"
            "/about — о боте"
        )
    await message.answer(text)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    preset = get_preset()

    if preset.messages.help:
        text = preset.messages.help.format(
            commands_help=format_commands_for_help(),
            bot_name=preset.name,
        )
    else:
        text = (
            f"<b>Команды:</b>\n"
            f"{format_commands_for_help()}\n\n"
            "<b>Загрузка документов:</b>\n"
            "Отправь файл (PDF, DOCX, TXT, MD) — я его проиндексирую.\n\n"
            "<b>Вопросы:</b>\n"
            "Отправь текстовое сообщение — я поищу ответ.\n"
            "Уточняющие вопросы учитывают контекст диалога.\n\n"
            "<b>Голосовые сообщения:</b>\n"
            "Отправь голосовое — я распознаю речь и поищу ответ."
        )
    await message.answer(text)


@router.message(Command("about"))
async def cmd_about(message: Message) -> None:
    preset = get_preset()

    if preset.messages.about:
        text = preset.messages.about.format(bot_name=preset.name)
    else:
        text = (
            f"<b>{preset.name} — AI-ассистент по документам</b>\n\n"
            "Загружайте документы (PDF, DOCX, TXT, MD) и задавайте вопросы.\n\n"
            "<b>Возможности:</b>\n"
            "• Умный поиск по загруженным документам\n"
            "• Голосовые сообщения\n"
            "• Контекстный диалог с памятью\n\n"
            "Версия: 1.0"
        )
    await message.answer(text)


@router.message(Command("new"))
async def cmd_new(message: Message, user: User, **kwargs) -> None:
    clear_context(user.telegram_id)
    await message.answer("Контекст диалога сброшен. Можете задать новый вопрос.")


@router.message(Command("stats"))
async def cmd_stats(message: Message, user: User, session: AsyncSession) -> None:
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)
    ready_count = sum(1 for d in docs if d.status == "ready")
    total_chunks = sum(d.chunk_count for d in docs)
    registered = user.created_at.strftime("%d.%m.%Y")

    tier_names = {"free": "Free", "pro": "Pro", "admin": "Admin"}
    tier_name = tier_names.get(user.tier, user.tier)

    tier_line = f"Тариф: <b>{tier_name}</b>"
    if user.tier != "free" and user.tier_expires_at:
        tier_line += f" (до {user.tier_expires_at.strftime('%d.%m.%Y')})"

    is_unlimited = user.documents_limit >= 999_000
    if is_unlimited:
        docs_line = f"Документов: {len(docs)} (готовых: {ready_count})"
        queries_line = f"Запросов сегодня: {user.queries_today}"
    else:
        docs_line = f"Документов: {len(docs)} из {user.documents_limit} (готовых: {ready_count})"
        queries_line = f"Запросов сегодня: {user.queries_today} из {user.queries_limit}"

    # Progress bar for queries
    if not is_unlimited and user.queries_limit > 0:
        ratio = user.queries_today / user.queries_limit
        filled = int(ratio * 10)
        bar = "█" * filled + "░" * (10 - filled)
        queries_line += f"  [{bar}]"

    await message.answer(
        f"\U0001f4ca <b>Статистика</b>\n\n"
        f"\U0001f464 {tier_line}\n"
        f"\U0001f4c4 {docs_line}\n"
        f"\U0001f9e9 Фрагментов: {total_chunks}\n"
        f"\U0001f4ac {queries_line}\n"
        f"\U0001f4c5 Зарегистрирован: {registered}"
    )
