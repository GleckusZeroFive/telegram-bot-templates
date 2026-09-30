import asyncio
import logging
import uuid

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.formatting import esc
from app.bot.keyboards import (
    get_delete_confirm_keyboard,
    get_documents_keyboard,
    get_reset_confirm_keyboard,
)
from app.core.rag_pipeline import get_pipeline
from app.db.models import User
from app.db.repositories.document import DocumentRepository

logger = logging.getLogger(__name__)
router = Router()

_STATUS_EMOJI = {
    "ready": "\u2705",
    "processing": "\u23f3",
    "error": "\u274c",
}


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    return f"{size_bytes / (1024 * 1024):.1f} МБ"


# ── /docs ────────────────────────────────────────────────────────

@router.message(Command("docs"))
async def cmd_docs(message: Message, user: User, session: AsyncSession) -> None:
    try:
        doc_repo = DocumentRepository(session)
        docs = await doc_repo.get_by_user(user.id)
    except Exception:
        logger.exception("Ошибка при получении списка документов")
        await message.answer("Не удалось получить список документов. Попробуйте позже.")
        return

    if not docs:
        await message.answer("У вас пока нет загруженных документов.")
        return

    lines = ["<b>Ваши документы:</b>\n"]
    for i, doc in enumerate(docs, 1):
        emoji = _STATUS_EMOJI.get(doc.status, "❓")
        size = _format_file_size(doc.file_size)
        date = doc.created_at.strftime("%d.%m.%Y %H:%M")

        version_str = f" v{doc.version}" if doc.version > 1 else ""
        backup_str = " [бэкап]" if doc.is_backup else ""

        lines.append(
            f"{i}. {emoji} <b>{esc(doc.filename)}</b>{version_str}{backup_str}\n"
            f"   {doc.file_type.upper()} · {size} · {doc.chunk_count} фр. · {date}"
        )

    await message.answer("\n".join(lines))


# ── /delete ──────────────────────────────────────────────────────

@router.message(Command("delete"))
async def cmd_delete(message: Message, user: User, session: AsyncSession) -> None:
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)
    ready_docs = [d for d in docs if d.status in ("ready", "error")]

    if not ready_docs:
        await message.answer("Нет документов для удаления.")
        return

    await message.answer(
        "Выберите документ для удаления:",
        reply_markup=get_documents_keyboard(ready_docs),
    )


@router.callback_query(F.data.startswith("delete:"))
async def cb_delete_select(
    callback: CallbackQuery, user: User, session: AsyncSession
) -> None:
    doc_id_str = callback.data.split(":", 1)[1]

    try:
        doc_uuid = uuid.UUID(doc_id_str)
    except ValueError:
        await callback.answer("Документ не найден.", show_alert=True)
        return

    doc_repo = DocumentRepository(session)
    doc = await doc_repo.get_by_id(doc_uuid)

    if not doc or doc.user_id != user.id:
        await callback.answer("Документ не найден.", show_alert=True)
        return

    await callback.message.edit_text(
        f"Удалить документ <b>{esc(doc.filename)}</b> ({doc.chunk_count} фрагментов)?",
        reply_markup=get_delete_confirm_keyboard(doc_id_str),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("confirm_delete:"))
async def cb_delete_confirm(
    callback: CallbackQuery, user: User, session: AsyncSession
) -> None:
    doc_id_str = callback.data.split(":", 1)[1]

    try:
        doc_uuid = uuid.UUID(doc_id_str)
    except ValueError:
        await callback.answer("Документ не найден.", show_alert=True)
        return

    doc_repo = DocumentRepository(session)
    doc = await doc_repo.get_by_id(doc_uuid)

    if not doc or doc.user_id != user.id:
        await callback.answer("Документ не найден.", show_alert=True)
        return

    filename = esc(doc.filename)

    # Удаляем чанки из Qdrant
    if doc.qdrant_collection:
        try:
            # Синхронный клиент Qdrant — из event loop уводим в поток
            indexer = get_pipeline().indexer
            await asyncio.to_thread(indexer.delete_document, doc.qdrant_collection, str(doc.id))
        except Exception as e:
            logger.warning("Ошибка удаления из Qdrant: %s", e)
            await callback.message.edit_text(
                f"Не удалось удалить данные документа <b>{filename}</b> из поиска. "
                "Попробуйте позже."
            )
            await callback.answer()
            return

    # Удаляем запись из PostgreSQL
    await doc_repo.delete(doc.id)

    await callback.message.edit_text(f"Документ <b>{filename}</b> удалён.")
    await callback.answer()


@router.callback_query(F.data == "cancel_delete")
async def cb_delete_cancel(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Удаление отменено.")
    await callback.answer()


# ── /reset ───────────────────────────────────────────────────────

@router.message(Command("reset"))
async def cmd_reset(message: Message, user: User, session: AsyncSession) -> None:
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)

    if not docs:
        await message.answer("У вас нет документов — нечего удалять.")
        return

    await message.answer(
        f"Вы уверены, что хотите удалить <b>все документы</b> ({len(docs)} шт.)?\n"
        "Это действие необратимо.",
        reply_markup=get_reset_confirm_keyboard(),
    )


@router.callback_query(F.data == "confirm_reset")
async def cb_reset_confirm(
    callback: CallbackQuery, user: User, session: AsyncSession
) -> None:
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user(user.id)

    # Удаляем коллекцию из Qdrant
    collection_name = f"user_{user.telegram_id}"
    try:
        indexer = get_pipeline().indexer
        await asyncio.to_thread(indexer.delete_collection, collection_name)
    except Exception as e:
        logger.warning("Ошибка удаления коллекции из Qdrant: %s", e)
        await callback.message.edit_text(
            "Не удалось удалить данные из поиска. Попробуйте позже."
        )
        await callback.answer()
        return

    # Удаляем все документы из PostgreSQL
    for doc in docs:
        await doc_repo.delete(doc.id)

    await callback.message.edit_text(
        f"Все документы удалены ({len(docs)} шт.)."
    )
    await callback.answer()


@router.callback_query(F.data == "cancel_reset")
async def cb_reset_cancel(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Очистка отменена.")
    await callback.answer()
