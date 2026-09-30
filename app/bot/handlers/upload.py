import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.filters import SUPPORTED_EXTENSIONS
from app.bot.formatting import esc
from app.config import settings
from app.core.embedder import EmbeddingServiceError
from app.core.rag_pipeline import get_pipeline
from app.db.models import User
from app.db.repositories.document import DocumentRepository

logger = logging.getLogger(__name__)
router = Router()


@router.message(F.document)
async def handle_document_upload(
    message: Message, bot: Bot, user: User, session: AsyncSession
) -> None:
    document = message.document
    if not document:
        return

    filename = document.file_name or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Проверка типа файла
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        await message.answer(
            f"Неподдерживаемый формат файла: <b>.{esc(ext)}</b>\n"
            f"Поддерживаются: {supported}"
        )
        return

    # Проверка размера
    file_size = document.file_size or 0
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_bytes:
        await message.answer(
            f"Файл слишком большой ({file_size / 1024 / 1024:.1f} МБ).\n"
            f"Максимум: {settings.max_file_size_mb} МБ."
        )
        return

    # Проверка лимита документов
    doc_repo = DocumentRepository(session)
    user_docs = await doc_repo.get_by_user(user.id)
    if len(user_docs) >= user.documents_limit:
        await message.answer(
            f"Достигнут лимит документов ({user.documents_limit}).\n"
            "Удалите старые документы, чтобы загрузить новые."
        )
        return

    status_msg = await message.answer("📥 Загружаю файл...")

    # Скачиваем файл
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name  # убираем любые ../ и /
    file_path = upload_dir / f"{user.telegram_id}_{safe_name}"

    doc_record = None
    indexed = False
    try:
        await bot.download(document, destination=file_path)

        # Создаём запись в БД
        collection_name = f"user_{user.telegram_id}"
        doc_record = await doc_repo.create(
            user_id=user.id,
            filename=filename,
            file_type=ext,
            file_size=file_size,
            qdrant_collection=collection_name,
        )

        await status_msg.edit_text("📄 Анализирую текст и создаю индекс...\nЭто может занять несколько секунд ⏳")

        async def _on_progress(batch_num: int, total_batches: int) -> None:
            try:
                pct = int(batch_num / total_batches * 100) if total_batches > 0 else 0
                filled = int(pct / 10)
                bar = "█" * filled + "░" * (10 - filled)
                await status_msg.edit_text(
                    f"Индексирую документ...  {pct}%\n[{bar}]"
                )
            except Exception:
                logger.debug("Не удалось обновить прогресс-сообщение", exc_info=True)

        # Полный RAG pipeline: парсинг → чанки → эмбеддинги → Qdrant
        pipeline = get_pipeline()
        chunk_count, full_text = await pipeline.ingest_document(
            file_path=file_path,
            file_type=ext,
            document_id=str(doc_record.id),
            filename=filename,
            user_telegram_id=user.telegram_id,
            on_progress=_on_progress,
        )

        if chunk_count == 0:
            await doc_repo.update_status(doc_record.id, "error", error_message="Пустой документ")
            await status_msg.edit_text("Документ пустой — нет текста для индексации.")
            return

        await doc_repo.update_status(doc_record.id, "ready", chunk_count=chunk_count)
        indexed = True
        # Сохраняем полный текст для diff при будущих обновлениях
        if full_text:
            await doc_repo.update_full_text(doc_record.id, full_text)

        await status_msg.edit_text(
            f"Документ <b>{esc(filename)}</b> загружен и проиндексирован.\n"
            f"Создано фрагментов: {chunk_count}\n\n"
            "Теперь вы можете задавать вопросы по этому документу."
        )
        logger.info(
            "Документ %s проиндексирован: %d чанков (user=%s)",
            filename, chunk_count, user.telegram_id,
        )

    except EmbeddingServiceError as e:
        logger.warning("Сервис эмбеддингов недоступен (upload): %s", e)
        if doc_record:
            await doc_repo.update_status(doc_record.id, "error", error_message=str(e)[:200])
        await status_msg.edit_text(
            "Сервис эмбеддингов временно недоступен. Попробуйте позже."
        )
    except Exception as e:
        logger.exception("Ошибка обработки документа %s", filename)
        # Сбой после индексации (например, при отправке уведомления) не делает
        # документ битым — он уже доступен для поиска
        if doc_record and not indexed:
            await doc_repo.update_status(
                doc_record.id, "error", error_message=str(e)
            )
        await status_msg.edit_text("Ошибка при обработке документа. Попробуйте позже.")
    finally:
        if file_path.exists():
            file_path.unlink()
