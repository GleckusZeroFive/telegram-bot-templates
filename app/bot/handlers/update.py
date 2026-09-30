"""Обработчик /update — обновление документов (замена и дополнение)."""

import asyncio
import contextlib
import logging
import uuid
from datetime import datetime
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.filters import SUPPORTED_EXTENSIONS
from app.bot.formatting import esc, llm_to_html
from app.bot.keyboards import (
    get_append_content_type_keyboard,
    get_append_image_mode_keyboard,
    get_backup_ask_keyboard,
    get_backup_name_keyboard,
    get_confirm_keyboard,
    get_diff_choice_keyboard,
    get_update_documents_keyboard,
    get_update_mode_keyboard,
)
from app.bot.states import UpdateDocumentFSM
from app.config import settings
from app.core.embedder import EmbeddingServiceError
from app.core.rag_pipeline import get_pipeline
from app.db.models import User
from app.db.repositories.document import DocumentRepository
from app.llm.provider import LLMError

logger = logging.getLogger(__name__)
router = Router()


# ── /update — точка входа ────────────────────────────────────


@router.message(Command("update"))
async def cmd_update(
    message: Message, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Показать список документов для обновления."""
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.get_by_user_non_backup(user.id)
    ready_docs = [d for d in docs if d.status == "ready"]

    if not ready_docs:
        await message.answer("Нет документов для обновления.")
        return

    await message.answer(
        "Выберите документ для обновления:",
        reply_markup=get_update_documents_keyboard(ready_docs),
    )
    await state.set_state(UpdateDocumentFSM.choose_document)


# ── Выбор документа и режима ─────────────────────────────────


@router.callback_query(UpdateDocumentFSM.choose_document, F.data.startswith("update_select:"))
async def cb_update_select(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Документ выбран — спросить режим."""
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

    await state.update_data(doc_id=doc_id_str, doc_filename=doc.filename)

    await callback.message.edit_text(
        f"Документ: <b>{esc(doc.filename)}</b> (v{doc.version}, {doc.chunk_count} фрагментов)\n\n"
        "Что вы хотите сделать?",
        reply_markup=get_update_mode_keyboard(),
    )
    await state.set_state(UpdateDocumentFSM.choose_mode)
    await callback.answer()


@router.callback_query(UpdateDocumentFSM.choose_mode, F.data.startswith("update_mode:"))
async def cb_update_mode(callback: CallbackQuery, state: FSMContext) -> None:
    """Выбран режим: замена или дополнение."""
    mode = callback.data.split(":", 1)[1]
    await state.update_data(mode=mode)

    if mode == "replace":
        await callback.message.edit_text(
            "Создать резервную копию текущей версии перед заменой?",
            reply_markup=get_backup_ask_keyboard(),
        )
        await state.set_state(UpdateDocumentFSM.replace_backup_ask)
    else:
        await callback.message.edit_text(
            "Какой контент вы хотите добавить?",
            reply_markup=get_append_content_type_keyboard(),
        )
        await state.set_state(UpdateDocumentFSM.append_content_type)

    await callback.answer()


# ══════════════════════════════════════════════════════════════
#  ВЕТКА ЗАМЕНЫ
# ══════════════════════════════════════════════════════════════


@router.callback_query(UpdateDocumentFSM.replace_backup_ask, F.data.startswith("update_backup:"))
async def cb_replace_backup(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Ответ на вопрос о бэкапе (ветка замены)."""
    want_backup = callback.data.split(":", 1)[1] == "yes"
    await state.update_data(want_backup=want_backup)

    if want_backup:
        data = await state.get_data()
        doc_repo = DocumentRepository(session)

        # Проверка лимита документов (бэкап занимает слот)
        user_docs = await doc_repo.get_by_user(user.id)
        if len(user_docs) >= user.documents_limit:
            await callback.message.edit_text(
                f"Невозможно создать бэкап: достигнут лимит документов ({user.documents_limit}).\n"
                "Удалите ненужные документы или продолжите без бэкапа.",
                reply_markup=get_backup_ask_keyboard(),
            )
            await callback.answer()
            return

        # Предложить имя для бэкапа
        doc = await doc_repo.get_by_id(uuid.UUID(data["doc_id"]))
        date_str = datetime.now().strftime("%Y%m%d")
        name_parts = doc.filename.rsplit(".", 1)
        if len(name_parts) == 2:
            proposed = f"{name_parts[0]}_backup_{date_str}.{name_parts[1]}"
        else:
            proposed = f"{doc.filename}_backup_{date_str}"

        await state.update_data(proposed_backup_name=proposed)
        await callback.message.edit_text(
            f"Имя бэкапа: <b>{esc(proposed)}</b>",
            reply_markup=get_backup_name_keyboard(proposed),
        )
        await state.set_state(UpdateDocumentFSM.replace_backup_name)
    else:
        await callback.message.edit_text("Отправьте новый файл для замены документа.")
        await state.set_state(UpdateDocumentFSM.replace_upload)

    await callback.answer()


@router.callback_query(UpdateDocumentFSM.replace_backup_name, F.data.startswith("update_backup_name:"))
async def cb_replace_backup_name(callback: CallbackQuery, state: FSMContext) -> None:
    """Принять имя бэкапа или запросить пользовательское."""
    choice = callback.data.split(":", 1)[1]

    if choice == "accept":
        data = await state.get_data()
        await state.update_data(backup_name=data["proposed_backup_name"])
        await callback.message.edit_text("Отправьте новый файл для замены документа.")
        await state.set_state(UpdateDocumentFSM.replace_upload)
    else:
        await callback.message.edit_text("Введите имя для бэкапа:")
        # Остаёмся в replace_backup_name — следующее текстовое сообщение обработается

    await callback.answer()


@router.message(UpdateDocumentFSM.replace_backup_name, F.text)
async def handle_replace_custom_backup_name(message: Message, state: FSMContext) -> None:
    """Пользователь ввёл своё имя бэкапа."""
    name = message.text.strip()
    if not name or len(name) > 200:
        await message.answer("Имя должно быть от 1 до 200 символов. Попробуйте ещё раз.")
        return

    await state.update_data(backup_name=name)
    await message.answer("Отправьте новый файл для замены документа.")
    await state.set_state(UpdateDocumentFSM.replace_upload)


@router.message(UpdateDocumentFSM.replace_upload, F.document)
async def handle_replace_upload(
    message: Message, bot: Bot, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Пользователь загрузил файл для замены."""
    document = message.document
    if not document:
        return

    filename = document.file_name or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Валидация
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        await message.answer(f"Неподдерживаемый формат: <b>.{esc(ext)}</b>\nПоддерживаются: {supported}")
        return

    file_size = document.file_size or 0
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_bytes:
        await message.answer(f"Файл слишком большой ({file_size / 1024 / 1024:.1f} МБ).")
        return

    status_msg = await message.answer("Обрабатываю новый файл...")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{user.telegram_id}_update_{Path(filename).name}"

    try:
        await bot.download(document, destination=file_path)

        # Извлечь текст из нового файла
        pipeline = get_pipeline()
        new_text = await pipeline.processor.process(file_path, ext)

        if not new_text.strip():
            await status_msg.edit_text("Новый файл пустой — нет текста.")
            if file_path.exists():
                file_path.unlink()
            return

        await state.update_data(
            new_file_path=str(file_path),
            new_file_type=ext,
            new_filename=filename,
            new_file_size=file_size,
            new_text=new_text[:50000],  # Ограничение для FSM storage
        )

        # Проверить, есть ли старый текст для diff
        data = await state.get_data()
        doc_repo = DocumentRepository(session)
        doc = await doc_repo.get_by_id(uuid.UUID(data["doc_id"]))

        if doc and doc.full_text:
            await status_msg.edit_text(
                "Файл загружен. Хотите увидеть сводку изменений перед применением?",
                reply_markup=get_diff_choice_keyboard(),
            )
            await state.set_state(UpdateDocumentFSM.replace_diff_ask)
        else:
            await status_msg.edit_text(
                "Файл загружен. Старая версия не сохранена для сравнения.\n"
                "Применить замену?",
                reply_markup=get_confirm_keyboard("update_replace"),
            )
            await state.set_state(UpdateDocumentFSM.replace_confirm)

    except Exception:
        logger.exception("Ошибка обработки файла замены")
        await status_msg.edit_text("Ошибка обработки файла. Попробуйте позже.")
        if file_path.exists():
            file_path.unlink()


@router.callback_query(UpdateDocumentFSM.replace_diff_ask, F.data.startswith("update_diff:"))
async def cb_replace_diff_choice(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Показать diff или применить сразу."""
    choice = callback.data.split(":", 1)[1]

    if choice == "show":
        data = await state.get_data()
        doc_repo = DocumentRepository(session)
        doc = await doc_repo.get_by_id(uuid.UUID(data["doc_id"]))

        await callback.message.edit_text("Генерирую сводку изменений...")

        try:
            pipeline = get_pipeline()
            diff_summary = await pipeline.generate_diff_summary(
                doc.full_text or "", data.get("new_text", "")
            )
            await callback.message.edit_text(
                f"<b>Изменения:</b>\n\n{llm_to_html(diff_summary)}\n\nПрименить замену?",
                reply_markup=get_confirm_keyboard("update_replace"),
            )
        except (LLMError, Exception):
            logger.exception("Ошибка генерации diff")
            await callback.message.edit_text(
                "Не удалось сгенерировать сводку. Применить замену?",
                reply_markup=get_confirm_keyboard("update_replace"),
            )

        await state.set_state(UpdateDocumentFSM.replace_confirm)
    else:
        # Применить сразу
        await callback.answer()
        await _execute_replace(callback.message, user, session, state)
        return

    await callback.answer()


@router.callback_query(UpdateDocumentFSM.replace_confirm, F.data == "update_replace:confirm")
async def cb_replace_confirm(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Подтверждение замены."""
    await callback.answer()
    await _execute_replace(callback.message, user, session, state)


async def _execute_replace(
    message: Message, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Основная логика замены: бэкап → удаление старых чанков → загрузка новых."""
    data = await state.get_data()
    doc_id = uuid.UUID(data["doc_id"])
    doc_repo = DocumentRepository(session)
    doc = await doc_repo.get_by_id(doc_id)

    if not doc:
        await message.edit_text("Документ не найден.")
        await state.clear()
        return

    file_path = Path(data["new_file_path"])

    try:
        await message.edit_text("Выполняю замену...")

        pipeline = get_pipeline()
        collection_name = f"user_{user.telegram_id}"

        # 1. Создать бэкап если запрошен
        if data.get("want_backup"):
            backup_name = data["backup_name"]
            backup_doc = await doc_repo.create_backup(doc, backup_name)

            copied = await asyncio.to_thread(
                pipeline.indexer.copy_document_points,
                collection_name,
                str(doc.id),
                str(backup_doc.id),
                backup_name,
            )
            logger.info("Бэкап создан: %s (%d чанков)", backup_name, copied)

        # 2. Замена: удалить старые чанки + загрузить новые
        await message.edit_text("Индексирую новый документ...")

        async def _on_progress(batch_num: int, total_batches: int) -> None:
            # Прогресс необязателен: сбой обновления не должен прерывать индексацию
            with contextlib.suppress(Exception):
                await message.edit_text(
                    f"Индексирую новый документ... ({batch_num}/{total_batches})"
                )

        chunk_count, full_text = await pipeline.replace_document(
            file_path=file_path,
            file_type=data["new_file_type"],
            document_id=str(doc.id),
            filename=data["new_filename"],
            user_telegram_id=user.telegram_id,
            on_progress=_on_progress,
        )

        # 3. Обновить БД
        await doc_repo.update_status(doc.id, "ready", chunk_count=chunk_count)
        await doc_repo.update_full_text(doc.id, full_text)
        await doc_repo.increment_version(doc.id)

        backup_msg = ""
        if data.get("want_backup"):
            backup_msg = f"\nБэкап: <b>{esc(data['backup_name'])}</b>"

        await message.edit_text(
            f"Документ <b>{esc(doc.filename)}</b> обновлён (v{doc.version + 1}).\n"
            f"Фрагментов: {chunk_count}{backup_msg}"
        )

    except EmbeddingServiceError as e:
        logger.warning("Сервис эмбеддингов недоступен (replace): %s", e)
        await message.edit_text(
            "Сервис эмбеддингов временно недоступен. Попробуйте позже."
        )
    except Exception:
        logger.exception("Ошибка замены документа")
        await message.edit_text("Ошибка при замене документа. Попробуйте позже.")
    finally:
        if file_path.exists():
            file_path.unlink()
        await state.clear()


# ══════════════════════════════════════════════════════════════
#  ВЕТКА ДОПОЛНЕНИЯ
# ══════════════════════════════════════════════════════════════


@router.callback_query(UpdateDocumentFSM.append_content_type, F.data.startswith("append_type:"))
async def cb_append_type(callback: CallbackQuery, state: FSMContext) -> None:
    """Выбор типа контента: текст или изображение."""
    content_type = callback.data.split(":", 1)[1]
    await state.update_data(append_content_type=content_type)

    if content_type == "text":
        await callback.message.edit_text(
            "Отправьте текст, который нужно добавить к документу."
        )
        await state.set_state(UpdateDocumentFSM.append_input)
    else:
        await callback.message.edit_text(
            "Выберите способ обработки изображения:",
            reply_markup=get_append_image_mode_keyboard(),
        )
        await state.set_state(UpdateDocumentFSM.append_image_mode)

    await callback.answer()


@router.callback_query(UpdateDocumentFSM.append_image_mode, F.data.startswith("append_ocr:"))
async def cb_append_ocr_mode(callback: CallbackQuery, state: FSMContext) -> None:
    """Выбор OCR/Vision режима."""
    ocr_mode = callback.data.split(":", 1)[1]
    await state.update_data(ocr_mode=ocr_mode)

    await callback.message.edit_text("Отправьте изображение.")
    await state.set_state(UpdateDocumentFSM.append_input)
    await callback.answer()


@router.message(UpdateDocumentFSM.append_input, F.text)
async def handle_append_text(message: Message, state: FSMContext) -> None:
    """Пользователь отправил текст для дополнения."""
    data = await state.get_data()

    # Если ожидаем изображение, а пришёл текст
    if data.get("append_content_type") == "image":
        await message.answer("Ожидается изображение. Отправьте фото или нажмите Отмена.")
        return

    text = message.text.strip()
    if not text:
        await message.answer("Текст пуст. Отправьте текст для добавления.")
        return

    await state.update_data(append_text=text)

    # Показать превью
    preview = text[:500] + ("..." if len(text) > 500 else "")
    await message.answer(
        f"<b>Текст для добавления:</b>\n\n{esc(preview)}\n\n"
        "Создать бэкап перед добавлением?",
        reply_markup=get_backup_ask_keyboard(),
    )
    await state.set_state(UpdateDocumentFSM.append_backup_ask)


@router.message(UpdateDocumentFSM.append_input, F.photo)
async def handle_append_image(
    message: Message, bot: Bot, user: User, state: FSMContext
) -> None:
    """Пользователь отправил изображение для дополнения."""
    data = await state.get_data()
    ocr_mode = data.get("ocr_mode", "tesseract")

    status_msg = await message.answer("Обрабатываю изображение...")

    # Скачать изображение (максимальное разрешение)
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{user.telegram_id}_append_{photo.file_id}.jpg"

    try:
        await bot.download_file(file.file_path, destination=image_path)

        from app.core.image_processor import describe_with_vision_llm, ocr_tesseract

        if ocr_mode == "tesseract":
            text = await ocr_tesseract(image_path)
        else:
            text = await describe_with_vision_llm(image_path)

        if not text.strip():
            await status_msg.edit_text("Не удалось извлечь текст из изображения.")
            return

        await state.update_data(append_text=text)

        preview = text[:500] + ("..." if len(text) > 500 else "")
        await status_msg.edit_text(
            f"<b>Извлечённый текст:</b>\n\n{esc(preview)}\n\n"
            "Создать бэкап перед добавлением?",
            reply_markup=get_backup_ask_keyboard(),
        )
        await state.set_state(UpdateDocumentFSM.append_backup_ask)

    except LLMError as e:
        logger.warning("Vision LLM ошибка: %s", e)
        await status_msg.edit_text(
            "Не удалось распознать изображение через модель. "
            "Попробуйте режим OCR или повторите позже."
        )
        await state.clear()
    except Exception:
        logger.exception("Ошибка обработки изображения")
        await status_msg.edit_text("Ошибка обработки изображения. Попробуйте позже.")
        await state.clear()
    finally:
        if image_path.exists():
            image_path.unlink()


# ── Бэкап для дополнения (аналогично замене) ─────────────────


@router.callback_query(UpdateDocumentFSM.append_backup_ask, F.data.startswith("update_backup:"))
async def cb_append_backup(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Ответ на вопрос о бэкапе (ветка дополнения)."""
    want_backup = callback.data.split(":", 1)[1] == "yes"
    await state.update_data(want_backup=want_backup)

    if want_backup:
        data = await state.get_data()
        doc_repo = DocumentRepository(session)

        user_docs = await doc_repo.get_by_user(user.id)
        if len(user_docs) >= user.documents_limit:
            await callback.message.edit_text(
                f"Невозможно создать бэкап: достигнут лимит документов ({user.documents_limit}).\n"
                "Удалите ненужные документы или продолжите без бэкапа.",
                reply_markup=get_backup_ask_keyboard(),
            )
            await callback.answer()
            return

        doc = await doc_repo.get_by_id(uuid.UUID(data["doc_id"]))
        date_str = datetime.now().strftime("%Y%m%d")
        name_parts = doc.filename.rsplit(".", 1)
        if len(name_parts) == 2:
            proposed = f"{name_parts[0]}_backup_{date_str}.{name_parts[1]}"
        else:
            proposed = f"{doc.filename}_backup_{date_str}"

        await state.update_data(proposed_backup_name=proposed)
        await callback.message.edit_text(
            f"Имя бэкапа: <b>{esc(proposed)}</b>",
            reply_markup=get_backup_name_keyboard(proposed),
        )
        await state.set_state(UpdateDocumentFSM.append_backup_name)
    else:
        await callback.message.edit_text(
            "Добавить контент к документу?",
            reply_markup=get_confirm_keyboard("update_append"),
        )
        await state.set_state(UpdateDocumentFSM.append_confirm)

    await callback.answer()


@router.callback_query(UpdateDocumentFSM.append_backup_name, F.data.startswith("update_backup_name:"))
async def cb_append_backup_name(callback: CallbackQuery, state: FSMContext) -> None:
    """Принять имя бэкапа или запросить пользовательское (ветка дополнения)."""
    choice = callback.data.split(":", 1)[1]

    if choice == "accept":
        data = await state.get_data()
        await state.update_data(backup_name=data["proposed_backup_name"])
        await callback.message.edit_text(
            "Добавить контент к документу?",
            reply_markup=get_confirm_keyboard("update_append"),
        )
        await state.set_state(UpdateDocumentFSM.append_confirm)
    else:
        await callback.message.edit_text("Введите имя для бэкапа:")

    await callback.answer()


@router.message(UpdateDocumentFSM.append_backup_name, F.text)
async def handle_append_custom_backup_name(message: Message, state: FSMContext) -> None:
    """Пользователь ввёл своё имя бэкапа (ветка дополнения)."""
    name = message.text.strip()
    if not name or len(name) > 200:
        await message.answer("Имя должно быть от 1 до 200 символов. Попробуйте ещё раз.")
        return

    await state.update_data(backup_name=name)
    await message.answer(
        "Добавить контент к документу?",
        reply_markup=get_confirm_keyboard("update_append"),
    )
    await state.set_state(UpdateDocumentFSM.append_confirm)


@router.callback_query(UpdateDocumentFSM.append_confirm, F.data == "update_append:confirm")
async def cb_append_confirm(
    callback: CallbackQuery, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Подтверждение дополнения."""
    await callback.answer()
    await _execute_append(callback.message, user, session, state)


async def _execute_append(
    message: Message, user: User, session: AsyncSession, state: FSMContext
) -> None:
    """Основная логика дополнения: бэкап → чанкинг → эмбеддинг → индексация."""
    data = await state.get_data()
    doc_id = uuid.UUID(data["doc_id"])
    doc_repo = DocumentRepository(session)
    doc = await doc_repo.get_by_id(doc_id)

    if not doc:
        await message.edit_text("Документ не найден.")
        await state.clear()
        return

    try:
        await message.edit_text("Добавляю контент...")

        pipeline = get_pipeline()
        collection_name = f"user_{user.telegram_id}"

        # 1. Бэкап если запрошен
        if data.get("want_backup"):
            backup_name = data["backup_name"]
            backup_doc = await doc_repo.create_backup(doc, backup_name)

            copied = await asyncio.to_thread(
                pipeline.indexer.copy_document_points,
                collection_name,
                str(doc.id),
                str(backup_doc.id),
                backup_name,
            )
            logger.info("Бэкап создан: %s (%d чанков)", backup_name, copied)

        # 2. Дополнение
        async def _on_progress(batch_num: int, total_batches: int) -> None:
            # Прогресс необязателен: сбой обновления не должен прерывать индексацию
            with contextlib.suppress(Exception):
                await message.edit_text(
                    f"Индексирую новый контент... ({batch_num}/{total_batches})"
                )

        new_chunks = await pipeline.append_text_to_document(
            text=data["append_text"],
            document_id=str(doc.id),
            filename=doc.filename,
            user_telegram_id=user.telegram_id,
            on_progress=_on_progress,
        )

        # 3. Обновить БД
        new_total = doc.chunk_count + new_chunks
        await doc_repo.update_chunk_count(doc.id, new_total)
        await doc_repo.increment_version(doc.id)

        # Обновить full_text
        old_text = doc.full_text or ""
        new_text = old_text + "\n\n" + data["append_text"]
        await doc_repo.update_full_text(doc.id, new_text)

        backup_msg = ""
        if data.get("want_backup"):
            backup_msg = f"\nБэкап: <b>{esc(data['backup_name'])}</b>"

        await message.edit_text(
            f"Добавлено {new_chunks} новых фрагментов к <b>{esc(doc.filename)}</b> (v{doc.version + 1}).\n"
            f"Всего фрагментов: {new_total}{backup_msg}"
        )

    except EmbeddingServiceError as e:
        logger.warning("Сервис эмбеддингов недоступен (append): %s", e)
        await message.edit_text(
            "Сервис эмбеддингов временно недоступен. Попробуйте позже."
        )
    except Exception:
        logger.exception("Ошибка дополнения документа")
        await message.edit_text("Ошибка при добавлении контента. Попробуйте позже.")
    finally:
        await state.clear()


# ══════════════════════════════════════════════════════════════
#  ОТМЕНА (любой шаг)
# ══════════════════════════════════════════════════════════════


@router.callback_query(F.data == "update_cancel")
async def cb_update_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    """Отмена обновления на любом шаге."""
    data = await state.get_data()

    # Удалить временный файл если он был загружен
    file_path_str = data.get("new_file_path")
    if file_path_str:
        file_path = Path(file_path_str)
        if file_path.exists():
            file_path.unlink()

    await state.clear()
    await callback.message.edit_text("Обновление отменено.")
    await callback.answer()
