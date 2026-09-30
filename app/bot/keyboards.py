from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.db.models import Document


def get_documents_keyboard(docs: list[Document]) -> InlineKeyboardMarkup:
    """Inline-кнопки с документами для выбора при удалении."""
    buttons = [
        [InlineKeyboardButton(
            text=f"{doc.filename} ({doc.chunk_count} фр.)",
            callback_data=f"delete:{doc.id}",
        )]
        for doc in docs
    ]
    buttons.append([InlineKeyboardButton(text="Отмена", callback_data="cancel_delete")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_delete_confirm_keyboard(doc_id: str) -> InlineKeyboardMarkup:
    """Подтверждение удаления документа."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, удалить", callback_data=f"confirm_delete:{doc_id}"),
                InlineKeyboardButton(text="Отмена", callback_data="cancel_delete"),
            ],
        ]
    )


def get_reset_confirm_keyboard() -> InlineKeyboardMarkup:
    """Подтверждение полной очистки."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Да, удалить всё", callback_data="confirm_reset"),
                InlineKeyboardButton(text="Отмена", callback_data="cancel_reset"),
            ],
        ]
    )


# ── Клавиатуры для обновления документов (/update) ───────────


def get_update_documents_keyboard(docs: list[Document]) -> InlineKeyboardMarkup:
    """Список документов для обновления (без бэкапов)."""
    buttons = [
        [InlineKeyboardButton(
            text=f"{doc.filename} (v{doc.version}, {doc.chunk_count} фр.)",
            callback_data=f"update_select:{doc.id}",
        )]
        for doc in docs
    ]
    buttons.append([InlineKeyboardButton(text="Отмена", callback_data="update_cancel")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_update_mode_keyboard() -> InlineKeyboardMarkup:
    """Выбор режима обновления: замена или дополнение."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Заменить документ", callback_data="update_mode:replace")],
        [InlineKeyboardButton(text="Дополнить документ", callback_data="update_mode:append")],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])


def get_backup_ask_keyboard() -> InlineKeyboardMarkup:
    """Спросить о создании бэкапа."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Да, создать бэкап", callback_data="update_backup:yes"),
            InlineKeyboardButton(text="Нет", callback_data="update_backup:no"),
        ],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])


def get_backup_name_keyboard(proposed_name: str) -> InlineKeyboardMarkup:
    """Принять предложенное имя бэкапа или ввести своё."""
    # Обрезаем имя для кнопки (макс ~40 символов)
    display_name = proposed_name if len(proposed_name) <= 40 else proposed_name[:37] + "..."
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"Принять: {display_name}",
            callback_data="update_backup_name:accept",
        )],
        [InlineKeyboardButton(text="Ввести своё имя", callback_data="update_backup_name:custom")],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])


def get_diff_choice_keyboard() -> InlineKeyboardMarkup:
    """Показать сводку изменений или применить сразу."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Показать изменения", callback_data="update_diff:show")],
        [InlineKeyboardButton(text="Применить сразу", callback_data="update_diff:apply")],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])


def get_confirm_keyboard(prefix: str) -> InlineKeyboardMarkup:
    """Универсальная клавиатура подтверждения."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Подтвердить", callback_data=f"{prefix}:confirm"),
            InlineKeyboardButton(text="Отмена", callback_data="update_cancel"),
        ],
    ])


def get_append_content_type_keyboard() -> InlineKeyboardMarkup:
    """Выбор типа контента для дополнения: текст или изображение."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Текст", callback_data="append_type:text")],
        [InlineKeyboardButton(text="Изображение", callback_data="append_type:image")],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])


def get_append_image_mode_keyboard() -> InlineKeyboardMarkup:
    """Выбор способа обработки изображения."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Tesseract OCR (текст/скан)", callback_data="append_ocr:tesseract")],
        [InlineKeyboardButton(text="Vision LLM (диаграммы)", callback_data="append_ocr:vision")],
        [InlineKeyboardButton(text="Отмена", callback_data="update_cancel")],
    ])
