"""Безопасная подготовка текста для Telegram HTML (parse_mode=HTML по умолчанию).

Любая строка, пришедшая извне — имя файла, имя пользователя, распознанная речь,
текст документа, ответ LLM или внешнего API, — перед вставкой в HTML-сообщение
проходит через esc() или llm_to_html(). Иначе символ «<» или «&» в такой строке
ломает разметку, и Telegram отклоняет сообщение целиком.
"""

import html
import re


def esc(value: object) -> str:
    """Экранировать произвольное значение для вставки в Telegram HTML."""
    return html.escape(str(value), quote=False)


# Теги, которые Telegram понимает и которые разрешено оставлять из ответа LLM
_SAFE_TAGS = re.compile(r"<(/?)([bi]|code|pre)(/?)>", re.IGNORECASE)


def llm_to_html(text: str) -> str:
    """Ответ LLM (Markdown и/или простой HTML) → безопасный Telegram HTML.

    Сохраняет <b>, <i>, <code>, <pre>; всё остальное экранирует; конвертирует
    Markdown-разметку, которую модели выдают вопреки инструкции.
    """
    # Убираем CJK-символы, которые Qwen3 иногда вставляет в русский текст
    text = re.sub(r"[一-鿿㐀-䶿　-〿＀-￯]+", "", text)
    # Убираем артефакты LLM (внутренние теги моделей)
    text = re.sub(r"</?assistant>", "", text)
    # Нормализуем HTML-теги: <strong> -> <b>, <em> -> <i> (Telegram не поддерживает strong/em)
    text = re.sub(r"<(/?)strong\b[^>]*>", r"<\1b>", text, flags=re.IGNORECASE)
    text = re.sub(r"<(/?)em\b[^>]*>", r"<\1i>", text, flags=re.IGNORECASE)

    # Экранируем HTML-спецсимволы, сохраняя разрешённые теги
    placeholders: list[str] = []

    def _protect_tag(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00TAG{len(placeholders) - 1}\x00"

    text = _SAFE_TAGS.sub(_protect_tag, text)
    text = html.escape(text, quote=False)
    for i, tag in enumerate(placeholders):
        text = text.replace(f"\x00TAG{i}\x00", tag)

    # Блоки кода ```...```
    text = re.sub(r"```\w*\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)

    # Инлайн-код `...`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Жирный **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Курсив *...* (не внутри слов, не после *)
    text = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"<i>\1</i>", text)

    # Заголовки #{1,6} → жирный текст
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Списки: - элемент или * элемент → • элемент
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)

    return text
