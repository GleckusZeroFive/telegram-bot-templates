"""Обработка изображений: OCR (Tesseract) и Vision LLM (модель должна поддерживать картинки)."""

import asyncio
import base64
import logging
from pathlib import Path

from app.config import settings
from app.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)


async def ocr_tesseract(image_path: Path) -> str:
    """Извлечь текст из изображения через Tesseract OCR (PyMuPDF)."""

    def _run() -> str:
        import pymupdf

        doc = pymupdf.open(str(image_path))
        try:
            page = doc[0]
            tp = page.get_textpage_ocr(
                language=settings.ocr_languages,
                dpi=300,
                full=True,
            )
            return page.get_text("text", textpage=tp).strip()
        finally:
            doc.close()

    text = await asyncio.to_thread(_run)
    logger.info("OCR Tesseract: %d символов из %s", len(text), image_path.name)
    return text


async def describe_with_vision_llm(image_path: Path) -> str:
    """Проанализировать изображение через Vision LLM (сконфигурированная модель с поддержкой изображений)."""
    # Читаем изображение и кодируем в base64
    image_bytes = await asyncio.to_thread(image_path.read_bytes)
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    # Определяем MIME-тип по расширению
    suffix = image_path.suffix.lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_types.get(suffix, "image/jpeg")

    provider = get_llm_provider()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Опиши содержимое этого изображения максимально подробно на русском языке. "
                        "Если на изображении есть текст — перепиши его дословно. "
                        "Если есть таблицы, диаграммы или схемы — передай их содержимое текстом."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                },
            ],
        }
    ]

    text = await provider.generate(messages, max_tokens=2048)
    logger.info("Vision LLM (%s): %d символов из %s", provider.model, len(text), image_path.name)
    return text
