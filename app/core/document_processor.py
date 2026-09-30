import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    @abstractmethod
    async def parse(self, file_path: Path) -> str:
        """Извлечь текст из файла."""


class PDFParser(BaseParser):
    """
    PDF парсер на основе PyMuPDF (pymupdf).
    - Извлекает текст со всех страниц
    - Определяет страницы, требующие OCR (мало текста)
    - Автоматически применяет Tesseract OCR для таких страниц
    - Graceful degradation: если Tesseract не установлен — работает без OCR
    """

    async def parse(self, file_path: Path) -> str:
        return await asyncio.to_thread(self._read, file_path)

    def _read(self, file_path: Path) -> str:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        try:
            pages: list[str] = []

            for i, page in enumerate(doc):
                page_num = i + 1
                text = page.get_text("text").strip()

                # Если текста мало и OCR включён — пробуем OCR
                if len(text) < settings.ocr_text_threshold and settings.ocr_enabled:
                    ocr_text = self._try_ocr(page, page_num)
                    if ocr_text:
                        text = ocr_text

                if text:
                    pages.append(f"[Страница {page_num}]\n{text}")

            return "\n\n".join(pages)
        finally:
            doc.close()

    @staticmethod
    def _try_ocr(page: "pymupdf.Page", page_num: int) -> str | None:
        """
        Попытка OCR для страницы с малым количеством текста.
        Возвращает OCR-текст или None при ошибке.
        """
        try:
            tp = page.get_textpage_ocr(
                language=settings.ocr_languages,
                dpi=300,
                full=True,
            )
            ocr_text = page.get_text("text", textpage=tp).strip()
            if len(ocr_text) >= settings.ocr_text_threshold:
                logger.info("OCR для страницы %d: %d символов", page_num, len(ocr_text))
                return ocr_text
            elif ocr_text:
                logger.debug("OCR для страницы %d: %d символов (ниже порога, отброшено)", page_num, len(ocr_text))
        except Exception as e:
            logger.warning("OCR ошибка на странице %d: %s", page_num, e)
        return None


class DOCXParser(BaseParser):
    async def parse(self, file_path: Path) -> str:
        from docx import Document

        def _read() -> str:
            doc = Document(str(file_path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        return await asyncio.to_thread(_read)


class TextParser(BaseParser):
    async def parse(self, file_path: Path) -> str:
        def _read() -> str:
            # UTF-8 с фолбэком на cp1251 (частая кодировка для русских файлов)
            for encoding in ("utf-8", "cp1251", "latin-1"):
                try:
                    return file_path.read_text(encoding=encoding)
                except (UnicodeDecodeError, ValueError):
                    continue
            logger.warning(
                "Не удалось определить кодировку %s, используется UTF-8 с заменой нечитаемых символов",
                file_path.name,
            )
            return file_path.read_text(encoding="utf-8", errors="replace")

        return await asyncio.to_thread(_read)


class MarkdownParser(TextParser):
    """MD — тот же текстовый файл."""
    pass


SUPPORTED_TYPES = {"pdf", "docx", "txt", "md"}

_PARSERS: dict[str, BaseParser] = {
    "pdf": PDFParser(),
    "docx": DOCXParser(),
    "txt": TextParser(),
    "md": MarkdownParser(),
}


class DocumentProcessor:
    @staticmethod
    async def process(file_path: Path, file_type: str) -> str:
        """Извлечь текст из файла указанного типа."""
        parser = _PARSERS.get(file_type)
        if not parser:
            raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

        text = await parser.parse(file_path)
        # Нормализация: убираем лишние пробелы
        text = "\n".join(line.rstrip() for line in text.splitlines())
        # Убираем больше двух переносов подряд
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        logger.info("Файл %s обработан: %d символов", file_path.name, len(text))
        return text.strip()
