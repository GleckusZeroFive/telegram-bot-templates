from __future__ import annotations

import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

_PAGE_PATTERN = re.compile(r"\[Страница (\d+)\]")
# Регистронезависим только маркер главы. Глобальный IGNORECASE превращал ветку
# КАПС-строк в «любая строка из букв», и заголовком считался обычный текст.
_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"#{1,6}\s+.+|"                                                               # Markdown headers
    r"(?i:глава|раздел|часть|chapter|section|part)\s+[\dIVXLCDM]+[.:)]*\s*.*|"    # Явные маркеры глав
    r"[А-ЯЁA-Z][А-ЯЁA-Z\s\d.]{3,80}$"                                             # КАПС-строки
    r")",
    re.MULTILINE,
)


class _TextSegment:
    """Сегмент текста с метаданными о структуре документа."""
    __slots__ = ("page_number", "section_header", "text")

    def __init__(
        self,
        text: str,
        page_number: int | None = None,
        section_header: str | None = None,
    ) -> None:
        self.text = text
        self.page_number = page_number
        self.section_header = section_header


class TextChunker:
    """Базовый чанкер (backward compatibility)."""

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """
        Разбить текст на чанки с метаданными.

        Returns:
            [{"text": str, "chunk_index": int, "metadata": dict}, ...]
        """
        chunks = self._splitter.split_text(text)
        result = [
            {
                "text": chunk,
                "chunk_index": i,
                "metadata": {**(metadata or {}), "chunk_index": i},
            }
            for i, chunk in enumerate(chunks)
        ]
        logger.info("Текст разбит на %d чанков (chunk_size=%d)", len(result), settings.chunk_size)
        return result


class AdvancedChunker:
    """
    Продвинутый чанкер с учётом структуры документа.
    - Разбивает текст на сегменты по маркерам [Страница N]
    - Детектирует заголовки разделов
    - Чанки НЕ пересекают границы страниц
    - Улучшенные разделители для предложений
    - Метаданные: page_number, section_header
    """

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list[dict]:
        """
        Разбить текст на качественные чанки с метаданными.

        Returns:
            [{"text": str, "chunk_index": int, "metadata": dict}, ...]
        """
        if not text.strip():
            return []

        segments = self._extract_segments(text)
        result: list[dict] = []
        chunk_index = 0

        for segment in segments:
            if not segment.text.strip():
                continue

            sub_chunks = self._splitter.split_text(segment.text)

            for sub_chunk in sub_chunks:
                # Фильтрация мусорных чанков (колонтитулы, артефакты OCR)
                if len(sub_chunk.strip()) < settings.min_chunk_size:
                    continue

                chunk_meta = {
                    **(metadata or {}),
                    "chunk_index": chunk_index,
                }
                if segment.page_number is not None:
                    chunk_meta["page_number"] = segment.page_number
                if segment.section_header:
                    chunk_meta["section_header"] = segment.section_header

                result.append({
                    "text": sub_chunk,
                    "chunk_index": chunk_index,
                    "metadata": chunk_meta,
                })
                chunk_index += 1

        logger.info(
            "Текст разбит на %d чанков (chunk_size=%d, сегментов=%d)",
            len(result), settings.chunk_size, len(segments),
        )
        return result

    @staticmethod
    def _extract_segments(text: str) -> list[_TextSegment]:
        """
        Разбить текст на сегменты по маркерам [Страница N].
        Для каждого сегмента определить номер страницы и заголовок раздела.
        Если маркеров нет — весь текст = один сегмент.
        """
        # Препроцессинг: убрать декоративные разделители (═══...═══)
        text = re.sub(r"\n[═=]{5,}\n", "\n\n", text)

        page_markers = list(_PAGE_PATTERN.finditer(text))

        if not page_markers:
            header = _detect_header(text)
            return [_TextSegment(text=text, section_header=header)]

        segments: list[_TextSegment] = []
        current_section_header: str | None = None

        # Текст до первого маркера (если есть)
        if page_markers[0].start() > 0:
            prefix = text[:page_markers[0].start()].strip()
            if prefix:
                header = _detect_header(prefix)
                if header:
                    current_section_header = header
                segments.append(_TextSegment(
                    text=prefix,
                    section_header=current_section_header,
                ))

        for idx, marker in enumerate(page_markers):
            page_num = int(marker.group(1))

            start = marker.end()
            end = page_markers[idx + 1].start() if idx + 1 < len(page_markers) else len(text)
            page_text = text[start:end].strip()

            if not page_text:
                continue

            header = _detect_header(page_text)
            if header:
                current_section_header = header

            segments.append(_TextSegment(
                text=page_text,
                page_number=page_num,
                section_header=current_section_header,
            ))

        return segments


def _detect_header(text: str) -> str | None:
    """Обнаружить заголовок раздела в первых строках текста."""
    lines = text.strip().splitlines()[:3]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = _HEADER_PATTERN.match(line)
        if match:
            header = line.lstrip("#").strip()
            if len(header) > 100:
                header = header[:100] + "..."
            return header
    return None
