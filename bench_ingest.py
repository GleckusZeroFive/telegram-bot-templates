#!/usr/bin/env python3
"""Бенчмарк полного pipeline: parse → chunk → embed."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Настраиваем logging ДО импорта app модулей
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from app.config import settings  # noqa: E402
from app.core.chunker import AdvancedChunker  # noqa: E402
from app.core.document_processor import DocumentProcessor  # noqa: E402
from app.core.embedder import LocalEmbedder  # noqa: E402

logger = logging.getLogger("bench")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bench_ingest.py <file_path>")
        print("  Supported: .pdf, .txt, .md, .docx")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        logger.error("Файл %s не найден", file_path)
        sys.exit(1)

    file_type = file_path.suffix.lstrip(".")
    if file_type not in ("pdf", "txt", "md", "docx"):
        logger.error("Неподдерживаемый тип: %s", file_type)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("БЕНЧМАРК: %s (%.0f КБ)", file_path.name, file_path.stat().st_size / 1024)
    logger.info("=" * 60)

    # === 1. ПАРСИНГ ===
    t0 = time.perf_counter()
    text = await DocumentProcessor.process(file_path, file_type)
    t_parse = time.perf_counter() - t0
    logger.info("ПАРСИНГ: %.1fс, %d символов", t_parse, len(text))

    # === 2. ЧАНКИНГ ===
    t0 = time.perf_counter()
    chunker = AdvancedChunker()
    chunks = chunker.chunk(text, metadata={"source": file_path.name})
    t_chunk = time.perf_counter() - t0
    logger.info("ЧАНКИНГ: %.2fс, %d чанков", t_chunk, len(chunks))

    if not chunks:
        logger.warning("Нет чанков — файл пустой?")
        return

    # Статистика чанков
    sizes = [len(c["text"]) for c in chunks]
    pages = {c["metadata"].get("page_number") for c in chunks if c["metadata"].get("page_number")}
    sections = {c["metadata"].get("section_header") for c in chunks if c["metadata"].get("section_header")}
    logger.info(
        "  Размеры: min=%d, max=%d, avg=%d, median=%d",
        min(sizes), max(sizes), sum(sizes) // len(sizes),
        sorted(sizes)[len(sizes) // 2],
    )
    if pages:
        logger.info("  Уникальных страниц: %d, разделов: %d", len(pages), len(sections))

    # Мелкие чанки (< 100 символов)
    small = [
        (c["metadata"].get("page_number", "?"), len(c["text"]), c["text"][:60])
        for c in chunks
        if len(c["text"]) < 100
    ]
    if small:
        logger.warning("  Мелких чанков (<100 симв): %d", len(small))
        for page, sz, preview in small[:5]:
            logger.warning("    [стр. %s] (%d симв): %r", page, sz, preview)
    else:
        logger.info("  Мелких чанков (<100 симв): 0 — отлично!")

    # === 3. ЭМБЕДДИНГ ===
    texts = [c["text"] for c in chunks]
    embedder = LocalEmbedder()
    batches = embedder._split_into_batches(texts)
    logger.info(
        "ЭМБЕДДИНГ: %d текстов → %d батчей (batch_size=%d, model=%s)",
        len(texts), len(batches), settings.embedding_batch_max_items, settings.embedding_model,
    )

    async def on_progress(batch_num: int, total: int) -> None:
        elapsed = time.perf_counter() - t0_embed
        logger.info("  Прогресс: %d/%d батчей (%.0fс)", batch_num, total, elapsed)

    t0_embed = time.perf_counter()
    embeddings = await embedder.embed_documents(texts, on_progress=on_progress)
    t_embed = time.perf_counter() - t0_embed

    logger.info("ЭМБЕДДИНГ: %.1fс (%.1f мин), %d эмбеддингов", t_embed, t_embed / 60, len(embeddings))

    # === ИТОГО ===
    t_total = t_parse + t_chunk + t_embed
    logger.info("=" * 60)
    logger.info("ИТОГО: %.1fс (%.1f мин)", t_total, t_total / 60)
    logger.info("  Парсинг:  %.1fс", t_parse)
    logger.info("  Чанкинг:  %.2fс", t_chunk)
    logger.info("  Эмбеддинг: %.1fс (%.1f мин)", t_embed, t_embed / 60)
    logger.info("  Чанков: %d, эмбеддингов: %d, dim: %d", len(chunks), len(embeddings), settings.embedding_dimension)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
