from app.core.chunker import AdvancedChunker, _detect_header


def test_chunk_returns_list(sample_text: str) -> None:
    chunker = AdvancedChunker()
    chunks = chunker.chunk(sample_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_structure(sample_text: str) -> None:
    chunker = AdvancedChunker()
    chunks = chunker.chunk(sample_text)
    for chunk in chunks:
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "metadata" in chunk
        assert isinstance(chunk["text"], str)
        assert isinstance(chunk["chunk_index"], int)


def test_chunk_with_metadata(sample_text: str) -> None:
    chunker = AdvancedChunker()
    meta = {"filename": "test.pdf", "file_type": "pdf"}
    chunks = chunker.chunk(sample_text, metadata=meta)
    for chunk in chunks:
        assert chunk["metadata"]["filename"] == "test.pdf"
        assert chunk["metadata"]["file_type"] == "pdf"


def test_chunk_indexes_sequential(sample_text: str) -> None:
    chunker = AdvancedChunker()
    chunks = chunker.chunk(sample_text)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_empty_text() -> None:
    chunker = AdvancedChunker()
    chunks = chunker.chunk("")
    assert chunks == []


def test_page_markers_extracted() -> None:
    """Маркеры [Страница N] корректно парсятся в метаданные."""
    text = (
        "[Страница 1]\nТекст первой страницы, достаточно длинный для чанка. Он содержит несколько предложений.\n\n"
        "[Страница 2]\nТекст второй страницы, тоже достаточно длинный. И тут тоже несколько предложений для объёма."
    )
    chunker = AdvancedChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2

    page_numbers = [c["metadata"].get("page_number") for c in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunks_dont_cross_page_boundaries() -> None:
    """Чанки не пересекают границы страниц."""
    text = (
        "[Страница 1]\n" + "А" * 500 + "\n\n"
        "[Страница 2]\n" + "Б" * 500
    )
    chunker = AdvancedChunker()
    chunks = chunker.chunk(text)

    for chunk in chunks:
        # Каждый чанк содержит символы только одной страницы
        has_a = "А" in chunk["text"]
        has_b = "Б" in chunk["text"]
        assert not (has_a and has_b), "Чанк содержит текст с разных страниц"


def test_header_detection_markdown() -> None:
    """Markdown-заголовки обнаруживаются."""
    assert _detect_header("# Введение\nТекст") == "Введение"
    assert _detect_header("## Глава 1\nТекст") == "Глава 1"


def test_header_detection_chapter() -> None:
    """Маркеры глав обнаруживаются."""
    assert _detect_header("Глава 5. Существительные\nТекст") is not None
    assert _detect_header("Chapter 3: Introduction\nТекст") is not None


def test_header_detection_caps() -> None:
    """КАПС-строки обнаруживаются как заголовки."""
    assert _detect_header("ВВЕДЕНИЕ В ЛИНГВИСТИКУ\nТекст") is not None


def test_header_detection_none() -> None:
    """Обычный текст не распознаётся как заголовок."""
    assert _detect_header("Это обычный текст с маленькими буквами.") is None


def test_section_header_in_metadata() -> None:
    """Заголовок раздела сохраняется в метаданных."""
    text = (
        "[Страница 1]\n# Введение\n"
        "Текст введения достаточно длинный для прохождения фильтра минимального размера чанка."
    )
    chunker = AdvancedChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert chunks[0]["metadata"].get("section_header") == "Введение"


def test_no_page_markers_single_segment() -> None:
    """Текст без маркеров страниц обрабатывается как один сегмент."""
    text = "Просто текст без маркеров страниц. Достаточно длинный для чанка."
    chunker = AdvancedChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert chunks[0]["metadata"].get("page_number") is None
