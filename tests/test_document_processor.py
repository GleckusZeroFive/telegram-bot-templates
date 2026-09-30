from pathlib import Path

import pytest

from app.core.document_processor import SUPPORTED_TYPES, DocumentProcessor


@pytest.fixture
def tmp_txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.txt"
    p.write_text("Привет, мир! Это тестовый файл.", encoding="utf-8")
    return p


@pytest.fixture
def tmp_md_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.md"
    p.write_text("# Заголовок\n\nТекст документа.", encoding="utf-8")
    return p


@pytest.fixture
def tmp_empty_file(tmp_path: Path) -> Path:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


def test_supported_types() -> None:
    assert "pdf" in SUPPORTED_TYPES
    assert "docx" in SUPPORTED_TYPES
    assert "txt" in SUPPORTED_TYPES
    assert "md" in SUPPORTED_TYPES


@pytest.mark.asyncio
async def test_process_txt(tmp_txt_file: Path) -> None:
    processor = DocumentProcessor()
    text = await processor.process(tmp_txt_file, "txt")
    assert "Привет, мир!" in text


@pytest.mark.asyncio
async def test_process_md(tmp_md_file: Path) -> None:
    processor = DocumentProcessor()
    text = await processor.process(tmp_md_file, "md")
    assert "Заголовок" in text
    assert "Текст документа" in text


@pytest.mark.asyncio
async def test_process_empty_file(tmp_empty_file: Path) -> None:
    processor = DocumentProcessor()
    text = await processor.process(tmp_empty_file, "txt")
    assert text == ""


@pytest.mark.asyncio
async def test_unsupported_type(tmp_txt_file: Path) -> None:
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="Неподдерживаемый"):
        await processor.process(tmp_txt_file, "xyz")
