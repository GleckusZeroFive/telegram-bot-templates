"""Регрессии гибридного поиска на in-memory Qdrant (без внешнего сервера)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient

from app.core.indexer import QdrantIndexer
from app.core.retriever import QdrantRetriever

_DIM = 4
_COLLECTION = "user_42"


def _vec(*values: float) -> list[float]:
    # Эмбеддинги всегда float; локальный Qdrant на int-векторе падает при нормировке
    return [float(v) for v in values]


@pytest.fixture
def qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def indexer(qdrant) -> QdrantIndexer:
    idx = QdrantIndexer.__new__(QdrantIndexer)
    idx.client = qdrant
    idx.dimension = _DIM
    return idx


@pytest.fixture
def retriever(qdrant, monkeypatch) -> QdrantRetriever:
    monkeypatch.setattr("app.core.retriever.settings.retriever_score_threshold", 0.5)
    monkeypatch.setattr("app.core.retriever.QdrantClient", MagicMock(return_value=qdrant))
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=_vec(1, 0, 0, 0))
    return QdrantRetriever(embedder)


def _index(indexer: QdrantIndexer, texts_and_vectors: list[tuple[str, list[float]]]) -> None:
    chunks = [
        {"text": text, "chunk_index": i, "metadata": {"chunk_index": i}}
        for i, (text, _) in enumerate(texts_and_vectors)
    ]
    indexer.index_chunks(
        collection_name=_COLLECTION,
        chunks=chunks,
        embeddings=[vector for _, vector in texts_and_vectors],
        document_id="doc-1",
        filename="policy.txt",
    )


@pytest.mark.asyncio
async def test_search_before_first_upload_does_not_break_later_searches(retriever, indexer):
    """Раньше «коллекции нет» кэшировалось навсегда: после загрузки документа
    поиск шёл по semantic-пути без имени вектора и падал до перезапуска бота."""
    assert await retriever.retrieve(_COLLECTION, "отпуск") == []

    _index(indexer, [
        ("Ежегодный отпуск составляет 28 календарных дней.", _vec(1, 0, 0, 0)),
        ("Зарплата выплачивается два раза в месяц.", _vec(0, 1, 0, 0)),
    ])

    chunks = await retriever.retrieve(_COLLECTION, "отпуск")
    assert chunks, "после загрузки документа поиск должен находить чанки"
    assert chunks[0]["filename"] == "policy.txt"
    assert "отпуск" in chunks[0]["text"]


@pytest.mark.asyncio
async def test_hybrid_collection_is_detected_and_cached(retriever, indexer):
    _index(indexer, [("Текст про отпуск.", _vec(1, 0, 0, 0))])

    assert retriever._is_hybrid_collection(_COLLECTION) is True
    assert _COLLECTION in retriever._hybrid_collections


def test_missing_collection_is_not_cached(retriever):
    assert retriever._is_hybrid_collection("user_missing") is False
    assert "user_missing" not in retriever._hybrid_collections


@pytest.mark.asyncio
async def test_relevance_gate_drops_unrelated_results(retriever, indexer):
    """RRF всегда возвращает top-k; без семантически близкого чанка ответ пустой."""
    _index(indexer, [("Зарплата выплачивается два раза в месяц.", _vec(0, 1, 0, 0))])

    assert await retriever.retrieve(_COLLECTION, "отпуск") == []
