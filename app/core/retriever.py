import asyncio
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from app.config import settings
from app.core.embedder import LocalEmbedder
from app.core.sparse_encoder import encode_sparse_query

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(self, embedder: LocalEmbedder) -> None:
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
        self.embedder = embedder
        self.top_k = settings.retriever_top_k
        self.score_threshold = settings.retriever_score_threshold
        # Коллекции, для которых подтверждена поддержка hybrid search.
        # Кэшируется только положительный ответ: коллекции без sparse-векторов
        # (ещё не созданные, legacy до /reset, недоступные из-за сбоя Qdrant)
        # после загрузки документа или восстановления становятся гибридными.
        self._hybrid_collections: set[str] = set()

    def _build_filter(self, document_id: str | None) -> Filter | None:
        if not document_id:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    def _collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def _is_hybrid_collection(self, collection_name: str) -> bool:
        """Проверить, поддерживает ли коллекция sparse vectors (BM25)."""
        if collection_name in self._hybrid_collections:
            return True

        try:
            info = self.client.get_collection(collection_name)
        except Exception as e:
            err_msg = str(e).lower()
            if "not found" not in err_msg and "doesn't exist" not in err_msg:
                logger.warning("Qdrant ошибка при проверке коллекции %s: %s", collection_name, e)
            return False

        sparse_config = info.config.params.sparse_vectors
        is_hybrid = sparse_config is not None and "bm25" in sparse_config
        if is_hybrid:
            self._hybrid_collections.add(collection_name)
        return is_hybrid

    def _hybrid_search_with_sparse(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector,
        document_id: str | None,
    ) -> list[dict]:
        """Hybrid search: dense + BM25 с RRF fusion."""
        if not self._collection_exists(collection_name):
            return []

        search_filter = self._build_filter(document_id)

        result = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    filter=search_filter,
                    limit=settings.hybrid_semantic_top_k,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="bm25",
                    filter=search_filter,
                    limit=settings.hybrid_bm25_top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=settings.hybrid_final_top_k,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "filename": hit.payload["filename"],
                "chunk_index": hit.payload["chunk_index"],
                "document_id": hit.payload["document_id"],
                "page_number": hit.payload.get("metadata", {}).get("page_number"),
                "section_header": hit.payload.get("metadata", {}).get("section_header"),
            }
            for hit in result.points
        ]

    def _semantic_search_sync(
        self,
        collection_name: str,
        query_vector: list[float],
        document_id: str | None,
        top_k: int,
    ) -> list[dict]:
        """Fallback: semantic-only search для legacy-коллекций (без sparse vectors)."""
        if not self._collection_exists(collection_name):
            return []

        search_filter = self._build_filter(document_id)

        # Коллекции с named vectors требуют tuple ("dense", vector)
        qv = ("dense", query_vector) if self._is_hybrid_collection(collection_name) else query_vector
        results = self.client.search(
            collection_name=collection_name,
            query_vector=qv,
            limit=top_k,
            score_threshold=self.score_threshold,
            query_filter=search_filter,
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "filename": hit.payload["filename"],
                "chunk_index": hit.payload["chunk_index"],
                "document_id": hit.payload["document_id"],
                "page_number": hit.payload.get("metadata", {}).get("page_number"),
                "section_header": hit.payload.get("metadata", {}).get("section_header"),
            }
            for hit in results
        ]

    async def retrieve(
        self,
        collection_name: str,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
        query_vector: list[float] | None = None,
        gate_vector: list[float] | None = None,
    ) -> list[dict]:
        """
        Async поиск релевантных чанков.

        Автоматически выбирает hybrid (RRF) или semantic-only
        в зависимости от возможностей коллекции.

        Args:
            query_vector: Вектор для dense search (может быть HyDE-вектором).
            gate_vector: Вектор для relevance gate (исходный запрос, не HyDE).
                         Если None — используется query_vector.

        Returns:
            [{"text", "score", "filename", "chunk_index", "document_id", ...}, ...]
        """
        if query_vector is None:
            query_vector = await self.embedder.embed_query(query)
        if gate_vector is None:
            gate_vector = query_vector

        use_hybrid = settings.hybrid_search_enabled and await asyncio.to_thread(
            self._is_hybrid_collection, collection_name
        )

        if use_hybrid:
            sparse_vector = encode_sparse_query(query)
            chunks = await asyncio.to_thread(
                self._hybrid_search_with_sparse,
                collection_name,
                query_vector,
                sparse_vector,
                document_id,
            )
            logger.info(
                "Hybrid search: найдено %d чанков для '%s...' в %s",
                len(chunks), query[:50], collection_name,
            )

            # Relevance gate: RRF не имеет score_threshold, поэтому всегда
            # возвращает hybrid_final_top_k результатов. Проверяем, что хотя бы
            # один чанк семантически релевантен запросу (cosine >= score_threshold).
            if chunks:
                gate_check = await asyncio.to_thread(
                    self._semantic_search_sync,
                    collection_name,
                    gate_vector,
                    document_id,
                    1,
                )
                if not gate_check:
                    logger.info(
                        "Relevance gate: '%s...' не прошёл semantic threshold %.2f — "
                        "отбрасываем %d hybrid результатов",
                        query[:50], self.score_threshold, len(chunks),
                    )
                    chunks = []
        else:
            chunks = await asyncio.to_thread(
                self._semantic_search_sync,
                collection_name,
                query_vector,
                document_id,
                top_k or self.top_k,
            )
            logger.info(
                "Semantic search: найдено %d чанков для '%s...' в %s",
                len(chunks), query[:50], collection_name,
            )

        return chunks
