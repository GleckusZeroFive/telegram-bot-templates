import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.core.sparse_encoder import encode_sparse

logger = logging.getLogger(__name__)


class QdrantIndexer:
    def __init__(self) -> None:
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
        self.dimension = settings.embedding_dimension

    def ensure_collection(self, collection_name: str) -> None:
        """Создать коллекцию с dense + sparse (BM25) векторами, если не существует."""
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
            )
            logger.info("Создана коллекция: %s (dense + bm25)", collection_name)

    def index_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
    ) -> int:
        """
        Индексировать чанки в Qdrant (dense + sparse BM25 векторы).

        Returns:
            Количество проиндексированных точек.
        """
        self.ensure_collection(collection_name)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": embedding,
                    "bm25": encode_sparse(chunk["text"]),
                },
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        # Загрузка батчами по 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i : i + batch_size],
            )

        logger.info(
            "Проиндексировано %d чанков в коллекцию %s (документ: %s)",
            len(points), collection_name, filename,
        )
        return len(points)

    def delete_document(self, collection_name: str, document_id: str) -> None:
        """Удалить все чанки документа из Qdrant."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Удалён документ %s из коллекции %s", document_id, collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Удалить коллекцию целиком."""
        self.client.delete_collection(collection_name=collection_name)
        logger.info("Удалена коллекция: %s", collection_name)

    # ── Методы для обновления документов ─────────────────────────

    def copy_document_points(
        self,
        collection_name: str,
        source_document_id: str,
        target_document_id: str,
        target_filename: str,
    ) -> int:
        """
        Копировать все точки документа для бэкапа (без повторного эмбеддинга).

        Обрабатывает батчами итеративно (scroll → transform → upsert)
        без накопления всех точек в памяти.

        Returns:
            Количество скопированных точек.
        """
        self.ensure_collection(collection_name)

        total_copied = 0
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=source_document_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            if results:
                new_points = []
                for point in results:
                    new_payload = dict(point.payload)
                    new_payload["document_id"] = target_document_id
                    new_payload["filename"] = target_filename
                    if "metadata" in new_payload and isinstance(new_payload["metadata"], dict):
                        new_payload["metadata"] = dict(new_payload["metadata"])
                        new_payload["metadata"]["filename"] = target_filename
                    new_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=point.vector,
                            payload=new_payload,
                        )
                    )
                self.client.upsert(
                    collection_name=collection_name,
                    points=new_points,
                )
                total_copied += len(new_points)

            if offset is None:
                break

        if total_copied:
            logger.info(
                "Скопировано %d точек: %s → %s (коллекция %s)",
                total_copied, source_document_id, target_document_id, collection_name,
            )
        return total_copied

    def get_max_chunk_index(self, collection_name: str, document_id: str) -> int:
        """
        Получить максимальный chunk_index для документа.

        Returns:
            Максимальный chunk_index или -1 если точек нет.
        """
        max_index = -1
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                idx = point.payload.get("chunk_index", -1)
                if idx > max_index:
                    max_index = idx
            if offset is None:
                break

        return max_index
