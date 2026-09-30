import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.chunker import AdvancedChunker, TextChunker
from app.core.document_processor import DocumentProcessor
from app.core.embedder import LocalEmbedder, ProgressCallback
from app.core.generator import ResponseGenerator
from app.core.indexer import QdrantIndexer
from app.core.retriever import QdrantRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Оркестратор RAG-пайплайна.
    Два основных потока:
    1. ingest_document() — загрузка документа
    2. query_stream() — вопрос-ответ со стримингом
    """

    def __init__(self) -> None:
        self.processor = DocumentProcessor()
        self.chunker = AdvancedChunker() if settings.chunker_mode == "advanced" else TextChunker()
        self.embedder = LocalEmbedder()
        self.indexer = QdrantIndexer()
        self.retriever = QdrantRetriever(self.embedder)
        self.generator = ResponseGenerator()

        self._law_client = None
        if settings.law_corpus_enabled:
            from app.core.law_client import LawClient
            self._law_client = LawClient()

    async def ingest_document(
        self,
        file_path: Path,
        file_type: str,
        document_id: str,
        filename: str,
        user_telegram_id: int,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[int, str]:
        """
        Полный поток загрузки документа:
        файл → текст → чанки → эмбеддинги → Qdrant

        Returns:
            (chunk_count, full_text) — количество чанков и полный извлечённый текст.
        """
        collection_name = f"user_{user_telegram_id}"

        # 1. Извлечение текста
        text = await self.processor.process(file_path, file_type)
        if not text.strip():
            return 0, ""

        # 2. Чанкинг
        chunks = self.chunker.chunk(
            text,
            metadata={"filename": filename, "file_type": file_type},
        )
        if not chunks:
            return 0, text

        # 3. Эмбеддинг (async, с прогрессом)
        texts = [c["text"] for c in chunks]
        embeddings = await self.embedder.embed_documents(texts, on_progress=on_progress)

        # 4. Индексация в Qdrant (синхронный клиент → to_thread)
        count = await asyncio.to_thread(
            self.indexer.index_chunks,
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=filename,
        )

        logger.info(
            "Документ '%s' проиндексирован: %d чанков (user=%d)",
            filename, count, user_telegram_id,
        )
        return count, text

    # ── Методы для обновления документов ─────────────────────────

    async def replace_document(
        self,
        file_path: Path,
        file_type: str,
        document_id: str,
        filename: str,
        user_telegram_id: int,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[int, str]:
        """
        Замена документа: удалить старые чанки, загрузить новые.

        Returns:
            (chunk_count, full_text)
        """
        collection_name = f"user_{user_telegram_id}"

        # 1. Удалить старые чанки
        await asyncio.to_thread(
            self.indexer.delete_document, collection_name, document_id
        )

        # 2. Загрузить новые (переиспользуем ingest_document)
        count, text = await self.ingest_document(
            file_path=file_path,
            file_type=file_type,
            document_id=document_id,
            filename=filename,
            user_telegram_id=user_telegram_id,
            on_progress=on_progress,
        )

        logger.info(
            "Документ '%s' заменён: %d чанков (user=%d)",
            filename, count, user_telegram_id,
        )
        return count, text

    async def append_text_to_document(
        self,
        text: str,
        document_id: str,
        filename: str,
        user_telegram_id: int,
        on_progress: ProgressCallback | None = None,
    ) -> int:
        """
        Дополнить документ новым текстом: чанкинг → эмбеддинг → индексация с offset.

        Returns:
            Количество добавленных чанков.
        """
        collection_name = f"user_{user_telegram_id}"

        # 1. Получить максимальный chunk_index
        max_idx = await asyncio.to_thread(
            self.indexer.get_max_chunk_index, collection_name, document_id
        )
        start_index = max_idx + 1

        # 2. Чанкинг нового текста
        chunks = self.chunker.chunk(
            text,
            metadata={"filename": filename, "file_type": "append"},
        )
        if not chunks:
            return 0

        # 3. Перенумеровать chunk_index с offset
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = start_index + i
            if "metadata" in chunk:
                chunk["metadata"]["chunk_index"] = start_index + i

        # 4. Эмбеддинг
        texts = [c["text"] for c in chunks]
        embeddings = await self.embedder.embed_documents(texts, on_progress=on_progress)

        # 5. Индексация
        count = await asyncio.to_thread(
            self.indexer.index_chunks,
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            filename=filename,
        )

        logger.info(
            "Дополнено %d чанков к '%s' (с индекса %d, user=%d)",
            count, filename, start_index, user_telegram_id,
        )
        return count

    async def generate_diff_summary(self, old_text: str, new_text: str) -> str:
        """Сгенерировать краткую сводку изменений между двумя версиями текста."""
        max_chars = 8000

        def _truncate(t: str) -> str:
            if len(t) <= max_chars:
                return t
            half = max_chars // 2
            return t[:half] + "\n\n[...пропущено...]\n\n" + t[-half:]

        old_truncated = _truncate(old_text)
        new_truncated = _truncate(new_text)

        prompt = (
            "Сравни два текста документа и кратко опиши изменения (3-5 пунктов).\n\n"
            "СТАРЫЙ ТЕКСТ:\n"
            f"{old_truncated}\n\n"
            "НОВЫЙ ТЕКСТ:\n"
            f"{new_truncated}\n\n"
            "Кратко опиши:\n"
            "1. Что добавлено?\n"
            "2. Что удалено?\n"
            "3. Что изменено?\n"
            "Формат: HTML для Telegram (<b>, <i>, списки с •). Будь лаконичен."
        )

        messages = [
            {"role": "system", "content": "Ты — ассистент для сравнения документов."},
            {"role": "user", "content": prompt},
        ]
        return await self.generator.provider.generate(messages)

    async def _prepare_search(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> tuple[str, list[float], list[float]]:
        """Переформулировка запроса и HyDE.

        Оба шага — улучшения поиска, а не обязательные этапы: при ошибке или
        превышении llm_aux_timeout поиск идёт по исходному вопросу.
        Вызовы последовательные, чтобы не упираться в rate-limit провайдера.

        Returns:
            (search_query, query_vector, dense_vector) — query_vector считается по
            исходному вопросу (для relevance gate), dense_vector — по HyDE-документу.
        """
        search_query = question
        if settings.query_rewrite_enabled:
            try:
                rewritten = await asyncio.wait_for(
                    self.generator.rewrite_query(question, conversation_history),
                    timeout=settings.llm_aux_timeout,
                )
                if rewritten and rewritten != question:
                    search_query = rewritten
                    logger.info("Query rewritten: '%s' -> '%s'", question[:60], search_query[:60])
            except Exception as e:
                logger.warning("rewrite_query failed: %r", e)

        query_vector = await self.embedder.embed_query(question)
        dense_vector = query_vector
        if settings.hyde_enabled:
            try:
                hyde_doc = await asyncio.wait_for(
                    self.generator.generate_hypothetical(search_query),
                    timeout=settings.llm_aux_timeout,
                )
                if hyde_doc:
                    logger.info("HyDE doc: '%s...'", hyde_doc[:80])
                    dense_vector = await self.embedder.embed_passage(hyde_doc)
            except Exception as e:
                logger.warning("generate_hypothetical failed: %r", e)

        return search_query, query_vector, dense_vector

    async def query_stream(
        self,
        question: str,
        user_telegram_id: int,
        document_id: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        law_search_enabled: bool = False,
        user_state: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Стриминг Q&A: retrieval (быстро), затем yield дельт генерации.

        Yields:
            {"type": "meta", "sources": [...], "model": str, "law_search_failed": bool}
            {"type": "delta", "text": str}  — текстовые дельты от LLM
        """
        collection_name = f"user_{user_telegram_id}"

        search_query, query_vector, dense_vector = await self._prepare_search(
            question, conversation_history,
        )

        user_chunks = await self.retriever.retrieve(
            collection_name=collection_name,
            query=search_query,
            document_id=document_id,
            query_vector=dense_vector,
            gate_vector=query_vector,
        )
        for c in user_chunks:
            c["source_type"] = "user"

        law_chunks = []
        law_search_failed = False
        if law_search_enabled and self._law_client:
            try:
                raw = await self._law_client.search(
                    question, query_vector, settings.law_corpus_top_k,
                )
                law_chunks = [{"source_type": "law", **r} for r in raw]
            except Exception:
                logger.warning("Law API недоступен", exc_info=True)
                law_search_failed = True

        all_chunks = self._merge_results(user_chunks, law_chunks)

        # Метаданные: sources + model (доступны до генерации)
        sources = ResponseGenerator.extract_sources(all_chunks)
        yield {
            "type": "meta",
            "sources": sources,
            "model": self.generator.provider.model,
            "law_search_failed": law_search_failed,
        }

        # 5: Стриминг генерации
        async for delta in self.generator.generate_stream(
            question, all_chunks,
            conversation_history=conversation_history,
            user_state=user_state,
        ):
            yield {"type": "delta", "text": delta}

    @staticmethod
    def _merge_results(
        user_chunks: list[dict],
        law_chunks: list[dict],
    ) -> list[dict]:
        """Мерж пользовательских и law-чанков по взвешенному score."""
        if not law_chunks:
            return user_chunks
        if not user_chunks:
            return law_chunks

        for c in law_chunks:
            c["score"] = c.get("score", 0) * settings.law_corpus_weight

        merged = sorted(
            user_chunks + law_chunks,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return merged[:10]


# ── Синглтон ──────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Глобальный синглтон RAGPipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
