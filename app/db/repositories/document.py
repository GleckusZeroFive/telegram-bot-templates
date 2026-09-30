import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: int,
        filename: str,
        file_type: str,
        file_size: int,
        qdrant_collection: str | None = None,
    ) -> Document:
        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            qdrant_collection=qdrant_collection,
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: uuid.UUID) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: int) -> list[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        doc_id: uuid.UUID,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one()
        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message is not None:
            doc.error_message = error_message
        await self.session.commit()

    async def delete(self, doc_id: uuid.UUID) -> None:
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if doc:
            await self.session.delete(doc)
            await self.session.commit()

    # ── Методы для обновления документов ─────────────────────────

    async def update_full_text(self, doc_id: uuid.UUID, full_text: str) -> None:
        """Сохранить полный извлечённый текст документа."""
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one()
        doc.full_text = full_text
        await self.session.commit()

    async def create_backup(
        self,
        original: Document,
        backup_filename: str,
    ) -> Document:
        """Создать запись бэкапа документа (is_backup=True, parent_id=original.id)."""
        backup = Document(
            user_id=original.user_id,
            filename=backup_filename,
            file_type=original.file_type,
            file_size=original.file_size,
            chunk_count=original.chunk_count,
            status="ready",
            qdrant_collection=original.qdrant_collection,
            full_text=original.full_text,
            version=original.version,
            parent_id=original.id,
            is_backup=True,
        )
        self.session.add(backup)
        await self.session.commit()
        await self.session.refresh(backup)
        return backup

    async def increment_version(self, doc_id: uuid.UUID) -> None:
        """Увеличить версию документа и обновить updated_at."""
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one()
        doc.version += 1
        doc.updated_at = datetime.now(UTC).replace(tzinfo=None)
        await self.session.commit()

    async def get_by_user_non_backup(self, user_id: int) -> list[Document]:
        """Получить документы пользователя без бэкапов."""
        result = await self.session.execute(
            select(Document)
            .where(Document.user_id == user_id, Document.is_backup == False)  # noqa: E712
            .order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_chunk_count(self, doc_id: uuid.UUID, chunk_count: int) -> None:
        """Обновить количество чанков."""
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        doc = result.scalar_one()
        doc.chunk_count = chunk_count
        await self.session.commit()
