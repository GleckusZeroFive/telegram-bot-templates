import secrets
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InviteKey

# Символы без амбивалентных (0/O, 1/I/L)
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def generate_key() -> str:
    """Генерация ключа формата XXXX-XXXX-XXXX.

    Ключ даёт платный или admin-тариф, поэтому источник случайности —
    криптографический (secrets), а не предсказуемый random.
    """
    chars = "".join(secrets.choice(_ALPHABET) for _ in range(12))
    return f"{chars[:4]}-{chars[4:8]}-{chars[8:12]}"


def mask_key(key: str) -> str:
    """Ключ для логов: первая группа видна, остальное скрыто."""
    return key[:4] + "-****-****"


class InviteKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, tier: str, created_by_id: int | None = None) -> InviteKey:
        key = InviteKey(
            key=generate_key(),
            tier=tier,
            created_by_id=created_by_id,
        )
        self.session.add(key)
        await self.session.commit()
        await self.session.refresh(key)
        return key

    async def get_by_key(self, key: str) -> InviteKey | None:
        result = await self.session.execute(
            select(InviteKey).where(InviteKey.key == key)
        )
        return result.scalar_one_or_none()

    async def claim(self, key: str, user_id: int) -> str | None:
        """Атомарно занять ключ за пользователем.

        Проверка «не использован» и запись владельца — один UPDATE, поэтому один
        ключ не активируют двое, прислав его одновременно. Транзакцию не
        фиксирует: вызывающий коммитит её вместе со сменой тарифа.

        Returns:
            Тариф ключа, если ключ был свободен и теперь занят; иначе None.
        """
        result = await self.session.execute(
            update(InviteKey)
            .where(InviteKey.key == key, InviteKey.used_by_id.is_(None))
            .values(
                used_by_id=user_id,
                used_at=datetime.now(UTC).replace(tzinfo=None),
            )
            .returning(InviteKey.tier)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none()

    async def get_by_creator(self, user_id: int) -> list[InviteKey]:
        result = await self.session.execute(
            select(InviteKey)
            .where(InviteKey.created_by_id == user_id)
            .order_by(InviteKey.created_at.desc())
        )
        return list(result.scalars().all())
