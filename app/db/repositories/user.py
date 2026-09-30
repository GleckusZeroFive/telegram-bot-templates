from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.config import TIER_LIMITS
from app.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_telegram_id(self, telegram_id: int) -> User | None:
        result = await self.session.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
    ) -> User:
        user = await self.get_by_telegram_id(telegram_id)
        if user:
            # Обновляем данные профиля, если изменились
            changed = False
            if username is not None and user.username != username:
                user.username = username
                changed = True
            if first_name is not None and user.first_name != first_name:
                user.first_name = first_name
                changed = True
            if changed:
                await self.session.commit()
            return user

        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def try_consume_query(self, user: User) -> bool:
        """Атомарно списать один запрос из дневного лимита.

        Проверка и инкремент — один UPDATE: параллельные сообщения одного
        пользователя не могут пройти лимит, прочитав одно и то же значение.

        Returns:
            True — запрос списан; False — дневной лимит исчерпан.
        """
        result = await self.session.execute(
            update(User)
            .where(User.id == user.id, User.queries_today < User.queries_limit)
            .values(queries_today=User.queries_today + 1)
            .returning(User.queries_today)
            .execution_options(synchronize_session=False)
        )
        new_value = result.scalar_one_or_none()
        await self.session.commit()
        if new_value is None:
            return False
        # Обновляем объект в памяти без пометки «изменён»: иначе следующий
        # commit этой сессии записал бы устаревшее значение поверх чужого инкремента
        set_committed_value(user, "queries_today", new_value)
        return True

    async def refund_query(self, user_id: int) -> None:
        """Вернуть списанный запрос, если его обработка не удалась."""
        await self.session.execute(
            update(User)
            .where(User.id == user_id, User.queries_today > 0)
            .values(queries_today=User.queries_today - 1)
            .execution_options(synchronize_session=False)
        )
        await self.session.commit()

    async def update_tier(
        self,
        user_id: int,
        tier: str,
        expires_at: datetime,
    ) -> None:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one()
        limits = TIER_LIMITS[tier]
        user.tier = tier
        user.tier_expires_at = expires_at
        user.documents_limit = limits["documents_limit"]
        user.queries_limit = limits["queries_limit"]
        await self.session.commit()

    async def revert_to_free(self, user_id: int) -> None:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one()
        limits = TIER_LIMITS["free"]
        user.tier = "free"
        user.tier_expires_at = None
        user.documents_limit = limits["documents_limit"]
        user.queries_limit = limits["queries_limit"]
        await self.session.commit()

    async def reset_daily_queries(self) -> None:
        """Сброс счётчика запросов (для ежедневного cron-а) — один UPDATE на всю таблицу."""
        await self.session.execute(
            update(User)
            .where(User.queries_today != 0)
            .values(queries_today=0)
            .execution_options(synchronize_session=False)
        )
        await self.session.commit()
