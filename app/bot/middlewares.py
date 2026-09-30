import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from app.db.database import db
from app.db.repositories.user import UserRepository

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseMiddleware):
    """Автоматическая регистрация пользователей и инъекция User + Session в хендлеры."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Извлекаем пользователя из события
        tg_user = data.get("event_from_user")
        if not tg_user:
            return await handler(event, data)

        async with db.get_session() as session:
            user_repo = UserRepository(session)
            user = await user_repo.get_or_create(
                telegram_id=tg_user.id,
                username=tg_user.username,
                first_name=tg_user.first_name,
            )

            # Проверяем истечение тарифа (некритическая операция — не блокирует бота)
            try:
                if (
                    user.tier != "free"
                    and user.tier_expires_at is not None
                    and user.tier_expires_at < datetime.now(UTC).replace(tzinfo=None)
                ):
                    logger.info("Тариф %s истёк у пользователя %s", user.tier, user.telegram_id)
                    await user_repo.revert_to_free(user.id)
                    await session.refresh(user)
            except Exception:
                logger.exception("Ошибка при проверке истечения тарифа (user=%s)", tg_user.id)

            data["user"] = user
            data["session"] = session
            return await handler(event, data)
