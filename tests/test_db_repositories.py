"""Атомарность дневного лимита и инвайт-ключей на настоящем PostgreSQL.

Нужна БД с применёнными миграциями (alembic upgrade head):
    TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/db pytest -m db
Без переменной тесты пропускаются. В CI Postgres поднимается сервисом.
"""

import asyncio
import os
import random

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import InviteKey, User
from app.db.repositories.invite_key import InviteKeyRepository
from app.db.repositories.user import UserRepository

_DB_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not _DB_URL, reason="TEST_DATABASE_URL не задан"),
]


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(_DB_URL, pool_size=20, max_overflow=10)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def make_user(session_factory):
    created: list[int] = []

    async def _make(**fields) -> User:
        async with session_factory() as session:
            user = User(telegram_id=random.randint(10**9, 10**10), **fields)
            session.add(user)
            await session.commit()
            created.append(user.id)
            return user

    yield _make

    async with session_factory() as session:
        await session.execute(delete(InviteKey).where(InviteKey.created_by_id.in_(created)))
        await session.execute(delete(InviteKey).where(InviteKey.used_by_id.in_(created)))
        await session.execute(delete(User).where(User.id.in_(created)))
        await session.commit()


async def _queries_today(session_factory, user_id: int) -> int:
    async with session_factory() as session:
        return (await session.execute(
            select(User.queries_today).where(User.id == user_id)
        )).scalar_one()


@pytest.mark.asyncio
async def test_parallel_messages_cannot_exceed_daily_limit(session_factory, make_user):
    """Раньше лимит проверялся в Python, а инкремент шёл после ответа:
    параллельные сообщения проходили проверку на одном и том же значении."""
    user = await make_user(queries_today=0, queries_limit=3)

    async def consume() -> bool:
        async with session_factory() as session:
            db_user = await session.get(User, user.id)
            return await UserRepository(session).try_consume_query(db_user)

    results = await asyncio.gather(*(consume() for _ in range(10)))

    assert sum(results) == 3
    assert await _queries_today(session_factory, user.id) == 3


@pytest.mark.asyncio
async def test_consume_updates_in_memory_user_without_dirtying_it(session_factory, make_user):
    user = await make_user(queries_today=1, queries_limit=5)

    async with session_factory() as session:
        db_user = await session.get(User, user.id)
        assert await UserRepository(session).try_consume_query(db_user)
        assert db_user.queries_today == 2
        assert db_user not in session.dirty


@pytest.mark.asyncio
async def test_refund_returns_query_and_never_goes_negative(session_factory, make_user):
    user = await make_user(queries_today=1, queries_limit=5)

    async with session_factory() as session:
        repo = UserRepository(session)
        await repo.refund_query(user.id)
        await repo.refund_query(user.id)

    assert await _queries_today(session_factory, user.id) == 0


@pytest.mark.asyncio
async def test_reset_daily_queries(session_factory, make_user):
    a = await make_user(queries_today=7, queries_limit=50)
    b = await make_user(queries_today=50, queries_limit=50)

    async with session_factory() as session:
        await UserRepository(session).reset_daily_queries()

    assert await _queries_today(session_factory, a.id) == 0
    assert await _queries_today(session_factory, b.id) == 0


@pytest.mark.asyncio
async def test_invite_key_is_claimed_once_under_concurrency(session_factory, make_user):
    """Раньше «не использован» проверялось отдельным SELECT до записи владельца."""
    admin = await make_user(tier="admin")
    users = [await make_user() for _ in range(5)]

    async with session_factory() as session:
        key = await InviteKeyRepository(session).create(tier="pro", created_by_id=admin.id)

    async def claim(user_id: int) -> str | None:
        async with session_factory() as session:
            tier = await InviteKeyRepository(session).claim(key.key, user_id)
            await session.commit()
            return tier

    results = await asyncio.gather(*(claim(u.id) for u in users))

    winners = [u.id for u, tier in zip(users, results, strict=True) if tier == "pro"]
    assert len(winners) == 1
    assert results.count(None) == 4
    async with session_factory() as session:
        stored = await InviteKeyRepository(session).get_by_key(key.key)
        assert stored.used_by_id == winners[0]
        assert stored.used_at is not None


@pytest.mark.asyncio
async def test_claim_unknown_key_returns_none(session_factory, make_user):
    user = await make_user()
    async with session_factory() as session:
        assert await InviteKeyRepository(session).claim("ZZZZ-ZZZZ-ZZZZ", user.id) is None
