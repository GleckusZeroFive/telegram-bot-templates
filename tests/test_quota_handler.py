"""Списание и возврат дневного лимита вокруг обработки запроса."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.handlers import query


@pytest.fixture
def repo(monkeypatch):
    repo = MagicMock()
    repo.try_consume_query = AsyncMock(return_value=True)
    repo.refund_query = AsyncMock()
    monkeypatch.setattr(query, "UserRepository", MagicMock(return_value=repo))
    return repo


@pytest.fixture
def user():
    return SimpleNamespace(id=7, telegram_id=700, queries_limit=50)


@pytest.fixture
def session():
    return MagicMock(rollback=AsyncMock())


@pytest.mark.asyncio
async def test_successful_answer_keeps_the_query_counted(monkeypatch, repo, user, session):
    monkeypatch.setattr(query, "route_and_process", AsyncMock(return_value=True))

    await query.answer_with_quota("вопрос", MagicMock(), user, session)

    repo.try_consume_query.assert_awaited_once_with(user)
    repo.refund_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_answer_is_refunded(monkeypatch, repo, user, session):
    monkeypatch.setattr(query, "route_and_process", AsyncMock(return_value=False))

    await query.answer_with_quota("вопрос", MagicMock(), user, session)

    session.rollback.assert_awaited_once()
    repo.refund_query.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_exception_is_refunded_and_propagated(monkeypatch, repo, user, session):
    monkeypatch.setattr(query, "route_and_process", AsyncMock(side_effect=RuntimeError("db down")))

    with pytest.raises(RuntimeError):
        await query.answer_with_quota("вопрос", MagicMock(), user, session)

    repo.refund_query.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_exhausted_limit_skips_processing(monkeypatch, repo, user, session):
    repo.try_consume_query = AsyncMock(return_value=False)
    route = AsyncMock()
    edit = AsyncMock(return_value=True)
    monkeypatch.setattr(query, "route_and_process", route)
    monkeypatch.setattr(query, "_safe_edit_text", edit)
    status = MagicMock()

    await query.answer_with_quota("вопрос", status, user, session, limit_prefix="Распознано: x\n\n")

    route.assert_not_awaited()
    text = edit.await_args.args[1]
    assert text.startswith("Распознано: x")
    assert "лимит запросов (50)" in text
