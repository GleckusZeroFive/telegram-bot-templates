"""Регрессии слоя LLM: кэш ответов и FallbackProvider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from app.llm.provider import (
    FallbackProvider,
    LLMError,
    OpenAICompatibleProvider,
    ResponseCache,
    _response_cache,
)


def _completion(text: str):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _chunk(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://test"))


def _status_error(code: int) -> APIStatusError:
    response = httpx.Response(code, request=httpx.Request("POST", "http://test"))
    return APIStatusError("error", response=response, body=None)


class _Stream:
    """Стрим OpenAI: отдаёт чанки, затем (опционально) падает."""

    def __init__(self, texts: list[str], error: Exception | None = None):
        self._texts = list(texts)
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._texts:
            return _chunk(self._texts.pop(0))
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        raise StopAsyncIteration


def _provider(model: str = "primary-model") -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider(base_url="http://test/v1", model=model)
    p._client = MagicMock()
    return p


@pytest.fixture(autouse=True)
def _clean_cache():
    _response_cache.clear()
    yield
    _response_cache.clear()


# ── ResponseCache ─────────────────────────────────────────────

_SYSTEM = {"role": "system", "content": "Ты — классификатор сообщений бота-ассистента по документам." * 3}
_HISTORY = "История диалога:\nПользователь: " + "Какие условия расторжения договора аренды? " * 5


def test_cache_key_differs_for_questions_with_shared_long_prefix():
    """Раньше ключ брал первые 200 символов сообщения: вопросы после длинной
    одинаковой истории получали один ключ и чужой ответ."""
    m1 = [_SYSTEM, {"role": "user", "content": _HISTORY + "\nТекущее сообщение: а сроки?"}]
    m2 = [_SYSTEM, {"role": "user", "content": _HISTORY + "\nТекущее сообщение: привет"}]
    assert ResponseCache._key(m1, "m", 0.0) != ResponseCache._key(m2, "m", 0.0)


def test_cache_key_depends_on_role_model_and_temperature():
    base = [{"role": "user", "content": "вопрос"}]
    as_system = [{"role": "system", "content": "вопрос"}]
    assert ResponseCache._key(base, "m", 0.0) != ResponseCache._key(as_system, "m", 0.0)
    assert ResponseCache._key(base, "m", 0.0) != ResponseCache._key(base, "other", 0.0)
    assert ResponseCache._key(base, "m", 0.0) != ResponseCache._key(base, "m", 0.5)


def test_multimodal_messages_are_not_cached():
    """Раньше multimodal-контент ронял построение ключа с TypeError."""
    cache = ResponseCache()
    vision = [{"role": "user", "content": [
        {"type": "text", "text": "опиши"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    cache.put(vision, "m", 0.1, "ответ")
    assert cache.get(vision, "m", 0.1) is None


@pytest.mark.asyncio
async def test_generate_uses_cache_only_when_asked():
    p = _provider()
    p._client.chat.completions.create = AsyncMock(return_value=_completion("rag"))
    messages = [{"role": "user", "content": "вопрос"}]

    await p.generate(messages, temperature=0.0, use_cache=True)
    await p.generate(messages, temperature=0.0, use_cache=True)
    assert p._client.chat.completions.create.call_count == 1

    # Основные ответы не кэшируются, даже при низкой температуре
    await p.generate(messages, temperature=0.0)
    assert p._client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_cached_answer_is_not_returned_for_another_question():
    p = _provider()
    p._client.chat.completions.create = AsyncMock(
        side_effect=[_completion("followup"), _completion("chat")],
    )
    m1 = [_SYSTEM, {"role": "user", "content": _HISTORY + "\nТекущее сообщение: а сроки?"}]
    m2 = [_SYSTEM, {"role": "user", "content": _HISTORY + "\nТекущее сообщение: привет"}]

    assert await p.generate(m1, temperature=0.0, use_cache=True) == "followup"
    assert await p.generate(m2, temperature=0.0, use_cache=True) == "chat"


@pytest.mark.asyncio
async def test_generate_with_image_and_cache_flag_does_not_crash():
    p = _provider()
    p._client.chat.completions.create = AsyncMock(return_value=_completion("описание"))
    vision = [{"role": "user", "content": [
        {"type": "text", "text": "опиши"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    assert await p.generate(vision, use_cache=True) == "описание"


# ── FallbackProvider: стриминг ────────────────────────────────

@pytest.mark.asyncio
async def test_stream_failure_after_first_token_is_not_duplicated_by_fallback():
    """Раньше обрыв primary после первых токенов запускал fallback, и полный
    ответ дописывался после обрывка: ['PARTIAL-', 'FULL ANSWER']."""
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(
        return_value=_Stream(["PARTIAL-"], error=_connection_error()),
    )
    fallback._client.chat.completions.create = AsyncMock(
        return_value=_Stream(["FULL ANSWER"]),
    )
    fp = FallbackProvider(primary, fallback)

    received = []
    with pytest.raises(LLMError):
        async for chunk in fp.generate_stream([{"role": "user", "content": "q"}]):
            received.append(chunk)

    assert received == ["PARTIAL-"]
    fallback._client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_stream_falls_back_when_primary_fails_before_first_token(monkeypatch):
    monkeypatch.setattr("app.llm.provider.asyncio.sleep", AsyncMock())
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(side_effect=_connection_error())
    fallback._client.chat.completions.create = AsyncMock(return_value=_Stream(["ok"]))
    fp = FallbackProvider(primary, fallback)

    chunks = [c async for c in fp.generate_stream([{"role": "user", "content": "q"}])]

    assert chunks == ["ok"]
    assert fp.model == "primary"  # общий объект не перепрописывает модель на лету


@pytest.mark.asyncio
async def test_stream_falls_back_on_empty_primary_stream():
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(return_value=_Stream([]))
    fallback._client.chat.completions.create = AsyncMock(return_value=_Stream(["ok"]))
    fp = FallbackProvider(primary, fallback)

    assert [c async for c in fp.generate_stream([{"role": "user", "content": "q"}])] == ["ok"]


@pytest.mark.asyncio
async def test_stream_permanent_error_is_not_retried_on_fallback():
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(side_effect=_status_error(401))
    fallback._client.chat.completions.create = AsyncMock(return_value=_Stream(["ok"]))
    fp = FallbackProvider(primary, fallback)

    with pytest.raises(LLMError):
        async for _ in fp.generate_stream([{"role": "user", "content": "q"}]):
            pass
    fallback._client.chat.completions.create.assert_not_called()


# ── FallbackProvider: generate ────────────────────────────────

@pytest.mark.asyncio
async def test_generate_falls_back_on_transient_error(monkeypatch):
    monkeypatch.setattr("app.llm.provider.asyncio.sleep", AsyncMock())
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(side_effect=_status_error(503))
    fallback._client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
    fp = FallbackProvider(primary, fallback)

    assert await fp.generate([{"role": "user", "content": "q"}], model="primary-only") == "ok"
    # Переопределение модели относится к primary, fallback работает своей моделью
    assert fallback._client.chat.completions.create.call_args.kwargs["model"] == "fallback"
    assert fp.model == "primary"


@pytest.mark.asyncio
async def test_generate_does_not_fall_back_on_bad_request():
    primary, fallback = _provider("primary"), _provider("fallback")
    primary._client.chat.completions.create = AsyncMock(side_effect=_status_error(400))
    fallback._client.chat.completions.create = AsyncMock(return_value=_completion("ok"))
    fp = FallbackProvider(primary, fallback)

    with pytest.raises(LLMError):
        await fp.generate([{"role": "user", "content": "q"}])
    fallback._client.chat.completions.create.assert_not_called()
