"""
Тесты стриминга: provider retry/ошибки, generator, pipeline, handler.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.llm.provider import LLMError, OpenAICompatibleProvider

# ── Хелперы ──────────────────────────────────────────────────

def _make_chunk(text: str | None, finish_reason: str | None = None):
    """Создать mock OpenAI stream chunk."""
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class MockStreamResponse:
    """Async iterator, имитирующий OpenAI streaming response."""

    def __init__(self, chunks: list):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _make_status_error(status_code: int, message: str = "error") -> APIStatusError:
    """Создать APIStatusError с нужным кодом."""
    response = httpx.Response(status_code, request=httpx.Request("POST", "http://test"))
    return APIStatusError(message, response=response, body=None)


def _make_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "http://test"))


def _make_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://test"))


@pytest.fixture
def provider():
    """Провайдер с заглушкой клиента."""
    p = OpenAICompatibleProvider(base_url="http://test/v1", model="test-model")
    p._client = MagicMock()
    return p


# ── Provider: generate_stream() ─────────────────────────────

@pytest.mark.asyncio
async def test_stream_happy_path(provider):
    """Стрим возвращает 3 дельты → все yielded."""
    chunks = [_make_chunk("один"), _make_chunk("два"), _make_chunk("три")]
    provider._client.chat.completions.create = AsyncMock(
        return_value=MockStreamResponse(chunks)
    )

    result = []
    async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
        result.append(delta)

    assert result == ["один", "два", "три"]


@pytest.mark.asyncio
async def test_stream_empty_deltas_skipped(provider):
    """Чанки без content пропускаются."""
    chunks = [
        _make_chunk("текст"),
        _make_chunk(None),  # пустая дельта
        _make_chunk("ещё"),
    ]
    provider._client.chat.completions.create = AsyncMock(
        return_value=MockStreamResponse(chunks)
    )

    result = []
    async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
        result.append(delta)

    assert result == ["текст", "ещё"]


@pytest.mark.asyncio
async def test_stream_retry_on_429(provider):
    """При 429 на create() — retry, второй вызов успешен."""
    chunks = [_make_chunk("ok")]

    provider._client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_status_error(429, "rate limited"),
            MockStreamResponse(chunks),
        ]
    )

    with patch("app.llm.provider.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = []
        async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
            result.append(delta)

    assert result == ["ok"]
    assert provider._client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0)  # первый backoff


@pytest.mark.asyncio
async def test_stream_retry_on_502(provider):
    """При 502 — retry."""
    chunks = [_make_chunk("ok")]

    provider._client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_status_error(502),
            MockStreamResponse(chunks),
        ]
    )

    with patch("app.llm.provider.asyncio.sleep", new_callable=AsyncMock):
        result = []
        async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
            result.append(delta)

    assert result == ["ok"]


@pytest.mark.asyncio
async def test_stream_fatal_404_no_retry(provider, caplog):
    """404 — fatal, без retry. Пользователь видит общее сообщение, причина — в логе."""
    provider._client.chat.completions.create = AsyncMock(
        side_effect=_make_status_error(404, "model not found")
    )

    with pytest.raises(LLMError, match="временно недоступен"):
        async for _ in provider.generate_stream([{"role": "user", "content": "тест"}]):
            pass

    assert provider._client.chat.completions.create.call_count == 1
    assert "не найдена" in caplog.text


@pytest.mark.asyncio
async def test_stream_fatal_401_no_retry(provider, caplog):
    """401 — fatal, без retry. Пользователь видит общее сообщение, причина — в логе."""
    provider._client.chat.completions.create = AsyncMock(
        side_effect=_make_status_error(401, "unauthorized")
    )

    with pytest.raises(LLMError, match="временно недоступен"):
        async for _ in provider.generate_stream([{"role": "user", "content": "тест"}]):
            pass

    assert provider._client.chat.completions.create.call_count == 1
    assert "авторизации" in caplog.text


@pytest.mark.asyncio
async def test_stream_timeout_all_retries(provider):
    """Таймаут на всех 3 попытках → LLMError."""
    provider._client.chat.completions.create = AsyncMock(
        side_effect=_make_timeout_error()
    )

    with (
        patch("app.llm.provider.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(LLMError, match="не отвечает"),
    ):
        async for _ in provider.generate_stream([{"role": "user", "content": "тест"}]):
            pass

    assert provider._client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_stream_connection_error_retry_then_success(provider):
    """Connection error → retry → успех на 2й попытке."""
    chunks = [_make_chunk("ok")]

    provider._client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_connection_error(),
            MockStreamResponse(chunks),
        ]
    )

    with patch("app.llm.provider.asyncio.sleep", new_callable=AsyncMock):
        result = []
        async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
            result.append(delta)

    assert result == ["ok"]
    assert provider._client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_stream_connection_error_all_retries(provider):
    """Connection error на всех попытках → LLMError."""
    provider._client.chat.completions.create = AsyncMock(
        side_effect=_make_connection_error()
    )

    with (
        patch("app.llm.provider.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(LLMError, match="недоступен"),
    ):
        async for _ in provider.generate_stream([{"role": "user", "content": "тест"}]):
            pass

    assert provider._client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_stream_mid_stream_error(provider):
    """Ошибка mid-stream (после первых дельт) — пробрасывается без retry."""

    async def _failing_stream():
        yield _make_chunk("начало")
        raise _make_status_error(500, "internal error")

    class FailingStreamResponse:
        def __aiter__(self):
            return _failing_stream()

    provider._client.chat.completions.create = AsyncMock(
        return_value=FailingStreamResponse()
    )

    result = []
    with pytest.raises(LLMError):
        async for delta in provider.generate_stream([{"role": "user", "content": "тест"}]):
            result.append(delta)

    # Первая дельта получена до ошибки
    assert result == ["начало"]
    # create вызван только 1 раз (retry не было, ошибка mid-stream)
    assert provider._client.chat.completions.create.call_count == 1


# ── Generator: generate_stream() ────────────────────────────

@pytest.mark.asyncio
async def test_generator_stream_delegates():
    """Generator передаёт сообщения в provider и forwards дельты."""
    from app.core.generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)

    mock_provider = MagicMock()
    mock_provider.model = "test-model"

    async def _mock_stream(messages):
        yield "привет"
        yield " мир"

    mock_provider.generate_stream = _mock_stream
    gen.provider = mock_provider

    result = []
    async for delta in gen.generate_stream("вопрос", []):
        result.append(delta)

    assert result == ["привет", " мир"]


# ── Pipeline: query_stream() ────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_meta_then_deltas():
    """Pipeline yields meta (sources, model) затем дельты текста."""
    from app.core.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)

    # Mock embedder
    mock_embedder = AsyncMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)
    pipeline.embedder = mock_embedder

    # Mock retriever — пустые результаты
    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])
    pipeline.retriever = mock_retriever

    # Mock generator
    mock_generator = MagicMock()
    mock_generator.provider = MagicMock()
    mock_generator.provider.model = "test-model"
    mock_generator.rewrite_query = AsyncMock(return_value="тест")
    mock_generator.generate_hypothetical = AsyncMock(return_value="")

    async def _mock_gen_stream(question, chunks, conversation_history=None, user_state=None):
        yield "ответ"
        yield " на вопрос"

    mock_generator.generate_stream = _mock_gen_stream
    pipeline.generator = mock_generator

    pipeline._law_client = None

    events = []
    async for event in pipeline.query_stream(
        question="тест",
        user_telegram_id=123,
    ):
        events.append(event)

    # Первое событие — meta
    assert events[0]["type"] == "meta"
    assert events[0]["model"] == "test-model"
    assert events[0]["sources"] == []
    assert events[0]["law_search_failed"] is False

    # Далее — дельты
    assert events[1] == {"type": "delta", "text": "ответ"}
    assert events[2] == {"type": "delta", "text": " на вопрос"}
