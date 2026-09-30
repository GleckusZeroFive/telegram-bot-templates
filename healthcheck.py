"""
Быстрая проверка здоровья всех сервисов RAG-бота.
Все проверки выполняются параллельно с таймаутом 5 сек.

Запуск:
    python healthcheck.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass

from app.config import settings

_TIMEOUT = 5.0  # секунд на каждую проверку

# ANSI-цвета
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    elapsed_ms: int


async def _timed(name: str, coro) -> CheckResult:
    """Обёртка: замеряет время, ловит таймаут и исключения."""
    t0 = time.monotonic()
    try:
        detail = await asyncio.wait_for(coro, timeout=_TIMEOUT)
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult(name=name, ok=True, detail=detail, elapsed_ms=ms)
    except TimeoutError:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult(name=name, ok=False, detail="TIMEOUT", elapsed_ms=ms)
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult(name=name, ok=False, detail=str(e)[:80], elapsed_ms=ms)


async def check_postgres() -> str:
    import asyncpg

    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        timeout=_TIMEOUT,
    )
    try:
        await conn.fetchval("SELECT 1")
    finally:
        await conn.close()
    return f"{settings.postgres_db} @ {settings.postgres_host}:{settings.postgres_port}"


async def check_qdrant() -> str:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=_TIMEOUT)
    result = await asyncio.to_thread(client.get_collections)
    return f"{len(result.collections)} collections"


async def check_telegram() -> str:
    from aiogram import Bot

    bot = Bot(token=settings.telegram_bot_token)
    try:
        me = await bot.get_me()
    finally:
        await bot.session.close()
    return f"@{me.username}"


async def check_embedding() -> str:
    from app.core.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    result = await embedder.check_health()
    if result["status"] == "ok":
        return f"OK ({result['dimension']}d), model={settings.embedding_model}"
    raise RuntimeError(result.get("error", "unknown error"))


async def check_llm() -> str:
    import httpx

    # GET /models есть у OpenAI-совместимых API (OpenAI, OpenRouter, Groq, Ollama, vLLM)
    base_url = settings.llm_base_url.rstrip("/")
    api_key = settings.llm_api_key or settings.llm_api_keys.split(",")[0].strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{base_url}/models", headers=headers)
        resp.raise_for_status()
    return f"{base_url} OK, model={settings.llm_model or '(LLM_MODEL не задан)'}"


def _print_results(results: list[CheckResult]) -> None:
    print(f"\n{_BOLD}RAG Bot Health Check{_RESET}")
    print("\u2501" * 60)

    for r in results:
        icon = f"{_GREEN}\u2713{_RESET}" if r.ok else f"{_RED}\u2717{_RESET}"
        ms = f"{_DIM}{r.elapsed_ms:>5}ms{_RESET}"
        print(f" {icon} {r.name:<18} {ms}   {r.detail}")

    print("\u2501" * 60)
    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    color = _GREEN if ok_count == total else _RED
    print(f" {color}{_BOLD}{ok_count}/{total} OK{_RESET}")
    print()


async def main() -> int:
    results = await asyncio.gather(
        _timed("PostgreSQL", check_postgres()),
        _timed("Qdrant", check_qdrant()),
        _timed("Telegram", check_telegram()),
        _timed("Embedding", check_embedding()),
        _timed("LLM", check_llm()),
    )
    _print_results(list(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
