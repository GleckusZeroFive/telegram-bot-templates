import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.bot.handlers import (
    docs_router,
    keys_router,
    law_router,
    query_router,
    quiz_router,
    start_router,
    update_router,
    upload_router,
    voice_router,
)
from app.bot.middlewares import AuthMiddleware
from app.config import settings
from app.db.database import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_reset_task: asyncio.Task | None = None
_tz_krasnoyarsk = timezone(timedelta(hours=7))


async def _daily_reset_loop() -> None:
    """Фоновая задача: сброс счётчика ежедневных запросов по расписанию (UTC+7)."""
    while True:
        now = datetime.now(tz=_tz_krasnoyarsk)
        reset_time = now.replace(
            hour=settings.daily_reset_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        if now >= reset_time:
            reset_time += timedelta(days=1)
        sleep_seconds = (reset_time - now).total_seconds()
        logger.info(
            "Следующий сброс лимитов через %.0f сек (в %s UTC+7)",
            sleep_seconds,
            reset_time.strftime("%Y-%m-%d %H:%M"),
        )
        await asyncio.sleep(sleep_seconds)

        try:
            async with db.get_session() as session:
                from app.db.repositories.user import UserRepository

                repo = UserRepository(session)
                await repo.reset_daily_queries()
            logger.info("Сброс дневных лимитов запросов выполнен")
        except Exception:
            logger.exception("Ошибка при сбросе дневных лимитов запросов")

        # Очистка истёкших контекстов диалога
        from app.core.conversation import cleanup_expired
        cleaned = cleanup_expired()
        if cleaned:
            logger.info("Очищено %d истёкших контекстов диалога", cleaned)


async def on_startup() -> None:
    global _reset_task
    await db.connect()
    logger.info("Database connected")

    # Сброс счётчиков при старте (на случай перезапуска после полуночи)
    try:
        async with db.get_session() as session:
            from app.db.repositories.user import UserRepository

            repo = UserRepository(session)
            await repo.reset_daily_queries()
        logger.info("Сброс дневных лимитов при старте выполнен")
    except Exception:
        logger.exception("Ошибка сброса лимитов при старте")

    # Запускаем фоновый таск сброса лимитов
    _reset_task = asyncio.create_task(_daily_reset_loop())

    # Health check: проверяем локальную модель эмбеддингов (через pipeline singleton)
    try:
        from app.core.rag_pipeline import get_pipeline

        pipeline = get_pipeline()
        result = await pipeline.embedder.check_health()
        if result["status"] == "ok":
            logger.info("Embedding model: OK (dim=%d)", result["dimension"])
        else:
            logger.error("Embedding model: ОШИБКА — %s", result.get("error", "?"))
    except Exception as e:
        logger.error("Ошибка загрузки модели эмбеддингов: %s", e)

    # Калибровка LLM-модели
    try:
        from app.core.calibrator import run_calibration
        profile = await run_calibration()
        logger.info("LLM calibration: %s", profile)
    except Exception:
        logger.warning("LLM calibration failed, using strict defaults")


async def on_shutdown() -> None:
    global _reset_task
    if _reset_task and not _reset_task.done():
        _reset_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _reset_task

    # Закрываем HTTP-клиент LawClient (если pipeline был инициализирован)
    from app.core.rag_pipeline import _pipeline
    if _pipeline and _pipeline._law_client:
        await _pipeline._law_client.close()

    await db.disconnect()
    logger.info("Database disconnected")


async def main() -> None:
    logger.info("RAG Document Bot starting...")

    # Ошибки конфигурации (.env) — до старта polling, а не на первом сообщении
    from app.llm.factory import get_llm_provider
    from app.presets import get_preset

    get_preset()
    get_llm_provider()

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher(storage=MemoryStorage())

    # Middleware
    dp.message.middleware(AuthMiddleware())
    dp.callback_query.middleware(AuthMiddleware())

    # Роутеры (порядок важен: start первым, query последним как fallback)
    dp.include_router(start_router)
    dp.include_router(keys_router)
    dp.include_router(docs_router)
    dp.include_router(update_router)  # ДО upload — FSM перехватывает F.document в replace_upload
    dp.include_router(upload_router)
    dp.include_router(law_router)      # /law, /lawstats — до query чтобы не перехватывался
    dp.include_router(quiz_router)    # квиз по документам
    dp.include_router(voice_router)   # голосовые → текст → RAG / подсказка команды
    dp.include_router(query_router)   # последним — fallback для текстовых сообщений

    # Lifecycle
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    logger.info("Bot starting polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
