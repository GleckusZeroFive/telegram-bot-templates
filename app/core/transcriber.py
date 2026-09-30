import asyncio
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Ошибка транскрипции голосового сообщения."""


# Singleton: модель загружается один раз и живёт в памяти
_model = None
_load_lock = asyncio.Lock()


async def _ensure_model():
    """Lazy-load faster-whisper модели (singleton с asyncio.Lock)."""
    global _model
    if _model is not None:
        return _model

    async with _load_lock:
        if _model is not None:
            return _model

        if not shutil.which("ffmpeg"):
            raise TranscriptionError("ffmpeg не найден в системе")

        from faster_whisper import WhisperModel

        logger.info(
            "Загрузка whisper модели '%s' (CPU, int8)...",
            settings.whisper_model_size,
        )

        def _load():
            return WhisperModel(
                settings.whisper_model_size,
                device="cpu",
                compute_type="int8",
            )

        _model = await asyncio.to_thread(_load)
        logger.info("Whisper модель загружена")
        return _model


async def _convert_ogg_to_wav(ogg_path: Path, wav_path: Path) -> None:
    """Конвертация OGG/Opus → WAV 16kHz mono через ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-i", str(ogg_path),
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        str(wav_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.warning("ffmpeg stderr: %s", stderr.decode(errors="replace"))
        raise TranscriptionError("Ошибка конвертации аудио")


def _run_transcription(model, wav_path: Path) -> str:
    """Синхронная транскрипция (для asyncio.to_thread)."""
    segments, _info = model.transcribe(
        str(wav_path),
        language=settings.whisper_language,
        beam_size=5,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


async def transcribe(ogg_bytes: bytes) -> str:
    """Транскрибировать OGG/Opus аудио в текст.

    Args:
        ogg_bytes: содержимое .ogg файла (скачанное из Telegram).

    Returns:
        Распознанный текст (может быть пустой строкой).

    Raises:
        TranscriptionError: ffmpeg/whisper ошибка.
    """
    model = await _ensure_model()

    ogg_fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
    wav_path = ogg_path.replace(".ogg", ".wav")

    try:
        with open(ogg_fd, "wb") as f:
            f.write(ogg_bytes)

        await _convert_ogg_to_wav(Path(ogg_path), Path(wav_path))
        text = await asyncio.to_thread(_run_transcription, model, Path(wav_path))
        return text.strip()
    finally:
        for p in (ogg_path, wav_path):
            with contextlib.suppress(OSError):
                Path(p).unlink(missing_ok=True)
