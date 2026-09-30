"""Загрузчик пресетов бота из YAML-файлов.

Пресет определяет: имя бота, промпты, сообщения, фичи, ключевые слова.
Выбирается через BOT_PRESET в .env (default = "default").
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PresetPrompts:
    system: str = ""
    chat: str = ""
    followup: str = ""
    classifier: str = ""
    hypothetical: str = ""
    rewrite: str = ""
    system_strict: str = ""
    followup_strict: str = ""


@dataclass(frozen=True)
class PresetMessages:
    start: str = ""
    help: str = ""
    about: str = ""


@dataclass(frozen=True)
class PresetFeatures:
    law_search: bool = False
    voice: bool = True
    ocr: bool = True


@dataclass(frozen=True)
class PresetCommands:
    """Переопределение команд бота. Ключ = команда без /, значение = описание.
    Если пусто — используются дефолтные из commands.py."""
    items: dict[str, str] = field(default_factory=dict)
    short: str = ""  # Краткий список для SYSTEM_PROMPT


@dataclass(frozen=True)
class Preset:
    name: str = "RAGozin"
    description: str = "AI-ассистент по документам"
    prompts: PresetPrompts = field(default_factory=PresetPrompts)
    messages: PresetMessages = field(default_factory=PresetMessages)
    features: PresetFeatures = field(default_factory=PresetFeatures)
    commands: PresetCommands = field(default_factory=PresetCommands)
    rag_keywords: list[str] = field(default_factory=list)


def _load_yaml(preset_name: str) -> dict[str, Any]:
    """Загрузить YAML-файл пресета."""
    path = _PRESETS_DIR / f"{preset_name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Пресет {preset_name} не найден: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_preset(data: dict[str, Any]) -> Preset:
    """Распарсить YAML-данные в Preset."""
    prompts_data = data.get("prompts", {})
    messages_data = data.get("messages", {})
    features_data = data.get("features", {})
    commands_data = data.get("commands", {})

    return Preset(
        name=data.get("name", "RAGozin"),
        description=data.get("description", "AI-ассистент по документам"),
        prompts=PresetPrompts(
            system=prompts_data.get("system", ""),
            chat=prompts_data.get("chat", ""),
            followup=prompts_data.get("followup", ""),
            classifier=prompts_data.get("classifier", ""),
            hypothetical=prompts_data.get("hypothetical", ""),
            rewrite=prompts_data.get("rewrite", ""),
            system_strict=prompts_data.get("system_strict", ""),
            followup_strict=prompts_data.get("followup_strict", ""),
        ),
        messages=PresetMessages(
            start=messages_data.get("start", ""),
            help=messages_data.get("help", ""),
            about=messages_data.get("about", ""),
        ),
        features=PresetFeatures(
            law_search=features_data.get("law_search", False),
            voice=features_data.get("voice", True),
            ocr=features_data.get("ocr", True),
        ),
        commands=PresetCommands(
            items=commands_data.get("items", {}),
            short=commands_data.get("short", ""),
        ),
        rag_keywords=data.get("rag_keywords", []),
    )


_preset_cache: Preset | None = None


def get_preset() -> Preset:
    """Получить текущий пресет (синглтон, загружается один раз)."""
    global _preset_cache
    if _preset_cache is not None:
        return _preset_cache

    from app.config import settings
    preset_name = settings.bot_preset

    # Опечатка в BOT_PRESET должна останавливать запуск: пустой Preset() без промптов
    # давал бота, который стартует и отвечает мусором
    try:
        data = _load_yaml(preset_name)
    except FileNotFoundError as e:
        available = ", ".join(sorted(p.stem for p in _PRESETS_DIR.glob("*.yml")))
        raise RuntimeError(
            f"BOT_PRESET={preset_name!r} не найден. Доступные пресеты: {available}"
        ) from e

    _preset_cache = _parse_preset(data)
    logger.info("Пресет %s загружен: name=%s", preset_name, _preset_cache.name)
    return _preset_cache
