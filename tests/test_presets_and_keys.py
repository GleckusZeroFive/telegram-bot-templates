"""Пресеты и инвайт-ключи: проверки без БД."""

import re
from pathlib import Path

import pytest

from app.db.repositories import invite_key
from app.presets import loader

_PRESETS = sorted(p.stem for p in Path(loader.__file__).parent.glob("*.yml"))


def test_six_presets_are_shipped():
    assert _PRESETS == [
        "corporate_faq", "customer_support", "default",
        "legal_assistant", "tutor", "voice_assistant",
    ]


@pytest.mark.parametrize("name", _PRESETS)
def test_preset_loads_with_prompts(name, monkeypatch):
    monkeypatch.setattr(loader, "_preset_cache", None)
    monkeypatch.setattr("app.config.settings.bot_preset", name)

    preset = loader.get_preset()

    assert preset.name
    assert "{context}" in preset.prompts.system
    assert preset.prompts.chat
    assert preset.prompts.followup
    assert "{doc_list}" in preset.prompts.classifier


def test_unknown_preset_fails_fast(monkeypatch):
    """Раньше опечатка в BOT_PRESET давала бота с пустыми промптами."""
    monkeypatch.setattr(loader, "_preset_cache", None)
    monkeypatch.setattr("app.config.settings.bot_preset", "corporte_faq")

    with pytest.raises(RuntimeError, match="corporate_faq"):
        loader.get_preset()


def test_generated_key_format_and_alphabet():
    for _ in range(200):
        key = invite_key.generate_key()
        assert re.fullmatch(r"[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}", key)


def test_key_generation_uses_csprng(monkeypatch):
    """Ключ даёт admin-тариф — предсказуемый random недопустим."""
    calls = []
    real_choice = invite_key.secrets.choice
    monkeypatch.setattr(invite_key.secrets, "choice", lambda seq: calls.append(1) or real_choice(seq))
    invite_key.generate_key()
    assert len(calls) == 12


def test_mask_key_hides_secret_part():
    assert invite_key.mask_key("ABCD-EFGH-JKMN") == "ABCD-****-****"
