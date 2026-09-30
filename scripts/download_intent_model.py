"""Скачать веса ONNX-классификатора намерений с Hugging Face.

Модель: ruBERT-tiny2, дообученная на rag / chat / followup
(https://huggingface.co/Gleckus/intent-classifier-rubert-tiny2).
Веса не хранятся в git; токенизатор лежит в репозитории.

    python scripts/download_intent_model.py

Без модели бот работает: неоднозначные сообщения классифицирует LLM.
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Gleckus/intent-classifier-rubert-tiny2"
# Фиксированная ревизия: веса должны соответствовать токенизатору из репозитория
REVISION = "4e1738cd4d4af59a220bf86f19d8d98d6bea0ca7"
FILES = ("model.onnx", "model.onnx.data")
TARGET_DIR = Path(__file__).resolve().parent.parent / "app" / "core" / "intent_model"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = TARGET_DIR / name
        if target.exists():
            print(f"{name}: already present")
            continue
        cached = hf_hub_download(repo_id=REPO_ID, filename=name, revision=REVISION)
        shutil.copyfile(cached, target)
        print(f"{name}: {target.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
