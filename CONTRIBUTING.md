# Contributing to Telegram Bot Templates

## Development Setup

1. Clone and install:

```bash
git clone https://github.com/GleckusZeroFive/telegram-bot-templates.git
cd telegram-bot-templates
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_intent_model.py
```

2. Configure:

```bash
cp .env.example .env
# Set TELEGRAM_BOT_TOKEN, POSTGRES_PASSWORD and the LLM_* settings
```

3. Start infrastructure and apply migrations:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres qdrant
alembic upgrade head
```

## Project Structure

- `app/bot/` — Telegram bot handlers
- `app/core/` — RAG pipeline (search, chunking, embeddings)
- `app/llm/` — LLM integration
- `app/db/` — database models and repositories
- `app/presets/` — YAML preset configurations
- `app/config.py` — Pydantic settings
- `alembic/` — database migrations
- `tests/` — test suite

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run checks: `ruff check .` and `pytest` (see README for the PostgreSQL tests)
5. Commit with a clear message
6. Push and open a Pull Request

## Adding a New Preset

Presets define bot behavior via YAML. To add one:

1. Create a YAML file in `app/presets/` (use `default.yml` as reference)
2. Configure: bot name, prompts (system, chat, follow-up, classifier), messages, commands, RAG keywords
3. Keep the `{context}` placeholder in the system prompt and `{doc_list}` in the classifier prompt
4. Test with the bot to verify behavior
5. Update README if the preset adds a new use case

Existing presets: `default`, `corporate_faq`, `customer_support`, `legal_assistant`, `tutor`, `voice_assistant`

## Code Style

- Python 3.12
- Async-first (aiogram)
- Type hints where practical
- Anything user-controlled or model-generated goes through `app/bot/formatting.py` before it is
  placed into an HTML message
- Keep presets self-contained — all behavior differences should be in YAML, not code

## Reporting Issues

Open an issue with:
- Preset being used
- Steps to reproduce
- Expected vs actual behavior
