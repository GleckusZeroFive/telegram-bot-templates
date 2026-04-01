# Contributing to Telegram Bot Templates

## Development Setup

1. Clone and install:

```bash
git clone https://github.com/GleckusZeroFive/telegram-bot-templates.git
cd telegram-bot-templates
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Start infrastructure:

```bash
docker compose -f docker-compose.dev.yml up -d
```

3. Configure:

```bash
cp .env.example .env
# Set BOT_TOKEN, LLM settings, etc.
```

## Project Structure

- `app/bot/` — Telegram bot handlers
- `app/core/` — RAG pipeline (search, chunking, embeddings)
- `app/llm/` — LLM integration
- `app/db/` — database models and migrations
- `app/presets/` — YAML preset configurations
- `app/config.py` — Pydantic settings
- `alembic/` — database migrations
- `tests/` — test suite

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with a clear message
6. Push and open a Pull Request

## Adding a New Preset

Presets define bot behavior via YAML. To add one:

1. Create a YAML file in `app/presets/` (use `default.yml` as reference)
2. Configure: system prompt, greeting, temperature, chunking params
3. Test with the bot to verify behavior
4. Update README if the preset adds a new use case

Existing presets: `corporate_faq`, `customer_support`, `legal_assistant`, `tutor`, `voice_assistant`

## Code Style

- Python 3.10+
- Async-first (aiogram)
- Type hints where practical
- Keep presets self-contained — all behavior differences should be in YAML, not code

## Reporting Issues

Open an issue with:
- Preset being used
- Steps to reproduce
- Expected vs actual behavior
