# Telegram RAG Bot Templates

[![CI](https://github.com/GleckusZeroFive/telegram-bot-templates/actions/workflows/ci.yml/badge.svg)](https://github.com/GleckusZeroFive/telegram-bot-templates/actions/workflows/ci.yml)

A Telegram bot that answers questions from the user's own documents. One codebase, six
product variants selected by a YAML preset: document Q&A, corporate FAQ, customer support,
legal assistant, tutor and a voice-first field assistant. Russian-first: morphology-aware
keyword search, multilingual embeddings, Russian prompts.

## What it does

- **Ingestion** — PDF, DOCX, TXT, MD. Scanned PDF pages go through Tesseract OCR. Text is split
  into chunks that never cross page boundaries and carry page and section metadata.
- **Hybrid search** — every chunk is stored in Qdrant with a dense vector
  (`intfloat/multilingual-e5-large`, local) and a sparse BM25-style vector (pymorphy3
  lemmatization, IDF applied by Qdrant). Results are fused with Reciprocal Rank Fusion; because
  RRF has no score threshold, a separate cosine check drops answers when nothing is actually
  relevant.
- **Query preparation** — the question is rewritten into a self-contained search query using the
  dialogue history, and a hypothetical answer (HyDE) is embedded for the dense search. Both steps
  are optional, bounded by `LLM_AUX_TIMEOUT` and fall back to the original question.
- **Routing: chat by default, RAG when needed** — regex fast paths for explicit document requests
  and small talk, then a local ONNX intent classifier (fine-tuned ruBERT-tiny2:
  `rag` / `chat` / `followup`), then the LLM when the classifier is unsure (confidence < 0.75).
- **Answers** — streamed into Telegram with a paced buffer and flood-control handling, sources
  listed as file + pages. LLM output is converted to safe Telegram HTML.
- **Around the core** — voice messages (faster-whisper), follow-up context, document replace /
  append with backups and an LLM-written change summary, quizzes generated from the documents,
  tiers with daily limits and invite keys, optional search over a Russian law corpus
  (external service, not included).

## Architecture

```
Telegram ──► aiogram handlers ──► router: regex → ONNX classifier → LLM classifier
                                         │
                    chat / follow-up ◄───┴───► RAG
                                                │
             rewrite + HyDE (LLM) ──► dense + BM25 query ──► Qdrant (RRF) ──► relevance gate
                                                │
                              prompt from preset + context ──► LLM (streaming) ──► Telegram

Upload: file ──► parser (+OCR) ──► chunker ──► e5 embeddings + BM25 ──► Qdrant (collection per user)
State:  PostgreSQL — users, tiers, limits, invite keys, documents, query history (Alembic)
```

The LLM is any OpenAI-compatible API (OpenAI, OpenRouter, Groq, Ollama, vLLM, ...), with an
optional fallback provider for 429 / 5xx / timeouts. Embeddings, BM25, OCR, speech recognition
and intent classification run locally.

## Presets

Set `BOT_PRESET` in `.env`. A preset holds the bot name, system / chat / follow-up / classifier
prompts, user-facing messages, command descriptions and RAG keywords.

| Preset | Bot name | Use case |
|---|---|---|
| `default` | RAGozin | Document Q&A + Russian law search |
| `corporate_faq` | DocHelper | Internal documentation / HR policies |
| `customer_support` | SupportBot | Store FAQ: delivery, payment, returns |
| `legal_assistant` | LegalDoc | Contracts and legal documents |
| `tutor` | StudyBot | Study materials, quizzes |
| `voice_assistant` | FieldBot | Technical documentation, voice-first |

An unknown preset name stops the bot at startup with the list of available presets.

## Quick start (Docker Compose)

```bash
cp .env.example .env
# set TELEGRAM_BOT_TOKEN, POSTGRES_PASSWORD, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
docker compose up -d --build
docker compose logs -f bot
```

Compose starts PostgreSQL, Qdrant and the bot; the bot applies migrations on start. The first
build downloads about 3 GB of models (embeddings, Whisper `small`, intent classifier) and bakes
them into the image, so the container then runs without access to Hugging Face.

## LLM configuration

| Provider | `LLM_BASE_URL` | `LLM_MODEL` example | `LLM_EXTRA_BODY` |
|---|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-v4.1-flash` | `{"reasoning": {"enabled": false}}` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4.1-mini` | — |
| Ollama on the host | `http://host.docker.internal:11434/v1` | `qwen3:8b` | — |

For Ollama on the Docker host, start with `-f docker-compose.yml -f docker-compose.ollama.yml`:
the override maps `host.docker.internal`, which Linux engines do not resolve by default.

**Reasoning models.** Most current models think before answering, and the thinking counts
against `max_tokens`. The intent classifier asks for one word with a 20-token budget, query
rewriting for 100 tokens — with reasoning on, those budgets are spent on thinking and the answer
comes back empty. `LLM_EXTRA_BODY` adds provider-specific JSON to every request to switch
reasoning off; some models (for example, several Gemini and GLM releases on OpenRouter) do not
allow that and are a poor fit for this bot.

`LLM_FALLBACK_MODEL` (plus optional `LLM_FALLBACK_BASE_URL` / `LLM_FALLBACK_API_KEY` /
`LLM_FALLBACK_EXTRA_BODY`) enables a second provider. Fallback happens only before the first
token is streamed, so a user never gets two answers glued together. Appending an image in
"vision" mode requires a model that accepts images.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_intent_model.py      # optional: without it the LLM classifies intents

docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres qdrant
alembic upgrade head
python -m app.main
python healthcheck.py                        # PostgreSQL, Qdrant, Telegram, embeddings, LLM
```

The dev override exposes PostgreSQL and Qdrant on localhost and mounts `app/` into the bot
container (`docker compose restart bot` picks up code changes). A mounted `app/` replaces the
copy inside the image, so download the intent model locally if you run the bot container in dev
mode.

### Tests

```bash
pytest                                        # unit tests, in-memory Qdrant
TEST_DATABASE_URL=postgresql+asyncpg://ragbot:<password>@localhost:5432/ragbot pytest -m db
ruff check .
```

The `db` tests check the daily limit and invite-key activation under concurrent requests on a
real PostgreSQL with migrations applied; without `TEST_DATABASE_URL` they are skipped. Tests
that need the embedding model skip when it is not available. CI runs lint and the full suite,
including the PostgreSQL tests.

## Project structure

```
app/
  bot/          aiogram handlers, middleware, keyboards, HTML formatting
  core/         parsing, chunking, embeddings, BM25 encoder, indexer, retriever,
                generator, intent classifiers, transcription, RAG pipeline
  llm/          OpenAI-compatible provider: retries, fallback, response cache
  db/           SQLAlchemy models and repositories
  presets/      YAML presets and their loader
alembic/        database migrations
scripts/        model download
tests/
```

## Limitations

- Dialogue context, FSM state and quiz state live in process memory: they reset on restart and
  the bot is meant to run as a single instance.
- One database session is held for the whole update, including answer streaming.
- The Qdrant client is synchronous; calls are moved to worker threads.
- Law corpus search needs a separate service and is disabled by default.

## License

Source Available License — see [LICENSE](LICENSE). Personal and educational use is permitted.
Commercial use requires explicit permission.

## Author

**Vladislav Pestov** — [gleckus.dev](https://gleckus.dev) · GitHub: [GleckusZeroFive](https://github.com/GleckusZeroFive)
