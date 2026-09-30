#!/bin/bash
set -e

# PostgreSQL and Qdrant are separate compose services; compose starts the bot
# only after the database healthcheck passes.
echo "==> Running Alembic migrations..."
alembic upgrade head

echo "==> Starting bot..."
exec python -m app.main
