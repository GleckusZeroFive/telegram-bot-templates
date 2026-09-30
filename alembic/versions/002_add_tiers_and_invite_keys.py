"""Add tiers and invite_keys

Revision ID: 002
Revises: 001
Create Date: 2026-02-16

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Начальный unlimited-ключ для владельца
BOOTSTRAP_KEY = "RAGK-7FWN-XMBT"


def upgrade() -> None:
    # Новые поля в users
    op.add_column("users", sa.Column("tier", sa.String(20), server_default="free", nullable=False))
    op.add_column("users", sa.Column("tier_expires_at", sa.DateTime(), nullable=True))

    # Таблица invite_keys
    op.create_table(
        "invite_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("key", sa.String(14), unique=True, nullable=False),
        sa.Column("tier", sa.String(20), nullable=False),
        sa.Column("created_by_id", sa.Integer(),
                  sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("used_by_id", sa.Integer(),
                  sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # Вставляем начальный unlimited-ключ
    op.execute(
        sa.text(
            "INSERT INTO invite_keys (id, key, tier) "
            "VALUES (gen_random_uuid(), :key, 'unlimited')"
        ).bindparams(key=BOOTSTRAP_KEY)
    )


def downgrade() -> None:
    op.drop_table("invite_keys")
    op.drop_column("users", "tier_expires_at")
    op.drop_column("users", "tier")
