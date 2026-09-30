"""Rename tier 'unlimited' to 'admin'

Revision ID: 005
Revises: 004
Create Date: 2026-02-20

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE users SET tier = 'admin' WHERE tier = 'unlimited'")
    op.execute("UPDATE invite_keys SET tier = 'admin' WHERE tier = 'unlimited'")


def downgrade() -> None:
    op.execute("UPDATE users SET tier = 'unlimited' WHERE tier = 'admin'")
    op.execute("UPDATE invite_keys SET tier = 'unlimited' WHERE tier = 'admin'")
