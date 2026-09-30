"""Add document update fields (full_text, version, parent_id, is_backup)

Revision ID: 003
Revises: 002
Create Date: 2026-02-17

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("full_text", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("updated_at", sa.DateTime(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("is_backup", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_foreign_key(
        "fk_documents_parent_id",
        "documents",
        "documents",
        ["parent_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_parent_id", "documents", type_="foreignkey")
    op.drop_column("documents", "is_backup")
    op.drop_column("documents", "parent_id")
    op.drop_column("documents", "version")
    op.drop_column("documents", "updated_at")
    op.drop_column("documents", "full_text")
