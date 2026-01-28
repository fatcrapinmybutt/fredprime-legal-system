"""initial

Revision ID: b978fb447583
Revises:
Create Date: 2025-07-14 05:17:23.871785

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b978fb447583"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "forms",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="forms",
    )
    op.create_table(
        "facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("info", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="knowledge",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("facts", schema="knowledge")
    op.drop_table("forms", schema="forms")
