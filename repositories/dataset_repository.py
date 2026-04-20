"""Dataset Repository (IMPLEMENTATION_PLAN §2.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from repositories.models import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def insert(
    session: Session,
    *,
    project_id: int,
    file_name: str,
    file_path: str,
    row_count: int,
    column_count: int,
    schema_json: dict[str, Any] | None = None,
) -> Dataset:
    dataset = Dataset(
        project_id=project_id,
        file_name=file_name,
        file_path=file_path,
        row_count=row_count,
        column_count=column_count,
        schema_json=schema_json,
    )
    session.add(dataset)
    session.flush()
    return dataset


def get(session: Session, dataset_id: int) -> Dataset | None:
    return session.get(Dataset, dataset_id)


def list_by_project(
    session: Session,
    project_id: int,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[Dataset]:
    stmt = (
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc(), Dataset.dataset_id.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()


def delete(session: Session, dataset_id: int) -> bool:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        return False
    session.delete(dataset)
    session.flush()
    return True
