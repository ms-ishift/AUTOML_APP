"""PredictionJob Repository (IMPLEMENTATION_PLAN §2.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from repositories.models import PredictionJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


ALLOWED_INPUT_TYPES: frozenset[str] = frozenset({"form", "file"})
ALLOWED_STATUSES: frozenset[str] = frozenset({"pending", "running", "completed", "failed"})


def insert(
    session: Session,
    *,
    model_id: int,
    input_type: str,
    input_file_path: str | None = None,
    result_path: str | None = None,
    status: str = "pending",
) -> PredictionJob:
    if input_type not in ALLOWED_INPUT_TYPES:
        raise ValueError(f"허용되지 않는 input_type: {input_type}")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"허용되지 않는 status: {status}")

    job = PredictionJob(
        model_id=model_id,
        input_type=input_type,
        input_file_path=input_file_path,
        result_path=result_path,
        status=status,
    )
    session.add(job)
    session.flush()
    return job


def update_status(
    session: Session,
    prediction_job_id: int,
    status: str,
    *,
    result_path: str | None = None,
) -> PredictionJob | None:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"허용되지 않는 status: {status}")
    job = session.get(PredictionJob, prediction_job_id)
    if job is None:
        return None
    job.status = status
    if result_path is not None:
        job.result_path = result_path
    session.flush()
    return job


def get(session: Session, prediction_job_id: int) -> PredictionJob | None:
    return session.get(PredictionJob, prediction_job_id)


def list_by_model(
    session: Session,
    model_id: int,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[PredictionJob]:
    stmt = (
        select(PredictionJob)
        .where(PredictionJob.model_id == model_id)
        .order_by(PredictionJob.created_at.desc(), PredictionJob.prediction_job_id.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()
