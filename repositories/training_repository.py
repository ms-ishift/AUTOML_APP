"""TrainingJob Repository (IMPLEMENTATION_PLAN §2.3).

상태 전이: ``pending → running → completed|failed``. Service 가 전이 시점을 제어한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from repositories.models import TrainingJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


ALLOWED_STATUSES: frozenset[str] = frozenset({"pending", "running", "completed", "failed"})


def insert(
    session: Session,
    *,
    project_id: int,
    dataset_id: int,
    task_type: str,
    target_column: str,
    metric_key: str,
    excluded_columns: list[str] | None = None,
) -> TrainingJob:
    job = TrainingJob(
        project_id=project_id,
        dataset_id=dataset_id,
        task_type=task_type,
        target_column=target_column,
        metric_key=metric_key,
        excluded_columns_json=excluded_columns or [],
        status="pending",
    )
    session.add(job)
    session.flush()
    return job


def get(session: Session, training_job_id: int) -> TrainingJob | None:
    return session.get(TrainingJob, training_job_id)


def list_by_project(
    session: Session,
    project_id: int,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[TrainingJob]:
    stmt = (
        select(TrainingJob)
        .where(TrainingJob.project_id == project_id)
        .order_by(TrainingJob.created_at.desc(), TrainingJob.training_job_id.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()


def update_status(
    session: Session,
    training_job_id: int,
    status: str,
    *,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> TrainingJob | None:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"허용되지 않는 status: {status}")

    job = session.get(TrainingJob, training_job_id)
    if job is None:
        return None

    job.status = status
    if started_at is not None:
        job.started_at = started_at
    if ended_at is not None:
        job.ended_at = ended_at
    # status=running 시 started_at 자동 세팅, completed/failed 시 ended_at 자동 세팅
    if status == "running" and job.started_at is None:
        job.started_at = datetime.utcnow()
    if status in {"completed", "failed"} and job.ended_at is None:
        job.ended_at = datetime.utcnow()

    session.flush()
    return job


def append_run_log(
    session: Session,
    training_job_id: int,
    line: str,
) -> TrainingJob | None:
    """run_log 에 줄 단위로 append. 멀티라인 로그 누적용."""
    job = session.get(TrainingJob, training_job_id)
    if job is None:
        return None

    if not line.endswith("\n"):
        line = line + "\n"
    job.run_log = (job.run_log or "") + line
    session.flush()
    return job
