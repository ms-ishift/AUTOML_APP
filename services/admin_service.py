"""Admin Service (IMPLEMENTATION_PLAN §4.7, FR-090~093).

운영 대시보드(`pages/07_admin.py`) 가 필요로 하는 교차 도메인 집계를 제공한다. 기존
`project_service`/`training_service`/`prediction_service` 가 **단일 프로젝트 단위**
유스케이스에 초점이 맞춰져 있어, 관리자 뷰처럼 여러 프로젝트를 가로지르는 쿼리를
여기에 집약한다.

책임:
- 통계 집계: `get_stats()` — projects/datasets/training_jobs/models/predictions +
  학습·예측 실패 건수 (status='failed')
- 학습 이력: `list_training_history(...)` — 필터(프로젝트/상태/기간) + Model 집계(성공/실패/베스트)
- 예측 이력: `list_prediction_history(...)` — 모델 → 학습 잡 → 프로젝트 역조인
- 최근 실패 로그: `list_recent_failures(...)` — AuditLog 에서 '*_failed' action_type 만 집계

규약 (`.cursor/rules/service-layer.mdc`):
- Streamlit 의존 금지, 반환은 모두 DTO.
- SQL 은 가능하면 집계 함수로 처리해 N+1 회피. MVP 규모에서는 간단한 서브쿼리/JOIN 수준 유지.
- 입력 검증: 음수 limit, 잘못된 status 는 `ValidationError` 로 거부.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, case, func, select

from repositories import audit_repository
from repositories.base import session_scope
from repositories.models import (
    AuditLog,
    Dataset,
    Model,
    PredictionJob,
    Project,
    TrainingJob,
)
from services.dto import (
    AdminStatsDTO,
    AuditLogEntryDTO,
    PredictionHistoryRowDTO,
    TrainingHistoryRowDTO,
)
from utils.errors import ValidationError
from utils.log_utils import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session

logger = get_logger(__name__)

DEFAULT_HISTORY_LIMIT = 200
DEFAULT_FAILURE_LIMIT = 50

_TRAINING_STATUSES: frozenset[str] = frozenset({"pending", "running", "completed", "failed"})
_PREDICTION_STATUSES: frozenset[str] = _TRAINING_STATUSES


# ------------------------------------------------------------------ helpers


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise ValidationError("limit 은 1 이상이어야 합니다.")
    return int(limit)


def _validate_status(status: str | None, allowed: frozenset[str]) -> str | None:
    if status is None:
        return None
    if status not in allowed:
        raise ValidationError(f"허용되지 않는 status: {status} (가능: {sorted(allowed)})")
    return status


def _duration_ms(started_at: datetime | None, ended_at: datetime | None) -> int | None:
    if started_at is None or ended_at is None:
        return None
    delta = ended_at - started_at
    return max(0, int(delta.total_seconds() * 1000))


# ---------------------------------------------------------------- use-cases


def get_stats() -> AdminStatsDTO:
    """전역 카드에 표시할 단순 카운트 집계 (FR-090)."""
    with session_scope() as session:
        projects = _scalar_count(session, Project.project_id)
        datasets = _scalar_count(session, Dataset.dataset_id)
        training_jobs = _scalar_count(session, TrainingJob.training_job_id)
        models = _scalar_count(session, Model.model_id)
        predictions = _scalar_count(session, PredictionJob.prediction_job_id)

        training_failures = int(
            session.execute(
                select(func.count(TrainingJob.training_job_id)).where(
                    TrainingJob.status == "failed"
                )
            ).scalar_one()
        )
        prediction_failures = int(
            session.execute(
                select(func.count(PredictionJob.prediction_job_id)).where(
                    PredictionJob.status == "failed"
                )
            ).scalar_one()
        )

    return AdminStatsDTO(
        projects=projects,
        datasets=datasets,
        training_jobs=training_jobs,
        models=models,
        predictions=predictions,
        training_failures=training_failures,
        prediction_failures=prediction_failures,
    )


def _scalar_count(session: Session, column: Any) -> int:
    return int(session.execute(select(func.count(column))).scalar_one())


def list_training_history(
    *,
    project_id: int | None = None,
    status: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[TrainingHistoryRowDTO]:
    """학습 이력 테이블 (FR-091). 프로젝트명 + Model 집계(성공/실패/베스트) 포함.

    정렬: `created_at DESC`. 성능상 N+1 을 피하기 위해 Model 집계는 단일 쿼리로 처리한다.
    """
    _validate_limit(limit)
    _validate_status(status, _TRAINING_STATUSES)

    with session_scope() as session:
        # 모델 집계 서브쿼리: training_job_id 별 성공/실패 수
        model_status_case = case(
            (Model.metric_score.is_(None), 1),
            else_=0,
        )
        model_agg = (
            select(
                Model.training_job_id.label("tj_id"),
                func.count(Model.model_id).label("n_total"),
                func.sum(model_status_case).label("n_failed"),
            )
            .group_by(Model.training_job_id)
            .subquery()
        )

        # 베스트 모델 서브쿼리: is_best=True 인 Model 1건
        best_model = (
            select(
                Model.training_job_id.label("tj_id"),
                Model.algorithm_name.label("best_algo"),
                Model.metric_score.label("best_metric"),
            )
            .where(Model.is_best.is_(True))
            .subquery()
        )

        stmt = (
            select(
                TrainingJob,
                Project.project_id.label("p_id"),
                Project.project_name.label("p_name"),
                model_agg.c.n_total,
                model_agg.c.n_failed,
                best_model.c.best_algo,
                best_model.c.best_metric,
            )
            .join(Project, Project.project_id == TrainingJob.project_id)
            .outerjoin(model_agg, model_agg.c.tj_id == TrainingJob.training_job_id)
            .outerjoin(best_model, best_model.c.tj_id == TrainingJob.training_job_id)
        )
        if project_id is not None:
            stmt = stmt.where(TrainingJob.project_id == int(project_id))
        if status is not None:
            stmt = stmt.where(TrainingJob.status == status)
        if since is not None:
            stmt = stmt.where(TrainingJob.created_at >= since)
        if until is not None:
            stmt = stmt.where(TrainingJob.created_at < until)

        stmt = stmt.order_by(
            TrainingJob.created_at.desc(),
            TrainingJob.training_job_id.desc(),
        ).limit(limit)

        rows: list[TrainingHistoryRowDTO] = []
        for (
            job,
            p_id,
            p_name,
            n_total,
            n_failed,
            best_algo,
            best_metric,
        ) in session.execute(stmt).all():
            n_total_int = int(n_total or 0)
            n_failed_int = int(n_failed or 0)
            rows.append(
                TrainingHistoryRowDTO(
                    id=job.training_job_id,
                    project_id=int(p_id),
                    project_name=str(p_name),
                    dataset_id=job.dataset_id,
                    task_type=job.task_type,
                    target_column=job.target_column,
                    metric_key=job.metric_key,
                    status=job.status,
                    started_at=job.started_at,
                    ended_at=job.ended_at,
                    duration_ms=_duration_ms(job.started_at, job.ended_at),
                    n_models_success=max(0, n_total_int - n_failed_int),
                    n_models_failed=n_failed_int,
                    best_algo=str(best_algo) if best_algo is not None else None,
                    best_metric=(float(best_metric) if best_metric is not None else None),
                )
            )
        return rows


def list_prediction_history(
    *,
    project_id: int | None = None,
    status: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[PredictionHistoryRowDTO]:
    """예측 이력 테이블 (FR-091). 모델 → 학습 잡 → 프로젝트 조인."""
    _validate_limit(limit)
    _validate_status(status, _PREDICTION_STATUSES)

    with session_scope() as session:
        stmt = (
            select(
                PredictionJob,
                Model.algorithm_name.label("algo"),
                Project.project_id.label("p_id"),
                Project.project_name.label("p_name"),
            )
            .join(Model, Model.model_id == PredictionJob.model_id)
            .join(TrainingJob, TrainingJob.training_job_id == Model.training_job_id)
            .join(Project, Project.project_id == TrainingJob.project_id)
        )
        if project_id is not None:
            stmt = stmt.where(TrainingJob.project_id == int(project_id))
        if status is not None:
            stmt = stmt.where(PredictionJob.status == status)
        if since is not None:
            stmt = stmt.where(PredictionJob.created_at >= since)
        if until is not None:
            stmt = stmt.where(PredictionJob.created_at < until)

        stmt = stmt.order_by(
            PredictionJob.created_at.desc(),
            PredictionJob.prediction_job_id.desc(),
        ).limit(limit)

        rows: list[PredictionHistoryRowDTO] = []
        for job, algo, p_id, p_name in session.execute(stmt).all():
            rows.append(
                PredictionHistoryRowDTO(
                    id=job.prediction_job_id,
                    model_id=job.model_id,
                    algorithm_name=str(algo),
                    project_id=int(p_id),
                    project_name=str(p_name),
                    input_type=job.input_type,
                    status=job.status,
                    created_at=job.created_at,
                    input_file_path=job.input_file_path,
                    result_path=job.result_path,
                )
            )
        return rows


def list_recent_failures(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = DEFAULT_FAILURE_LIMIT,
) -> list[AuditLogEntryDTO]:
    """최근 실패 감사 로그 요약 (FR-093).

    규칙:
    - `action_type` 이 `*_failed` 로 끝나는 항목만 수집 (프로젝트/데이터셋/학습/모델/예측).
    - 기본 정렬: `action_time DESC`.
    """
    _validate_limit(limit)

    with session_scope() as session:
        stmt = select(AuditLog).where(AuditLog.action_type.like("%_failed"))
        conditions = []
        if since is not None:
            conditions.append(AuditLog.action_time >= since)
        if until is not None:
            conditions.append(AuditLog.action_time < until)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(AuditLog.action_time.desc(), AuditLog.audit_log_id.desc()).limit(limit)

        return [
            AuditLogEntryDTO(
                id=entry.audit_log_id,
                action_type=entry.action_type,
                target_type=entry.target_type,
                target_id=entry.target_id,
                action_time=entry.action_time,
                detail=dict(entry.detail_json or {}),
            )
            for entry in session.execute(stmt).scalars().all()
        ]


# -------- 참조 편의: 감사 로그 단건을 raw 로 필요한 호출부가 있을 수 있어 래핑
def list_logs(**kwargs: Any) -> list[AuditLogEntryDTO]:  # pragma: no cover - passthrough
    """`audit_repository.list_logs` 의 얇은 DTO 래퍼."""
    with session_scope() as session:
        entries = audit_repository.list_logs(session, **kwargs)
        return [
            AuditLogEntryDTO(
                id=e.audit_log_id,
                action_type=e.action_type,
                target_type=e.target_type,
                target_id=e.target_id,
                action_time=e.action_time,
                detail=dict(e.detail_json or {}),
            )
            for e in entries
        ]


__all__ = [
    "DEFAULT_FAILURE_LIMIT",
    "DEFAULT_HISTORY_LIMIT",
    "get_stats",
    "list_logs",
    "list_prediction_history",
    "list_recent_failures",
    "list_training_history",
]
