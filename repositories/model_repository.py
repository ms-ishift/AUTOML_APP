"""Model Repository (IMPLEMENTATION_PLAN §2.3).

아티팩트 저장 보상 로직 (§4.3a):
- Service 는 먼저 ``bulk_insert`` 로 DB 레코드를 flush 해 ``model_id`` 를 확보하고,
- 이후 디스크에 파일을 쓰고 ``update_paths`` 로 경로를 기록한다.
- 파일 저장 실패 시 Service 가 해당 레코드를 ``delete`` 로 롤백한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy import update as sa_update

from repositories.models import Model, TrainingJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def bulk_insert(
    session: Session,
    training_job_id: int,
    rows: list[dict[str, Any]],
) -> list[Model]:
    """여러 Model 레코드를 flush 하고 id 확정된 엔터티 목록을 반환.

    각 row 는 다음 키를 포함::

        {
            "model_name": str,
            "algorithm_name": str,
            "metric_score": float | None,
            "metric_summary_json": dict | None,
            "feature_schema_json": dict | None,
        }

    경로(``model_path``, ``preprocessing_path``) 는 ``update_paths`` 로 나중에 채운다.
    """
    entities: list[Model] = []
    for row in rows:
        entities.append(
            Model(
                training_job_id=training_job_id,
                model_name=row["model_name"],
                algorithm_name=row["algorithm_name"],
                metric_score=row.get("metric_score"),
                metric_summary_json=row.get("metric_summary_json"),
                feature_schema_json=row.get("feature_schema_json"),
                is_best=False,
            )
        )
    session.add_all(entities)
    session.flush()
    return entities


def update_paths(
    session: Session,
    model_id: int,
    *,
    model_path: str | None = None,
    preprocessing_path: str | None = None,
) -> Model | None:
    model = session.get(Model, model_id)
    if model is None:
        return None
    if model_path is not None:
        model.model_path = model_path
    if preprocessing_path is not None:
        model.preprocessing_path = preprocessing_path
    session.flush()
    return model


def get(session: Session, model_id: int) -> Model | None:
    return session.get(Model, model_id)


def delete(session: Session, model_id: int) -> bool:
    model = session.get(Model, model_id)
    if model is None:
        return False
    session.delete(model)
    session.flush()
    return True


def list_by_training_job(session: Session, training_job_id: int) -> Sequence[Model]:
    stmt = (
        select(Model)
        .where(Model.training_job_id == training_job_id)
        .order_by(Model.metric_score.desc().nulls_last(), Model.model_id.asc())
    )
    return session.execute(stmt).scalars().all()


def list_by_project(session: Session, project_id: int) -> Sequence[Model]:
    stmt = (
        select(Model)
        .join(TrainingJob, TrainingJob.training_job_id == Model.training_job_id)
        .where(TrainingJob.project_id == project_id)
        .order_by(Model.created_at.desc(), Model.model_id.desc())
    )
    return session.execute(stmt).scalars().all()


def mark_best(session: Session, training_job_id: int, best_model_id: int) -> Model | None:
    """동일 TrainingJob 내에서 하나의 모델만 is_best=True 로 설정."""
    session.execute(
        sa_update(Model).where(Model.training_job_id == training_job_id).values(is_best=False)
    )
    best = session.get(Model, best_model_id)
    if best is None or best.training_job_id != training_job_id:
        return None
    best.is_best = True
    session.flush()
    return best
