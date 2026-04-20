"""admin_service 단위 테스트.

- stats/history/failures 를 end-to-end 시드(프로젝트·데이터셋·학습·예측 감사 로그)로 검증.
- `@pytest.mark.slow` 는 실제 `run_training` 을 수반하는 해피패스 1건에만 적용.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml.schemas import TrainingConfig
from repositories import audit_repository
from repositories.base import session_scope
from repositories.models import PredictionJob
from services import (
    admin_service,
    dataset_service,
    project_service,
    training_service,
)
from utils.errors import ValidationError


@dataclass
class FakeUpload:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


# --------------------------------------------------------------------- stats


def test_get_stats_empty_database_returns_zero(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    stats = admin_service.get_stats()
    assert stats.projects == 0
    assert stats.datasets == 0
    assert stats.training_jobs == 0
    assert stats.models == 0
    assert stats.predictions == 0
    assert stats.training_failures == 0
    assert stats.prediction_failures == 0


@pytest.mark.slow
def test_get_stats_counts_after_training(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project = project_service.create_project("stats-cls")
    ds = dataset_service.upload_dataset(
        project.id,
        FakeUpload(name=classification_csv.name, data=classification_csv.read_bytes()),
    )
    training_service.run_training(
        TrainingConfig(
            dataset_id=ds.id,
            task_type="classification",
            target_column="species",
        )
    )
    stats = admin_service.get_stats()
    assert stats.projects == 1
    assert stats.datasets == 1
    assert stats.training_jobs == 1
    assert stats.models >= 1


# ---------------------------------------------------------- training history


@pytest.mark.slow
def test_training_history_contains_joined_project_name_and_aggregates(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project = project_service.create_project("hist-cls")
    ds = dataset_service.upload_dataset(
        project.id,
        FakeUpload(name=classification_csv.name, data=classification_csv.read_bytes()),
    )
    training_service.run_training(
        TrainingConfig(
            dataset_id=ds.id,
            task_type="classification",
            target_column="species",
        )
    )

    rows = admin_service.list_training_history()
    assert len(rows) == 1
    row = rows[0]
    assert row.project_id == project.id
    assert row.project_name == "hist-cls"
    assert row.task_type == "classification"
    assert row.status == "completed"
    assert row.n_models_success >= 1
    assert row.best_algo is not None
    assert row.best_metric is not None
    assert row.duration_ms is not None and row.duration_ms >= 0


def _insert_dataset(session, project_id: int, name: str = "ds.csv"):
    from repositories.models import Dataset

    ds = Dataset(
        project_id=project_id,
        file_name=name,
        file_path=f"/tmp/{name}",
        row_count=10,
        column_count=3,
    )
    session.add(ds)
    session.flush()
    return ds


def test_training_history_filters_by_project_status_and_period(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    """필터 조합 동작을 실제 training 없이 최소 데이터로 검증."""
    from repositories import training_repository
    from repositories.base import session_scope

    p1 = project_service.create_project("proj-A")
    p2 = project_service.create_project("proj-B")

    with session_scope() as session:
        ds_a = _insert_dataset(session, p1.id, "a.csv")
        ds_b = _insert_dataset(session, p2.id, "b.csv")
        # 프로젝트 A: 완료 1, 실패 1
        a1 = training_repository.insert(
            session,
            project_id=p1.id,
            dataset_id=ds_a.dataset_id,
            task_type="classification",
            target_column="y",
            metric_key="accuracy",
        )
        training_repository.update_status(session, a1.training_job_id, "completed")
        a2 = training_repository.insert(
            session,
            project_id=p1.id,
            dataset_id=ds_a.dataset_id,
            task_type="classification",
            target_column="y",
            metric_key="accuracy",
        )
        training_repository.update_status(session, a2.training_job_id, "failed")
        # 프로젝트 B: 러닝 1
        b1 = training_repository.insert(
            session,
            project_id=p2.id,
            dataset_id=ds_b.dataset_id,
            task_type="regression",
            target_column="y",
            metric_key="rmse",
        )
        training_repository.update_status(session, b1.training_job_id, "running")

    # project_id 필터
    only_a = admin_service.list_training_history(project_id=p1.id)
    assert {r.project_id for r in only_a} == {p1.id}
    # status 필터
    failed = admin_service.list_training_history(status="failed")
    assert all(r.status == "failed" for r in failed)
    assert len(failed) == 1
    # 기간 필터: 먼 미래로 until 지정 → 0건
    empty = admin_service.list_training_history(until=datetime.utcnow() - timedelta(days=365))
    assert empty == []


def test_training_history_rejects_invalid_status(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    with pytest.raises(ValidationError):
        admin_service.list_training_history(status="bogus")


def test_training_history_rejects_invalid_limit(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    with pytest.raises(ValidationError):
        admin_service.list_training_history(limit=0)


# -------------------------------------------------------- prediction history


def test_prediction_history_joins_project_name(
    tmp_storage: Path,
    seeded_system_user: object,
    db_session: object,
) -> None:
    """실제 학습 없이 PredictionJob 단건만 시드해서 조인 경로 검증."""
    from repositories import training_repository
    from repositories.base import session_scope
    from repositories.models import Model

    project = project_service.create_project("pred-hist")

    with session_scope() as session:
        ds = _insert_dataset(session, project.id, "pred.csv")
        tj = training_repository.insert(
            session,
            project_id=project.id,
            dataset_id=ds.dataset_id,
            task_type="classification",
            target_column="y",
            metric_key="accuracy",
        )
        # Model 은 repository 에 insert 함수가 없으므로 ORM 직접 추가
        model = Model(
            training_job_id=tj.training_job_id,
            model_name="fake",
            algorithm_name="fake_algo",
            metric_score=0.9,
            is_best=True,
        )
        session.add(model)
        session.flush()
        pj = PredictionJob(
            model_id=model.model_id,
            input_type="form",
            status="completed",
        )
        session.add(pj)
        session.flush()

    rows = admin_service.list_prediction_history()
    assert len(rows) == 1
    row = rows[0]
    assert row.project_name == "pred-hist"
    assert row.algorithm_name == "fake_algo"
    assert row.input_type == "form"

    # status 필터
    fails = admin_service.list_prediction_history(status="failed")
    assert fails == []


# ---------------------------------------------------------- recent failures


def test_list_recent_failures_filters_by_action_suffix(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    """`*_failed` action 만 노출되는지 확인."""
    with session_scope() as session:
        audit_repository.write(
            session,
            action_type="project.created",
            target_type="Project",
            target_id=1,
            detail={"name": "noise"},
        )
        audit_repository.write(
            session,
            action_type="training.failed",
            target_type="TrainingJob",
            target_id=10,
            detail={"error": "boom"},
        )
        audit_repository.write(
            session,
            action_type="prediction.failed",
            target_type="PredictionJob",
            target_id=20,
            detail={"error": "bad input"},
        )

    rows = admin_service.list_recent_failures()
    types = {r.action_type for r in rows}
    assert types == {"training.failed", "prediction.failed"}
    for r in rows:
        assert r.action_type.endswith("_failed") or r.action_type.endswith(".failed")
        assert isinstance(r.detail, dict)


def test_list_recent_failures_rejects_invalid_limit(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    with pytest.raises(ValidationError):
        admin_service.list_recent_failures(limit=0)
