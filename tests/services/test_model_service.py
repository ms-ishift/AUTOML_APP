"""Model Service 단위/통합 테스트 (IMPLEMENTATION_PLAN §4.4).

- 학습을 실제로 한 번 돌려 DB + 디스크 아티팩트를 시드한 뒤 list/detail/save/delete 를 검증.
- delete_model 은 커밋 이후 파일까지 정리되는지도 확인.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from ml.schemas import TrainingConfig
from repositories import (
    audit_repository,
    dataset_repository,
    model_repository,
    project_repository,
)
from repositories.base import session_scope
from services import model_service, training_service
from utils.errors import NotFoundError


def _seed_project_and_dataset(sample_csv: Path, storage: Path) -> tuple[int, int]:
    target = storage / "datasets" / f"{uuid.uuid4().hex}{sample_csv.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_csv, target)
    with session_scope() as session:
        project = project_repository.insert(session, project_name="모델 테스트 프로젝트")
        dataset = dataset_repository.insert(
            session,
            project_id=project.project_id,
            file_name=sample_csv.name,
            file_path=str(target),
            row_count=0,
            column_count=0,
        )
        return project.project_id, dataset.dataset_id


@pytest.fixture()
def trained_project(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> tuple[int, int]:
    """학습을 1회 수행하고 (project_id, training_job_id) 를 반환."""
    project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    result = training_service.run_training(
        TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
    )
    return project_id, result.job_id


# -------------------------------------------------------------- list/detail


def test_list_models_returns_all_including_failed(
    trained_project: tuple[int, int],
) -> None:
    project_id, _ = trained_project
    models = model_service.list_models(project_id)
    assert len(models) >= 1
    assert all(hasattr(m, "algo_name") and hasattr(m, "is_best") for m in models)


def test_get_model_detail_restores_schema_and_metrics(
    trained_project: tuple[int, int],
) -> None:
    project_id, _ = trained_project
    models = model_service.list_models(project_id)
    success_model = next(m for m in models if m.metric_score is not None)

    detail = model_service.get_model_detail(success_model.id)
    assert detail.base.id == success_model.id
    assert detail.feature_schema.target == "species"
    assert detail.feature_schema.numeric  # iris 는 수치 피처 4개
    assert detail.metrics_summary  # 성공 모델은 최소 1개 지표
    assert all(isinstance(v, float) for v in detail.metrics_summary.values())


def test_get_model_detail_missing_raises() -> None:
    with pytest.raises(NotFoundError):
        model_service.get_model_detail(99999)


def test_get_feature_influence_after_training(trained_project: tuple[int, int]) -> None:
    """FR-094~095: 학습 직후 베스트 모델에 대해 순열 중요도가 산출된다."""
    _project_id, job_id = trained_project
    best = model_service.find_best_model(job_id)
    assert best is not None
    inf = model_service.get_feature_influence(best.id)
    assert inf.permutation_rows, "순열 중요도 행이 1개 이상이어야 한다"
    assert inf.n_rows_used >= 1
    assert inf.n_test_rows >= 1
    assert inf.scoring  # sklearn ``scoring`` 문자열
    with session_scope() as session:
        logs = audit_repository.list_logs(
            session, action_type="model.influence_computed", target_id=best.id
        )
        assert logs, "감사 로그 model.influence_computed 가 기록되어야 한다"


# ------------------------------------------------------------------- save


def test_save_model_promotes_is_best(trained_project: tuple[int, int]) -> None:
    project_id, job_id = trained_project
    models = model_service.list_models(project_id)
    candidates = [m for m in models if m.metric_score is not None]
    assert len(candidates) >= 2, "적어도 2개의 성공 모델이 있어야 pin 테스트가 의미 있음"

    non_best = next(m for m in candidates if not m.is_best)
    promoted = model_service.save_model(non_best.id)
    assert promoted.is_best is True

    # 같은 TrainingJob 내에 is_best 는 정확히 1개
    with session_scope() as session:
        job_models = list(model_repository.list_by_training_job(session, job_id))
        best_count = sum(1 for m in job_models if m.is_best)
        assert best_count == 1
        assert next(m.model_id for m in job_models if m.is_best) == non_best.id

        audits = audit_repository.list_logs(
            session, action_type="model.saved", target_id=non_best.id
        )
        assert any((a.detail_json or {}).get("manual") is True for a in audits)


def test_save_model_rejects_without_artifacts(
    trained_project: tuple[int, int],
) -> None:
    project_id, _ = trained_project
    models = model_service.list_models(project_id)
    target = next(m for m in models if m.metric_score is not None)

    # 아티팩트 디렉터리 삭제 → 저장 가능성 상실
    with session_scope() as session:
        orm = model_repository.get(session, target.id)
        assert orm is not None and orm.model_path is not None
        artifact_dir = Path(orm.model_path).parent
        shutil.rmtree(artifact_dir, ignore_errors=True)

    with pytest.raises(NotFoundError):
        model_service.save_model(target.id)


def test_save_model_missing_id_raises() -> None:
    with pytest.raises(NotFoundError):
        model_service.save_model(99999)


# ----------------------------------------------------------------- delete


def test_delete_model_removes_record_and_files(
    trained_project: tuple[int, int],
    tmp_storage: Path,
) -> None:
    project_id, _ = trained_project
    models = model_service.list_models(project_id)
    target = next(m for m in models if m.metric_score is not None)
    model_dir = tmp_storage / "models" / str(target.id)
    assert model_dir.exists()

    model_service.delete_model(target.id)

    assert not model_dir.exists()
    with session_scope() as session:
        assert model_repository.get(session, target.id) is None
        audits = audit_repository.list_logs(
            session, action_type="model.deleted", target_id=target.id
        )
        assert len(audits) == 1


def test_delete_model_missing_id_raises() -> None:
    with pytest.raises(NotFoundError):
        model_service.delete_model(99999)


# ------------------------------------------------------------ find_best_model


def test_find_best_model_after_training(trained_project: tuple[int, int]) -> None:
    _, job_id = trained_project
    best = model_service.find_best_model(job_id)
    assert best is not None
    assert best.is_best is True


def test_find_best_model_missing_job_returns_none() -> None:
    assert model_service.find_best_model(99999) is None


# ----------------------------------------------------------- plot_data


def test_get_model_plot_data_classification_has_confusion_matrix(
    trained_project: tuple[int, int],
) -> None:
    _, job_id = trained_project
    models = list(model_repository_list_success(job_id))
    assert models, "성공 모델이 하나 이상 있어야 검증 가능"
    data = model_service.get_model_plot_data(models[0].id)
    assert data is not None
    assert data["kind"] == "confusion_matrix"
    assert isinstance(data["labels"], list) and data["labels"]
    assert isinstance(data["matrix"], list) and data["matrix"]


def test_get_model_plot_data_missing_file_returns_none(
    trained_project: tuple[int, int],
    tmp_storage: Path,
) -> None:
    _, job_id = trained_project
    models = list(model_repository_list_success(job_id))
    assert models
    target = models[0]
    # plot_data.json 제거 → None 반환
    (tmp_storage / "models" / str(target.id) / "plot_data.json").unlink()
    assert model_service.get_model_plot_data(target.id) is None


def test_get_model_plot_data_unknown_model_returns_none() -> None:
    assert model_service.get_model_plot_data(99999) is None


# helper (테스트 전용) — 성공 모델 ModelDTO 리스트
def model_repository_list_success(job_id: int):
    with session_scope() as session:
        for m in model_repository.list_by_training_job(session, job_id):
            raw = m.metric_summary_json or {}
            if raw.get("status") == "success":
                yield type("X", (), {"id": m.model_id})()
