"""Training Service 단위/통합 테스트 (IMPLEMENTATION_PLAN §4.3).

- 분류/회귀 각각 한 번씩 전체 파이프라인을 돌려 DB/파일 산출물을 검증.
- 진행 콜백 호출 및 실패 경로 (데이터셋 없음, 타깃 컬럼 없음, 아티팩트 저장 실패)를 확인.
- 개별 알고리즘 실패가 전체 실행을 중단시키지 않는지 부분 실패 경로도 검증.
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
    training_repository,
)
from repositories.base import session_scope
from services import training_service
from utils.errors import MLTrainingError, NotFoundError, StorageError, ValidationError

# ------------------------------------------------------------------- helpers


def _seed_project_and_dataset(sample_csv: Path, storage: Path) -> tuple[int, int]:
    """샘플 CSV 를 임시 storage 로 복사하고 Project/Dataset 레코드를 직접 생성."""
    target = storage / "datasets" / f"{uuid.uuid4().hex}{sample_csv.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_csv, target)

    with session_scope() as session:
        project = project_repository.insert(session, project_name="테스트 프로젝트")
        dataset = dataset_repository.insert(
            session,
            project_id=project.project_id,
            file_name=sample_csv.name,
            file_path=str(target),
            row_count=0,
            column_count=0,
        )
        return project.project_id, dataset.dataset_id


# ----------------------------------------------------------- classification


def test_run_training_classification_happy(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)

    stages: list[tuple[str, float]] = []
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
        test_size=0.25,
    )
    result = training_service.run_training(config, on_progress=lambda s, r: stages.append((s, r)))

    assert result.metric_key == "f1"
    assert result.best_algo is not None
    assert any(r.is_best for r in result.rows)
    assert sum(1 for r in result.rows if r.is_best) == 1
    assert all(r.status in {"success", "failed"} for r in result.rows)
    success_rows = [r for r in result.rows if r.status == "success"]
    assert success_rows, "적어도 한 개 알고리즘은 성공해야 한다"
    # best row 는 success 여야 한다
    best_row = next(r for r in result.rows if r.is_best)
    assert best_row.status == "success"
    assert "f1" in best_row.metrics

    # 진행 콜백: 주요 단계가 적어도 1회씩 찍혔는지 검증
    names = [s for s, _ in stages]
    assert "preprocessing" in names
    assert "split" in names
    assert any(n.startswith("train:") for n in names)
    assert "score" in names
    assert "save" in names
    assert "completed" in names
    # 비내림차순
    ratios = [r for _, r in stages]
    assert ratios == sorted(ratios)

    # DB 검증
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.ended_at is not None
        assert (job.run_log or "").strip()

        stored_models = list(model_repository.list_by_training_job(session, result.job_id))
        assert len(stored_models) == len(result.rows)
        best_count = sum(1 for m in stored_models if m.is_best)
        assert best_count == 1

    # 파일 아티팩트 검증 (성공 모델 수만큼 디렉터리가 있어야 함)
    models_dir = tmp_storage / "models"
    assert models_dir.is_dir()
    saved_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    assert len(saved_dirs) == len(success_rows)
    for d in saved_dirs:
        assert (d / "model.joblib").exists()
        assert (d / "preprocessor.joblib").exists()
        assert (d / "feature_schema.json").exists()
        assert (d / "metrics.json").exists()

    # 감사 로그 검증
    with session_scope() as session:
        started = audit_repository.list_logs(session, action_type="training.started")
        completed = audit_repository.list_logs(session, action_type="training.completed")
        assert len(started) >= 1
        assert len(completed) >= 1


# ----------------------------------------------------------------- regression


def test_run_training_regression_happy(
    regression_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(regression_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="regression",
        target_column="progression",
        test_size=0.25,
    )
    result = training_service.run_training(config)
    assert result.metric_key == "rmse"
    assert result.best_algo is not None
    best_row = next(r for r in result.rows if r.is_best)
    assert "rmse" in best_row.metrics


# ---------------------------------------------------- validation / lookup


def test_run_training_raises_on_missing_dataset(tmp_storage: Path) -> None:
    config = TrainingConfig(
        dataset_id=9999,
        task_type="classification",
        target_column="y",
    )
    with pytest.raises(NotFoundError):
        training_service.run_training(config)


def test_run_training_raises_on_missing_target(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="does_not_exist",
    )
    with pytest.raises(ValidationError):
        training_service.run_training(config)


def test_run_training_raises_on_bad_metric_key(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
        metric_key="not_supported",
    )
    with pytest.raises(ValidationError):
        training_service.run_training(config)


def test_run_training_raises_on_missing_file(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    # 파일 제거 → StorageError 기대
    with session_scope() as session:
        ds = dataset_repository.get(session, dataset_id)
        assert ds is not None
        Path(ds.file_path).unlink()

    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    with pytest.raises(StorageError):
        training_service.run_training(config)


# ------------------------------------------------------------ partial failure


def test_run_training_partial_failure_is_recorded(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """개별 알고리즘이 실패해도 전체는 완료되고 failed 행이 DTO 에 포함된다."""
    from ml import registry
    from ml.registry import AlgoSpec

    def _boom() -> object:
        raise RuntimeError("boom")

    failing = AlgoSpec("always_fail", "classification", _boom, "f1")
    patched_specs = [*registry.get_specs("classification"), failing]
    monkeypatch.setattr(training_service, "get_specs", lambda task: patched_specs)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)
    assert any(r.status == "failed" and r.algo_name == "always_fail" for r in result.rows)
    assert any(r.status == "success" for r in result.rows)
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "completed"


def test_run_training_all_failed_marks_job_failed(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.registry import AlgoSpec

    def _boom() -> object:
        raise RuntimeError("boom")

    only_failing = [AlgoSpec("only_fail", "classification", _boom, "f1")]
    monkeypatch.setattr(training_service, "get_specs", lambda task: only_failing)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)
    assert result.best_algo is None
    assert all(r.status == "failed" for r in result.rows)
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "failed"


# ---------------------------------------------------- artifact save failure


def test_run_training_rolls_back_on_artifact_failure(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_model_bundle 실패 → DB 롤백 + 파일 정리 + MLTrainingError."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(training_service, "save_model_bundle", _raise)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    with pytest.raises(MLTrainingError):
        training_service.run_training(config)

    # DB: Model 레코드는 롤백되어 존재하지 않아야 한다
    with session_scope() as session:
        all_models = list(model_repository.list_by_project(session, project_id=1))
        assert all_models == []
        # 학습 잡은 failed 상태로 기록
        jobs = list(training_repository.list_by_project(session, project_id=1))
        assert len(jobs) == 1
        assert jobs[0].status == "failed"

    # storage/models 아래에 남은 디렉터리가 없어야 한다
    models_dir = tmp_storage / "models"
    assert not any(models_dir.iterdir())


# -------------------------------------------------------------- list/get


def test_list_and_get_training_job(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)

    jobs = training_service.list_training_jobs(project_id)
    assert len(jobs) == 1
    assert jobs[0].id == result.job_id
    assert jobs[0].status == "completed"

    reloaded = training_service.get_training_result(result.job_id)
    assert reloaded.job_id == result.job_id
    assert reloaded.best_algo == result.best_algo
    assert len(reloaded.rows) == len(result.rows)


def test_get_training_result_missing_raises(tmp_storage: Path) -> None:
    with pytest.raises(NotFoundError):
        training_service.get_training_result(99999)
