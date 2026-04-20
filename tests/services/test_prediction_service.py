"""Prediction Service 단위/통합 테스트 (IMPLEMENTATION_PLAN §4.5).

- 학습을 1회 실제로 돌려 아티팩트를 시드하고, 그 위에서 단건/배치 예측을 검증.
- §10.4 규칙: 누락 컬럼은 PredictionInputError, 추가 컬럼은 경고 후 무시, unseen 범주는 경고.
- PredictionJob 레코드(성공/실패) 와 감사 로그 기록을 함께 검증.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from ml.schemas import TrainingConfig
from repositories import (
    audit_repository,
    dataset_repository,
    model_repository,
    prediction_repository,
    project_repository,
)
from repositories.base import session_scope
from services import prediction_service, training_service
from services.prediction_service import (
    BATCH_PREVIEW_MAX_ROWS,
    PREDICTION_COLUMN,
    PROBABILITY_PREFIX,
)
from utils.errors import (
    NotFoundError,
    PredictionInputError,
    StorageError,
    ValidationError,
)


def _seed_project_and_dataset(sample_csv: Path, storage: Path) -> tuple[int, int]:
    target = storage / "datasets" / f"{uuid.uuid4().hex}{sample_csv.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_csv, target)
    with session_scope() as session:
        project = project_repository.insert(session, project_name="예측 테스트 프로젝트")
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
def classification_model(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> tuple[int, Path]:
    """학습 1회 수행 후 (best_model_id, 원본 CSV 경로) 반환."""
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    result = training_service.run_training(
        TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
    )
    assert result.best_algo is not None
    with session_scope() as session:
        job_models = list(model_repository.list_by_training_job(session, result.job_id))
        best = next(m for m in job_models if m.is_best)
        return best.model_id, classification_csv


@pytest.fixture()
def regression_model(
    regression_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> tuple[int, Path]:
    _, dataset_id = _seed_project_and_dataset(regression_csv, tmp_storage)
    result = training_service.run_training(
        TrainingConfig(
            dataset_id=dataset_id,
            task_type="regression",
            target_column="progression",
        )
    )
    with session_scope() as session:
        job_models = list(model_repository.list_by_training_job(session, result.job_id))
        best = next(m for m in job_models if m.is_best)
        return best.model_id, regression_csv


# ----------------------------------------------------------- predict_single


def test_predict_single_classification_returns_prediction_and_proba(
    classification_model: tuple[int, Path],
) -> None:
    model_id, csv_path = classification_model
    df = pd.read_csv(csv_path)
    payload = df.iloc[0].drop("species").to_dict()

    result = prediction_service.predict_single(model_id, payload)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert PREDICTION_COLUMN in row
    # iris 3 클래스 → 확률 컬럼 3개
    prob_keys = [k for k in row if k.startswith(PROBABILITY_PREFIX)]
    assert len(prob_keys) == 3
    # 확률 합은 대략 1
    assert abs(sum(row[k] for k in prob_keys) - 1.0) < 1e-6
    assert result.result_path is None

    # PredictionJob 기록 확인
    with session_scope() as session:
        pj = prediction_repository.get(session, result.job_id)
        assert pj is not None
        assert pj.status == "completed"
        assert pj.input_type == "form"


def test_predict_single_regression(regression_model: tuple[int, Path]) -> None:
    model_id, csv_path = regression_model
    df = pd.read_csv(csv_path)
    payload = df.iloc[0].drop("progression").to_dict()

    result = prediction_service.predict_single(model_id, payload)
    row = result.rows[0]
    assert PREDICTION_COLUMN in row
    assert all(not k.startswith(PROBABILITY_PREFIX) for k in row)
    assert isinstance(row[PREDICTION_COLUMN], (int, float))


def test_predict_single_missing_column_raises_and_marks_failed(
    classification_model: tuple[int, Path],
) -> None:
    model_id, _ = classification_model
    payload = {"sepal length (cm)": 5.0, "sepal width (cm)": 3.4}  # 2개 누락

    with pytest.raises(PredictionInputError) as excinfo:
        prediction_service.predict_single(model_id, payload)
    assert "누락" in str(excinfo.value) or "missing" in str(excinfo.value).lower()

    # 실패 감사 로그는 존재, 이 시점 PredictionJob 은 아직 insert 전이므로 레코드는 0~1 건.
    with session_scope() as session:
        fails = audit_repository.list_logs(session, action_type="prediction.failed")
        assert len(fails) >= 1


def test_predict_single_rejects_empty_payload(
    classification_model: tuple[int, Path],
) -> None:
    model_id, _ = classification_model
    with pytest.raises(PredictionInputError):
        prediction_service.predict_single(model_id, {})


def test_predict_single_unknown_model_raises() -> None:
    with pytest.raises(NotFoundError):
        prediction_service.predict_single(99999, {"x": 1})


# ------------------------------------------------------------- predict_batch


def test_predict_batch_saves_csv_and_returns_rows(
    classification_model: tuple[int, Path],
    tmp_storage: Path,
    tmp_path: Path,
) -> None:
    model_id, csv_path = classification_model
    df = pd.read_csv(csv_path)
    input_path = tmp_path / "batch.csv"
    df.drop(columns=["species"]).to_csv(input_path, index=False)

    result = prediction_service.predict_batch(model_id, input_path)
    assert result.result_path is not None
    result_file = Path(result.result_path)
    assert result_file.exists()
    assert result_file.parent == tmp_storage / "predictions"
    assert result_file.name == f"{result.job_id}.csv"

    # 저장된 CSV 검증: prediction 컬럼 포함 + 행 수 일치
    stored = pd.read_csv(result_file)
    assert PREDICTION_COLUMN in stored.columns
    assert len(stored) == len(df)
    # rows 는 BATCH_PREVIEW_MAX_ROWS 범위로 클램프
    assert len(result.rows) == min(len(df), BATCH_PREVIEW_MAX_ROWS)

    # PredictionJob 기록
    with session_scope() as session:
        pj = prediction_repository.get(session, result.job_id)
        assert pj is not None
        assert pj.status == "completed"
        assert pj.input_type == "file"
        assert pj.result_path == str(result_file)


def test_predict_batch_warns_on_extra_and_unseen(
    classification_model: tuple[int, Path],
    tmp_path: Path,
) -> None:
    model_id, csv_path = classification_model
    df = pd.read_csv(csv_path).drop(columns=["species"])
    df["extra_col"] = "ignore_me"
    input_path = tmp_path / "batch_extra.csv"
    df.to_csv(input_path, index=False)

    result = prediction_service.predict_batch(model_id, input_path)
    joined = " | ".join(result.warnings)
    assert "extra_col" in joined
    assert result.result_path is not None


def test_predict_batch_missing_column_raises(
    classification_model: tuple[int, Path],
    tmp_path: Path,
) -> None:
    model_id, csv_path = classification_model
    df = pd.read_csv(csv_path).drop(columns=["species", "petal width (cm)"])
    input_path = tmp_path / "batch_missing.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(PredictionInputError):
        prediction_service.predict_batch(model_id, input_path)


def test_predict_batch_missing_file_raises(
    classification_model: tuple[int, Path],
    tmp_path: Path,
) -> None:
    model_id, _ = classification_model
    with pytest.raises(ValidationError):
        prediction_service.predict_batch(model_id, tmp_path / "no_such.csv")


def test_predict_batch_rolls_back_on_save_failure(
    classification_model: tuple[int, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id, csv_path = classification_model
    df = pd.read_csv(csv_path)
    input_path = tmp_path / "batch.csv"
    df.drop(columns=["species"]).to_csv(input_path, index=False)

    original = pd.DataFrame.to_csv

    def _failing_to_csv(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        # settings.predictions_dir 하위 경로에만 실패 시키기 위해 path 기반 판단
        if args and isinstance(args[0], (str, Path)) and "predictions" in str(args[0]):
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _failing_to_csv)

    with pytest.raises(StorageError):
        prediction_service.predict_batch(model_id, input_path)

    with session_scope() as session:
        jobs = list(prediction_repository.list_by_model(session, model_id))
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        fails = audit_repository.list_logs(session, action_type="prediction.failed")
        assert len(fails) >= 1


# -------------------------------------------------------- model without path


def test_prediction_requires_saved_artifacts(
    classification_model: tuple[int, Path],
) -> None:
    model_id, _ = classification_model
    with session_scope() as session:
        orm = model_repository.get(session, model_id)
        assert orm is not None
        # 경로 제거하여 "아티팩트 없음" 상태 시뮬레이션
        orm.model_path = None

    with pytest.raises(NotFoundError):
        prediction_service.predict_single(model_id, {"x": 1})
