from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from repositories import dataset_repository, project_repository, training_repository
from services.dto import (
    ColumnProfileDTO,
    DatasetDTO,
    DatasetProfileDTO,
    FeatureSchemaDTO,
    ModelComparisonRowDTO,
    ModelDetailDTO,
    ModelDTO,
    PredictionResultDTO,
    ProjectDTO,
    TrainingJobDTO,
    TrainingResultDTO,
)


def test_project_dto_from_orm(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p-dto", description="desc")
    db_session.commit()

    dto = ProjectDTO.from_orm(p, dataset_count=3, model_count=7)
    assert dto.id == p.project_id
    assert dto.name == "p-dto"
    assert dto.description == "desc"
    assert dto.dataset_count == 3
    assert dto.model_count == 7
    assert isinstance(dto.created_at, datetime)


def test_dataset_dto_from_orm(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p")
    ds = dataset_repository.insert(
        db_session,
        project_id=p.project_id,
        file_name="x.csv",
        file_path="/tmp/x.csv",
        row_count=10,
        column_count=3,
    )
    db_session.commit()

    dto = DatasetDTO.from_orm(ds)
    assert dto.id == ds.dataset_id
    assert dto.project_id == p.project_id
    assert dto.row_count == 10
    assert dto.column_count == 3


def test_training_job_dto_from_orm(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p")
    ds = dataset_repository.insert(
        db_session,
        project_id=p.project_id,
        file_name="x.csv",
        file_path="/tmp/x.csv",
        row_count=1,
        column_count=1,
    )
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()

    dto = TrainingJobDTO.from_orm(job)
    assert dto.id == job.training_job_id
    assert dto.task_type == "classification"
    assert dto.status == "pending"


def test_dtos_are_frozen() -> None:
    dto = ProjectDTO(
        id=1,
        name="x",
        description=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )
    with pytest.raises(FrozenInstanceError):
        dto.name = "other"  # type: ignore[misc]


def test_training_result_dto_structure() -> None:
    row = ModelComparisonRowDTO(
        algo_name="random_forest",
        status="success",
        metrics={"f1": 0.9},
        train_time_ms=120,
        is_best=True,
    )
    result = TrainingResultDTO(
        job_id=42,
        rows=[row],
        best_algo="random_forest",
        metric_key="f1",
    )
    assert result.best_algo == "random_forest"
    assert result.rows[0].is_best


def test_model_detail_dto_composition() -> None:
    base = ModelDTO(
        id=1,
        training_job_id=1,
        algo_name="rf",
        metric_score=0.9,
        is_best=True,
        created_at=datetime(2026, 1, 1),
    )
    schema = FeatureSchemaDTO(
        numeric=["a"],
        categorical=["b"],
        target="y",
        categories={"b": ["x", "y"]},
    )
    detail = ModelDetailDTO(base=base, feature_schema=schema, metrics_summary={"f1": 0.9})
    assert detail.base.algo_name == "rf"
    assert detail.feature_schema.target == "y"


def test_dataset_profile_dto_uses_column_profile_dto() -> None:
    col = ColumnProfileDTO(
        name="a",
        dtype="int64",
        n_missing=0,
        n_unique=5,
        missing_ratio=0.0,
        unique_ratio=0.5,
    )
    prof = DatasetProfileDTO(rows=10, cols=1, columns=[col])
    assert prof.columns[0].name == "a"
    assert prof.rows == 10


def test_prediction_result_dto_defaults() -> None:
    dto = PredictionResultDTO(job_id=1, rows=[{"y": 1}])
    assert dto.result_path is None
    assert dto.warnings == []


# ---------------------------------------------------------- §9.7 Preprocessing


class TestPreprocessingConfigDTO:
    """§9.7: PreprocessingConfig ↔ PreprocessingConfigDTO 양방향 변환."""

    def test_from_config_defaults(self) -> None:
        from ml.schemas import PreprocessingConfig
        from services.dto import PreprocessingConfigDTO

        dto = PreprocessingConfigDTO.from_config(PreprocessingConfig())
        assert dto.numeric_impute == "median"
        assert dto.numeric_scale == "standard"
        assert dto.outlier == "none"
        assert dto.imbalance == "none"
        assert dto.datetime_parts == []

    def test_roundtrip_preserves_all_fields(self) -> None:
        from ml.schemas import PreprocessingConfig
        from services.dto import PreprocessingConfigDTO

        original = PreprocessingConfig(
            numeric_impute="median",
            numeric_scale="robust",
            outlier="iqr",
            outlier_iqr_k=2.0,
            winsorize_p=0.05,
            categorical_impute="constant",
            categorical_encoding="ordinal",
            highcard_threshold=30,
            highcard_auto_downgrade=False,
            datetime_decompose=True,
            datetime_parts=("year", "month"),
            bool_as_numeric=False,
            imbalance="class_weight",
            smote_k_neighbors=3,
        )
        dto = PreprocessingConfigDTO.from_config(original)
        restored = dto.to_config()
        assert restored == original

    def test_dto_is_frozen(self) -> None:
        from services.dto import PreprocessingConfigDTO

        dto = PreprocessingConfigDTO()
        with pytest.raises(FrozenInstanceError):
            dto.numeric_scale = "robust"  # type: ignore[misc]

    def test_to_config_raises_on_invalid_combination(self) -> None:
        """DTO 는 자유롭게 구성 가능하지만 to_config 단계에서 ml 레이어 검증이 적용된다."""
        from services.dto import PreprocessingConfigDTO

        bad = PreprocessingConfigDTO(datetime_decompose=True, datetime_parts=[])
        with pytest.raises(ValueError, match="datetime_parts"):
            bad.to_config()


class TestFeaturePreviewDTO:
    """§9.7: FeaturePreviewDTO 기본 구조 / 기본값 검증."""

    def test_construct_with_minimal_fields(self) -> None:
        from services.dto import FeaturePreviewDTO

        preview = FeaturePreviewDTO(n_cols_in=3, n_cols_out=5)
        assert preview.n_cols_in == 3
        assert preview.n_cols_out == 5
        assert preview.derived == ()
        assert preview.encoding_summary == {}
        assert preview.auto_downgraded == ()

    def test_frozen_slots(self) -> None:
        from services.dto import FeaturePreviewDTO

        preview = FeaturePreviewDTO(n_cols_in=1, n_cols_out=1)
        with pytest.raises(FrozenInstanceError):
            preview.n_cols_in = 99  # type: ignore[misc]
