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
