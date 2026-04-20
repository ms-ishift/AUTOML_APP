"""ml/artifacts.py 단위 테스트 (IMPLEMENTATION_PLAN §3.7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.artifacts import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    SCHEMA_FILENAME,
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
    validate_prediction_input,
)
from ml.preprocess import build_feature_schema, build_preprocessor, split_feature_types
from ml.schemas import FeatureSchema


def _fit_bundle() -> tuple[object, object, FeatureSchema, dict[str, float]]:
    """테스트용 간단 분류 모델: iris-like 2 feature 합성 데이터."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num_a": rng.normal(size=60),
            "num_b": rng.normal(size=60),
            "cat_a": rng.choice(["x", "y", "z"], size=60),
            "target": rng.choice([0, 1], size=60),
        }
    )

    num_cols, cat_cols = split_feature_types(df, target="target")
    schema = build_feature_schema(df, num_cols, cat_cols, target="target")
    pre = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=["target"])
    y = df["target"]

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", LogisticRegression(max_iter=200))])
    pipeline.fit(X, y)
    fitted_pre = pipeline.named_steps["preprocessor"]
    metrics = {"accuracy": 0.9, "f1": 0.88, "metric_key": "f1"}
    return pipeline, fitted_pre, schema, metrics


def test_save_creates_four_required_files(tmp_path: Path) -> None:
    estimator, preprocessor, schema, metrics = _fit_bundle()
    target_dir = tmp_path / "1"

    paths = save_model_bundle(
        target_dir,
        estimator=estimator,
        preprocessor=preprocessor,
        schema=schema,
        metrics=metrics,
    )

    assert target_dir.is_dir()
    for name in (MODEL_FILENAME, PREPROCESSOR_FILENAME, SCHEMA_FILENAME, METRICS_FILENAME):
        assert (target_dir / name).exists()
    assert paths["model"].name == MODEL_FILENAME
    assert paths["schema"].name == SCHEMA_FILENAME


def test_load_roundtrip_restores_bundle(tmp_path: Path) -> None:
    estimator, preprocessor, schema, metrics = _fit_bundle()
    target_dir = tmp_path / "2"
    save_model_bundle(
        target_dir,
        estimator=estimator,
        preprocessor=preprocessor,
        schema=schema,
        metrics=metrics,
    )

    bundle = load_model_bundle(target_dir)
    assert isinstance(bundle, ModelBundle)
    assert bundle.schema.target == schema.target
    assert bundle.schema.numeric == schema.numeric
    assert bundle.schema.categorical == schema.categorical
    assert bundle.metrics == metrics
    # 재로드된 estimator 로 예측이 가능해야 한다
    df = pd.DataFrame(
        {
            "num_a": [0.1, 0.2],
            "num_b": [-0.3, 0.4],
            "cat_a": ["x", "y"],
        }
    )
    preds = bundle.estimator.predict(df)
    assert len(preds) == 2


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "does-not-exist")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    estimator, preprocessor, schema, metrics = _fit_bundle()
    target_dir = tmp_path / "3"
    save_model_bundle(
        target_dir,
        estimator=estimator,
        preprocessor=preprocessor,
        schema=schema,
        metrics=metrics,
    )
    (target_dir / SCHEMA_FILENAME).unlink()

    with pytest.raises(FileNotFoundError):
        load_model_bundle(target_dir)


def test_schema_file_is_valid_json(tmp_path: Path) -> None:
    estimator, preprocessor, schema, metrics = _fit_bundle()
    target_dir = tmp_path / "4"
    save_model_bundle(
        target_dir,
        estimator=estimator,
        preprocessor=preprocessor,
        schema=schema,
        metrics=metrics,
    )

    data = json.loads((target_dir / SCHEMA_FILENAME).read_text("utf-8"))
    assert data["target"] == schema.target
    assert sorted(data["numeric"]) == sorted(schema.numeric)


# --------------------------------------------------- validate_prediction_input


@pytest.fixture()
def simple_schema() -> FeatureSchema:
    return FeatureSchema(
        numeric=("x", "y"),
        categorical=("g",),
        target="t",
        categories={"g": ("A", "B", "C")},
    )


def test_validate_returns_expected_columns_only(simple_schema: FeatureSchema) -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "g": ["A", "B"],
            "extra": [9, 9],
        }
    )
    cleaned = validate_prediction_input(df, simple_schema)
    assert list(cleaned.columns) == ["x", "y", "g"]
    assert cleaned["g"].dtype == object


def test_validate_missing_column_raises(simple_schema: FeatureSchema) -> None:
    df = pd.DataFrame({"x": [1], "g": ["A"]})
    with pytest.raises(ValueError, match="필수 입력 컬럼 누락"):
        validate_prediction_input(df, simple_schema)


def test_validate_numeric_coerces_strings(simple_schema: FeatureSchema) -> None:
    df = pd.DataFrame(
        {
            "x": ["1.5", "not-a-number"],
            "y": ["3", "4"],
            "g": ["A", "A"],
        }
    )
    cleaned = validate_prediction_input(df, simple_schema)
    assert pd.api.types.is_numeric_dtype(cleaned["x"])
    assert pd.isna(cleaned.loc[1, "x"])


def test_validate_categorical_casts_to_string(simple_schema: FeatureSchema) -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "g": [1, 2],
        }
    )
    cleaned = validate_prediction_input(df, simple_schema)
    assert cleaned["g"].iloc[0] == "1"
    assert cleaned["g"].iloc[1] == "2"


def test_validate_rejects_empty(simple_schema: FeatureSchema) -> None:
    with pytest.raises(ValueError):
        validate_prediction_input(pd.DataFrame(), simple_schema)
