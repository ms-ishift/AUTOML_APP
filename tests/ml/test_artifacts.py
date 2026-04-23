"""ml/artifacts.py 단위 테스트 (IMPLEMENTATION_PLAN §3.7 / §9.6)."""

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
    PREPROCESSING_FILENAME,
    PREPROCESSOR_FILENAME,
    SCHEMA_FILENAME,
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
    validate_prediction_input,
)
from ml.preprocess import build_feature_schema, build_preprocessor, split_feature_types
from ml.schemas import FeatureSchema, PreprocessingConfig


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


# ---------------------------------------------------------------- §9.6 bundle


class TestPreprocessingConfigPersistence:
    """§9.6: PreprocessingConfig 의 번들 저장/로드 왕복 및 하위호환."""

    def test_model_bundle_default_preprocessing(self) -> None:
        """ModelBundle.preprocessing 은 기본값이 제공되어 기존 인스턴스 생성자가 호환된다."""
        schema = FeatureSchema(numeric=("x",), categorical=(), target="y")
        bundle = ModelBundle(
            estimator=object(),
            preprocessor=object(),
            schema=schema,
            metrics={},
        )
        assert isinstance(bundle.preprocessing, PreprocessingConfig)
        assert bundle.preprocessing.is_default

    def test_save_without_preprocessing_config_skips_file(self, tmp_path: Path) -> None:
        """preprocessing_config 인자 생략 시 preprocessing_config.json 이 생성되지 않는다 (구 모델 동치)."""
        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "no_cfg"

        paths = save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
        )

        assert "preprocessing" not in paths
        assert not (target_dir / PREPROCESSING_FILENAME).exists()

    def test_save_with_preprocessing_config_writes_json(self, tmp_path: Path) -> None:
        """preprocessing_config 제공 시 JSON 파일 생성 및 경로 반환."""
        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "with_cfg"
        cfg = PreprocessingConfig(
            numeric_impute="median",
            numeric_scale="standard",
            imbalance="class_weight",
        )

        paths = save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
            preprocessing_config=cfg,
        )

        pp_path = target_dir / PREPROCESSING_FILENAME
        assert pp_path.exists()
        assert paths["preprocessing"] == pp_path
        data = json.loads(pp_path.read_text("utf-8"))
        assert data["imbalance"] == "class_weight"

    def test_load_roundtrips_preprocessing_config(self, tmp_path: Path) -> None:
        """save → load 왕복 시 PreprocessingConfig 가 동일한 값으로 복원되어야 한다."""
        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "rt"
        cfg = PreprocessingConfig(
            numeric_impute="median",
            outlier="iqr",
            outlier_iqr_k=2.0,
            numeric_scale="robust",
            bool_as_numeric=False,
            imbalance="smote",
            smote_k_neighbors=3,
        )
        save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
            preprocessing_config=cfg,
        )

        bundle = load_model_bundle(target_dir)
        assert bundle.preprocessing == cfg
        # Smoke check: estimator 는 여전히 예측 가능해야 한다
        df = pd.DataFrame({"num_a": [0.1], "num_b": [0.2], "cat_a": ["x"]})
        preds = bundle.estimator.predict(df)
        assert len(preds) == 1

    def test_load_legacy_bundle_falls_back_to_default_config(self, tmp_path: Path) -> None:
        """preprocessing_config.json 이 없는 구 번들은 PreprocessingConfig() 기본값으로 복원 (§9.6 하위호환)."""
        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "legacy"
        save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
        )
        assert not (target_dir / PREPROCESSING_FILENAME).exists()

        bundle = load_model_bundle(target_dir)
        assert bundle.preprocessing == PreprocessingConfig()
        assert bundle.preprocessing.is_default

    def test_load_missing_preprocessing_file_does_not_break_required_check(
        self, tmp_path: Path
    ) -> None:
        """preprocessing_config.json 은 _REQUIRED_FILES 에 포함되지 않으므로 부재만으로는 FileNotFoundError 발생하지 않는다."""
        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "no_pp"
        save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
        )
        load_model_bundle(target_dir)


# ---------------------------------------------------------- §9.8 하위호환 로그


class TestLegacyPreprocessingLoggedOnce:
    """§9.8: 구 번들 로드 시 model.legacy_preprocessing_loaded 가 정확히 1회 emit."""

    def test_legacy_bundle_emits_log_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ml import artifacts as artifacts_mod
        from utils.events import Event

        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "legacy_logged"
        save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
        )

        events: list[tuple[str, dict[str, object]]] = []

        def _spy(_logger: object, event: str, **extra: object) -> None:
            events.append((event, dict(extra)))

        monkeypatch.setattr(artifacts_mod, "log_event", _spy)

        load_model_bundle(target_dir)

        legacy_events = [e for e in events if e[0] == Event.MODEL_LEGACY_PREPROCESSING_LOADED]
        assert len(legacy_events) == 1
        assert legacy_events[0][1].get("model_dir") == str(target_dir)

    def test_modern_bundle_does_not_emit_legacy_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """preprocessing_config.json 이 존재하면 legacy 이벤트는 emit 되지 않는다."""
        from ml import artifacts as artifacts_mod
        from utils.events import Event

        estimator, preprocessor, schema, metrics = _fit_bundle()
        target_dir = tmp_path / "modern_no_log"
        save_model_bundle(
            target_dir,
            estimator=estimator,
            preprocessor=preprocessor,
            schema=schema,
            metrics=metrics,
            preprocessing_config=PreprocessingConfig(numeric_scale="standard"),
        )

        events: list[str] = []
        monkeypatch.setattr(
            artifacts_mod,
            "log_event",
            lambda _logger, event, **_extra: events.append(event),
        )

        load_model_bundle(target_dir)
        assert Event.MODEL_LEGACY_PREPROCESSING_LOADED not in events
