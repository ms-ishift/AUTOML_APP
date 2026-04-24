"""ml/schemas.py 확장 검증 — §9.1 PreprocessingConfig / DerivedFeature / FeatureSchema (FR-055~058)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ml.schemas import (
    DerivedFeature,
    FeatureSchema,
    PreprocessingConfig,
    TrainingConfig,
    TuningConfig,
)

# --------------------------------------------------------------------- #1~3
# PreprocessingConfig 기본값 / is_default / summary


def test_preprocessing_config_defaults_are_mvp_equivalent() -> None:
    cfg = PreprocessingConfig()
    assert cfg.numeric_impute == "median"
    assert cfg.numeric_scale == "standard"
    assert cfg.outlier == "none"
    assert cfg.categorical_impute == "most_frequent"
    assert cfg.categorical_encoding == "onehot"
    assert cfg.imbalance == "none"
    assert cfg.datetime_decompose is False
    assert cfg.datetime_parts == ()
    assert cfg.bool_as_numeric is True
    assert cfg.highcard_auto_downgrade is True


def test_preprocessing_config_is_default_flag() -> None:
    assert PreprocessingConfig().is_default is True
    # 단일 축만 바뀌어도 False
    assert PreprocessingConfig(numeric_scale="robust").is_default is False
    assert PreprocessingConfig(imbalance="class_weight").is_default is False


def test_preprocessing_config_summary() -> None:
    assert PreprocessingConfig().summary() == "default"
    cfg = PreprocessingConfig(numeric_scale="robust", imbalance="class_weight")
    s = cfg.summary()
    assert "numeric_scale=robust" in s
    assert "imbalance=class_weight" in s
    # 변경 안 한 축은 요약에 안 나옴
    assert "numeric_impute" not in s


# --------------------------------------------------------------------- #4~8
# __post_init__ 값 검증


def test_outlier_iqr_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="outlier_iqr_k"):
        PreprocessingConfig(outlier_iqr_k=0.0)
    with pytest.raises(ValueError, match="outlier_iqr_k"):
        PreprocessingConfig(outlier_iqr_k=-1.0)


def test_winsorize_p_range() -> None:
    with pytest.raises(ValueError, match="winsorize_p"):
        PreprocessingConfig(winsorize_p=0.0)
    with pytest.raises(ValueError, match="winsorize_p"):
        PreprocessingConfig(winsorize_p=0.5)
    with pytest.raises(ValueError, match="winsorize_p"):
        PreprocessingConfig(winsorize_p=0.9)


def test_smote_k_neighbors_positive() -> None:
    with pytest.raises(ValueError, match="smote_k_neighbors"):
        PreprocessingConfig(smote_k_neighbors=0)
    with pytest.raises(ValueError, match="smote_k_neighbors"):
        PreprocessingConfig(smote_k_neighbors=-3)


def test_highcard_threshold_min() -> None:
    with pytest.raises(ValueError, match="highcard_threshold"):
        PreprocessingConfig(highcard_threshold=1)
    with pytest.raises(ValueError, match="highcard_threshold"):
        PreprocessingConfig(highcard_threshold=0)


def test_datetime_decompose_requires_parts() -> None:
    with pytest.raises(ValueError, match="datetime_parts"):
        PreprocessingConfig(datetime_decompose=True, datetime_parts=())
    # 반대 방향 (parts 있고 decompose=False) 은 허용
    cfg = PreprocessingConfig(datetime_decompose=False, datetime_parts=("year",))
    assert cfg.datetime_parts == ("year",)


# --------------------------------------------------------------------- #9~11
# to_dict / from_dict 왕복


def test_preprocessing_config_to_from_dict_roundtrip_default() -> None:
    original = PreprocessingConfig()
    recovered = PreprocessingConfig.from_dict(original.to_dict())
    assert recovered == original
    assert recovered.is_default is True


def test_preprocessing_config_to_from_dict_roundtrip_custom() -> None:
    original = PreprocessingConfig(
        numeric_impute="mean",
        numeric_scale="robust",
        outlier="winsorize",
        outlier_iqr_k=2.0,
        winsorize_p=0.05,
        categorical_impute="constant_missing",
        categorical_encoding="frequency",
        highcard_threshold=30,
        highcard_auto_downgrade=False,
        datetime_decompose=True,
        datetime_parts=("year", "month", "weekday"),
        bool_as_numeric=False,
        imbalance="smote",
        smote_k_neighbors=7,
    )
    data = original.to_dict()
    # datetime_parts 는 list 로 직렬화되어야 한다 (JSON 호환)
    assert isinstance(data["datetime_parts"], list)
    recovered = PreprocessingConfig.from_dict(data)
    assert recovered == original
    # 복원 시 tuple 로 재캐스팅
    assert isinstance(recovered.datetime_parts, tuple)


def test_preprocessing_config_from_dict_with_missing_keys() -> None:
    # 완전히 빈 dict 라도 전부 기본값으로 복원
    recovered = PreprocessingConfig.from_dict({})
    assert recovered == PreprocessingConfig()
    # 일부 키만 있을 때 나머지도 기본값
    partial = PreprocessingConfig.from_dict({"numeric_scale": "robust"})
    assert partial.numeric_scale == "robust"
    assert partial.numeric_impute == "median"  # default
    assert partial.imbalance == "none"  # default


# --------------------------------------------------------------------- #12~14
# TrainingConfig 통합


def test_training_config_accepts_preprocessing_kwarg() -> None:
    cfg = TrainingConfig(
        dataset_id=1,
        task_type="classification",
        target_column="y",
        preprocessing=PreprocessingConfig(imbalance="class_weight"),
    )
    assert cfg.preprocessing is not None
    assert cfg.preprocessing.imbalance == "class_weight"


def test_training_config_none_preprocessing_is_allowed() -> None:
    # 명시 생략
    cfg1 = TrainingConfig(dataset_id=1, task_type="classification", target_column="y")
    assert cfg1.preprocessing is None
    # 명시적으로 None
    cfg2 = TrainingConfig(
        dataset_id=1,
        task_type="regression",
        target_column="y",
        preprocessing=None,
    )
    assert cfg2.preprocessing is None


def test_training_config_regression_with_smote_rejected() -> None:
    with pytest.raises(ValueError, match="SMOTE"):
        TrainingConfig(
            dataset_id=1,
            task_type="regression",
            target_column="y",
            preprocessing=PreprocessingConfig(imbalance="smote"),
        )
    # classification + SMOTE 는 OK
    ok = TrainingConfig(
        dataset_id=1,
        task_type="classification",
        target_column="y",
        preprocessing=PreprocessingConfig(imbalance="smote"),
    )
    assert ok.preprocessing is not None
    assert ok.preprocessing.imbalance == "smote"


# --------------------------------------------------------------------- #15~17
# FeatureSchema 확장 + DerivedFeature


def test_feature_schema_roundtrip_with_datetime_and_derived() -> None:
    schema = FeatureSchema(
        numeric=("age", "price"),
        categorical=("city",),
        target="y",
        categories={"city": ("seoul", "busan")},
        datetime=("signup_at",),
        derived=(
            DerivedFeature(name="signup_at_year", source="signup_at", kind="datetime_year"),
            DerivedFeature(name="signup_at_month", source="signup_at", kind="datetime_month"),
        ),
    )
    recovered = FeatureSchema.from_dict(schema.to_dict())
    assert recovered == schema
    assert recovered.datetime == ("signup_at",)
    assert len(recovered.derived) == 2
    assert recovered.derived[0].kind == "datetime_year"


def test_feature_schema_from_legacy_dict_missing_new_keys() -> None:
    # 구 아티팩트 (datetime/derived 키 없음) — to_dict 구버전 형태를 직접 구성
    legacy = {
        "numeric": ["age"],
        "categorical": ["city"],
        "target": "y",
        "categories": {"city": ["seoul", "busan"]},
    }
    schema = FeatureSchema.from_dict(legacy)
    assert schema.datetime == ()
    assert schema.derived == ()
    assert schema.numeric == ("age",)
    assert schema.categorical == ("city",)
    # input_columns 는 원본만 (datetime 제외)
    assert schema.input_columns == ("age", "city")


def test_derived_feature_frozen() -> None:
    d = DerivedFeature(name="x_year", source="x", kind="datetime_year")
    with pytest.raises(FrozenInstanceError):
        d.name = "other"  # type: ignore[misc]


# ---------------------------------------------------- §10.3 TrainingConfig 확장


def test_training_config_algorithms_default_is_none() -> None:
    """기본값 None → 필터 미적용 (v0.2.0 동치)."""
    cfg = TrainingConfig(dataset_id=1, task_type="classification", target_column="y")
    assert cfg.algorithms is None
    assert cfg.tuning is None


def test_training_config_algorithms_empty_tuple_rejected() -> None:
    with pytest.raises(ValueError, match="최소 1개"):
        TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="y",
            algorithms=(),
        )


def test_training_config_algorithms_duplicates_rejected() -> None:
    with pytest.raises(ValueError, match="중복"):
        TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="y",
            algorithms=("random_forest", "random_forest"),
        )


def test_training_config_algorithms_roundtrip() -> None:
    cfg = TrainingConfig(
        dataset_id=1,
        task_type="classification",
        target_column="y",
        algorithms=("random_forest", "logistic_regression"),
    )
    assert cfg.algorithms == ("random_forest", "logistic_regression")


# ------------------------------------------------------------- §10.3 TuningConfig


def test_tuning_config_defaults() -> None:
    t = TuningConfig()
    assert t.method == "none"
    assert t.cv_folds == 3
    assert t.max_iter is None
    assert t.timeout_sec is None


def test_tuning_config_accepts_grid_method() -> None:
    t = TuningConfig(method="grid", cv_folds=5)
    assert t.method == "grid"
    assert t.cv_folds == 5


def test_tuning_config_rejects_invalid_cv_folds() -> None:
    with pytest.raises(ValueError, match="cv_folds"):
        TuningConfig(cv_folds=1)


def test_tuning_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_sec"):
        TuningConfig(timeout_sec=0)


def test_training_config_with_tuning_slot() -> None:
    """§10.3: tuning 필드는 스키마만 — method='grid' 라도 TrainingConfig 는 수용."""
    cfg = TrainingConfig(
        dataset_id=1,
        task_type="classification",
        target_column="y",
        tuning=TuningConfig(method="grid"),
    )
    assert cfg.tuning is not None
    assert cfg.tuning.method == "grid"
