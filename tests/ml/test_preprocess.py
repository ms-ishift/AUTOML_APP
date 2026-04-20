"""ml.preprocess 단위 테스트 (IMPLEMENTATION_PLAN §3.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml.preprocess import (
    build_feature_schema,
    build_preprocessor,
    prepare_xy,
    split_feature_types,
)
from ml.schemas import TrainingConfig


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, np.nan, 40, 35],
            "income": [50.0, 60.0, 55.0, np.nan, 70.0],
            "city": ["A", "B", "A", "C", "B"],
            "membership": ["gold", "silver", "gold", "bronze", "silver"],
            "target": [0, 1, 0, 1, 1],
        }
    )


class TestSplitFeatureTypes:
    def test_numeric_vs_categorical(self, sample_df: pd.DataFrame) -> None:
        num, cat = split_feature_types(sample_df, target="target", excluded=("id",))
        assert num == ["age", "income"]
        assert cat == ["city", "membership"]

    def test_target_excluded(self, sample_df: pd.DataFrame) -> None:
        num, cat = split_feature_types(sample_df, target="age", excluded=())
        assert "age" not in num

    def test_preserves_column_order(self) -> None:
        df = pd.DataFrame({"b": [1, 2], "a": ["x", "y"], "c": [3.0, 4.0]})
        num, cat = split_feature_types(df, target="none")
        assert num == ["b", "c"]
        assert cat == ["a"]

    def test_bool_is_categorical(self) -> None:
        df = pd.DataFrame({"flag": [True, False, True], "target": [0, 1, 0]})
        num, cat = split_feature_types(df, target="target")
        assert num == []
        assert cat == ["flag"]


class TestBuildPreprocessor:
    def test_returns_column_transformer(self) -> None:
        pre = build_preprocessor(["age", "income"], ["city"])
        assert isinstance(pre, ColumnTransformer)

    def test_raises_when_all_empty(self) -> None:
        with pytest.raises(ValueError):
            build_preprocessor([], [])

    def test_fits_and_transforms_with_missing(self, sample_df: pd.DataFrame) -> None:
        pre = build_preprocessor(["age", "income"], ["city", "membership"])
        X = sample_df[["age", "income", "city", "membership"]]
        transformed = pre.fit_transform(X)
        assert transformed.shape[0] == 5
        assert not np.isnan(transformed).any()

    def test_unseen_category_does_not_error(self) -> None:
        pre = build_preprocessor([], ["city"])
        X_train = pd.DataFrame({"city": ["A", "B", "A"]})
        X_test = pd.DataFrame({"city": ["C"]})  # unseen
        pre.fit(X_train)
        out = pre.transform(X_test)
        assert out.shape[0] == 1


class TestBuildFeatureSchema:
    def test_categories_collected_and_sorted(self, sample_df: pd.DataFrame) -> None:
        schema = build_feature_schema(
            sample_df,
            num_cols=["age", "income"],
            cat_cols=["city", "membership"],
            target="target",
        )
        assert schema.numeric == ("age", "income")
        assert schema.categorical == ("city", "membership")
        assert schema.target == "target"
        assert schema.categories["city"] == ("A", "B", "C")
        assert schema.categories["membership"] == ("bronze", "gold", "silver")

    def test_input_columns_order(self, sample_df: pd.DataFrame) -> None:
        schema = build_feature_schema(
            sample_df,
            num_cols=["age"],
            cat_cols=["city"],
            target="target",
        )
        assert schema.input_columns == ("age", "city")


class TestPrepareXY:
    def test_drops_target_and_excluded(self, sample_df: pd.DataFrame) -> None:
        cfg = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="target",
            excluded_columns=("id",),
        )
        X, y = prepare_xy(sample_df, cfg)
        assert "target" not in X.columns
        assert "id" not in X.columns
        assert y.tolist() == [0, 1, 0, 1, 1]

    def test_missing_target_raises(self, sample_df: pd.DataFrame) -> None:
        cfg = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="nonexistent",
        )
        with pytest.raises(ValueError):
            prepare_xy(sample_df, cfg)

    def test_excluded_column_absence_is_ignored(self, sample_df: pd.DataFrame) -> None:
        cfg = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="target",
            excluded_columns=("not_exist",),
        )
        X, _ = prepare_xy(sample_df, cfg)
        assert set(X.columns) == {"id", "age", "income", "city", "membership"}
