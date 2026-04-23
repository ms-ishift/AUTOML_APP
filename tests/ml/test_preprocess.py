"""ml.preprocess 단위 테스트 (IMPLEMENTATION_PLAN §3.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml.preprocess import (
    FrequencyEncoder,
    IQRClipper,
    PreprocessingRouteReport,
    Winsorizer,
    build_feature_schema,
    build_preprocessor,
    plan_categorical_routing,
    prepare_xy,
    split_feature_types,
    split_feature_types_v2,
)
from ml.schemas import PreprocessingConfig, TrainingConfig


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


# ========================================================== §9.3 확장 테스트


class TestSplitFeatureTypesV2:
    def test_four_way_split_with_bool_and_datetime(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 20, 30],
                "city": ["a", "b", "c"],
                "flag": [True, False, True],
                "signup": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [0, 1, 0],
            }
        )
        num, cat, dt, bl = split_feature_types_v2(df, target="target")
        assert num == ["age"]
        assert cat == ["city"]
        assert dt == ["signup"]
        assert bl == ["flag"]

    def test_excluded_dropped_from_all_groups(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age": [10, 20, 30],
                "flag": [True, False, True],
                "target": [0, 1, 0],
            }
        )
        num, cat, dt, bl = split_feature_types_v2(df, target="target", excluded=("id",))
        assert "id" not in num + cat + dt + bl
        assert num == ["age"]
        assert bl == ["flag"]


class TestIQRClipper:
    def test_clips_high_and_low(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])
        clipper = IQRClipper(k=1.5).fit(X)
        out = clipper.transform(X)
        # 상한은 3rd quartile + 1.5*IQR 근처 → 100 은 clip 대상
        assert out.max() < 100.0
        # 원본 중앙값은 보존
        assert np.isclose(out[2, 0], 3.0)

    def test_sklearn_clone_preserves_params(self) -> None:
        from sklearn.base import clone

        c = IQRClipper(k=2.0)
        c2 = clone(c)
        assert c2.k == 2.0
        # clone 은 새 인스턴스여야 함
        assert c is not c2

    def test_nan_passthrough(self) -> None:
        X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        out = IQRClipper(k=1.5).fit(X).transform(X)
        assert np.isnan(out[2, 0])


class TestWinsorizer:
    def test_winsorize_clips_tails(self) -> None:
        X = np.arange(1, 101, dtype=float).reshape(-1, 1)  # 1..100
        w = Winsorizer(p=0.05).fit(X)
        out = w.transform(X)
        # 5% / 95% 분위수로 clip → 하/상단이 축소
        assert out.min() >= np.nanpercentile(X, 5) - 1e-9
        assert out.max() <= np.nanpercentile(X, 95) + 1e-9


class TestFrequencyEncoder:
    def test_encodes_to_frequency_ratios(self) -> None:
        X = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "c"]})
        enc = FrequencyEncoder().fit(X)
        out = enc.transform(X)
        assert out.shape == (6, 1)
        # a=3/6, b=2/6, c=1/6
        assert np.isclose(out[0, 0], 0.5)
        assert np.isclose(out[3, 0], 2 / 6)
        assert np.isclose(out[5, 0], 1 / 6)

    def test_unseen_category_maps_to_zero(self) -> None:
        X_train = pd.DataFrame({"c": ["a", "a", "b"]})
        X_test = pd.DataFrame({"c": ["a", "unseen"]})
        enc = FrequencyEncoder().fit(X_train)
        out = enc.transform(X_test)
        assert np.isclose(out[0, 0], 2 / 3)
        assert np.isclose(out[1, 0], 0.0)


class TestBuildPreprocessorConfig:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [10.0, 20.0, np.nan, 40.0, 200.0],  # 이상치 200
                "income": [50.0, 60.0, 55.0, 70.0, 65.0],
                "city": ["a", "b", "a", "c", "b"],
            }
        )

    def test_default_config_matches_mvp_shape(self) -> None:
        # config=PreprocessingConfig() 와 config=None 의 fit_transform 결과 shape 이 동일해야 함
        df = self._make_df()
        pre_none = build_preprocessor(["age", "income"], ["city"])
        pre_default = build_preprocessor(["age", "income"], ["city"], config=PreprocessingConfig())
        out_none = pre_none.fit_transform(df)
        out_default = pre_default.fit_transform(df)
        assert out_none.shape == out_default.shape

    def test_numeric_scale_none_skips_scaler(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(numeric_scale="none")
        pre = build_preprocessor(["age", "income"], [], config=cfg)
        out = pre.fit_transform(df)
        # 스케일링이 없으므로 원본 값 범위를 유지 (NaN 은 median 대치 후 통과)
        assert out.max() >= 50.0

    def test_numeric_scale_robust(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(numeric_scale="robust")
        pre = build_preprocessor(["age", "income"], [], config=cfg)
        out = pre.fit_transform(df)
        assert out.shape == (5, 2)

    def test_numeric_impute_constant_zero(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(numeric_impute="constant_zero", numeric_scale="none")
        pre = build_preprocessor(["age", "income"], [], config=cfg)
        out = pre.fit_transform(df)
        # NaN row(index=2) 의 age → 0 으로 대치, scale=none 이므로 원값 유지
        assert out[2, 0] == 0.0

    def test_iqr_clip_narrows_range(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(
            numeric_impute="median",
            numeric_scale="none",
            outlier="iqr_clip",
            outlier_iqr_k=1.5,
        )
        pre = build_preprocessor(["age", "income"], [], config=cfg)
        out = pre.fit_transform(df)
        # 이상치 200 이 clip 되었는지
        assert out[:, 0].max() < 200.0

    def test_categorical_encoding_ordinal(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(categorical_encoding="ordinal")
        pre = build_preprocessor([], ["city"], config=cfg)
        out = pre.fit_transform(df)
        # ordinal → 단일 컬럼
        assert out.shape == (5, 1)
        # 정수 코드
        assert np.all(out >= 0)

    def test_categorical_encoding_frequency(self) -> None:
        df = self._make_df()
        cfg = PreprocessingConfig(categorical_encoding="frequency")
        pre = build_preprocessor([], ["city"], config=cfg)
        out = pre.fit_transform(df)
        assert out.shape == (5, 1)
        # 비율이 [0, 1] 범위
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_highcard_auto_downgrade_records_in_report(self) -> None:
        # city 의 nunique 를 임계보다 크게 구성
        df = pd.DataFrame(
            {
                "city": [f"c_{i}" for i in range(60)],
                "tier": ["a", "b"] * 30,
            }
        )
        cfg = PreprocessingConfig(
            categorical_encoding="onehot",
            highcard_threshold=50,
            highcard_auto_downgrade=True,
        )
        pre = build_preprocessor([], ["city", "tier"], config=cfg, df_sample=df)
        report: PreprocessingRouteReport = pre._route_report_
        assert "city" in report.auto_downgraded
        assert "tier" not in report.auto_downgraded
        assert report.encoding_per_col["city"] == "frequency"
        assert report.encoding_per_col["tier"] == "onehot"

    def test_highcard_downgrade_disabled_keeps_onehot(self) -> None:
        df = pd.DataFrame({"city": [f"c_{i}" for i in range(60)]})
        cfg = PreprocessingConfig(
            categorical_encoding="onehot",
            highcard_threshold=50,
            highcard_auto_downgrade=False,
        )
        pre = build_preprocessor([], ["city"], config=cfg, df_sample=df)
        report: PreprocessingRouteReport = pre._route_report_
        assert report.auto_downgraded == ()
        assert report.encoding_per_col["city"] == "onehot"

    def test_bool_passthrough_native_bool(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 20, 30],
                "flag": [True, False, True],
            }
        )
        cfg = PreprocessingConfig(bool_as_numeric=True)
        pre = build_preprocessor(["age"], [], config=cfg, bool_cols=["flag"])
        out = pre.fit_transform(df)
        # num(1) + bool passthrough(1) = 2 컬럼
        assert out.shape == (3, 2)

    def test_bool_as_categorical_when_flag_off(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 20, 30, 40],
                "flag": [True, False, True, False],
            }
        )
        cfg = PreprocessingConfig(bool_as_numeric=False, categorical_encoding="onehot")
        pre = build_preprocessor(["age"], [], config=cfg, bool_cols=["flag"])
        out = pre.fit_transform(df)
        # bool 이 범주로 편입되어 onehot 으로 확장됨 (True/False 2개) → 1 + 2 = 3 cols
        assert out.shape == (4, 3)

    def test_datetime_decompose_true_guarded_without_feature_engineering(self) -> None:
        """§9.4 미구현 상태면 build 시 NotImplementedError, 구현 후면 fit_transform 성공."""
        df = pd.DataFrame(
            {
                "signup": pd.date_range("2024-01-01", periods=3, freq="D"),
                "age": [10, 20, 30],
            }
        )
        cfg = PreprocessingConfig(
            datetime_decompose=True,
            datetime_parts=("year", "month"),
        )
        try:
            from ml.feature_engineering import DatetimeDecomposer  # noqa: F401
        except ImportError:
            with pytest.raises(NotImplementedError, match="§9.4"):
                build_preprocessor(["age"], [], config=cfg, datetime_cols=["signup"])
        else:
            pre = build_preprocessor(["age"], [], config=cfg, datetime_cols=["signup"])
            out = pre.fit_transform(df)
            assert out.shape[0] == 3

    def test_datetime_decompose_true_integrates_with_feature_engineering(self) -> None:
        """§9.4 통합 후 경로: decompose=True 가 실제 동작해야 함."""
        df = pd.DataFrame(
            {
                "signup": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
                "age": [10, 20, 30],
            }
        )
        cfg = PreprocessingConfig(
            datetime_decompose=True,
            datetime_parts=("year", "month"),
        )
        pre = build_preprocessor(["age"], [], config=cfg, datetime_cols=["signup"])
        out = pre.fit_transform(df)
        # num(1) + datetime decompose(2 parts) = 3 컬럼
        assert out.shape == (3, 3)
        assert not np.isnan(out).any()

    def test_datetime_decompose_false_excludes_datetime_cols(self) -> None:
        df = pd.DataFrame(
            {
                "signup": pd.date_range("2024-01-01", periods=3, freq="D"),
                "age": [10, 20, 30],
            }
        )
        cfg = PreprocessingConfig(datetime_decompose=False)
        pre = build_preprocessor(["age"], [], config=cfg, datetime_cols=["signup"])
        out = pre.fit_transform(df)
        # datetime 은 제외되고 age 만 변환 → 1 컬럼
        assert out.shape == (3, 1)


class TestPlanCategoricalRouting:
    def test_onehot_downgrades_above_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "low_card": ["a", "b"] * 30,
                "high_card": [f"c_{i}" for i in range(60)],
            }
        )
        cfg = PreprocessingConfig(
            categorical_encoding="onehot",
            highcard_threshold=50,
            highcard_auto_downgrade=True,
        )
        report = plan_categorical_routing(df, ["low_card", "high_card"], cfg)
        assert report.encoding_per_col["low_card"] == "onehot"
        assert report.encoding_per_col["high_card"] == "frequency"
        assert report.auto_downgraded == ("high_card",)

    def test_none_df_sample_keeps_requested_encoding(self) -> None:
        cfg = PreprocessingConfig(categorical_encoding="onehot")
        report = plan_categorical_routing(None, ["a", "b"], cfg)
        assert report.encoding_per_col == {"a": "onehot", "b": "onehot"}
        assert report.auto_downgraded == ()


class TestBuildFeatureSchemaExtended:
    def test_default_path_unchanged(self) -> None:
        # config=None 이면 derived/datetime 는 빈 tuple
        df = pd.DataFrame({"age": [1, 2], "city": ["a", "b"]})
        schema = build_feature_schema(df, ["age"], ["city"], target="y")
        assert schema.derived == ()
        assert schema.datetime == ()

    def test_config_enumerates_onehot_derived(self) -> None:
        df = pd.DataFrame({"city": ["a", "b", "a", "c"]})
        cfg = PreprocessingConfig(categorical_encoding="onehot")
        schema = build_feature_schema(df, [], ["city"], target="y", config=cfg)
        kinds = {d.kind for d in schema.derived}
        assert kinds == {"onehot"}
        names = sorted(d.name for d in schema.derived)
        assert names == ["city__a", "city__b", "city__c"]

    def test_config_with_datetime_decompose(self) -> None:
        df = pd.DataFrame(
            {
                "signup": pd.date_range("2024-01-01", periods=3, freq="D"),
            }
        )
        cfg = PreprocessingConfig(
            datetime_decompose=True,
            datetime_parts=("year", "month"),
        )
        schema = build_feature_schema(df, [], [], target="y", datetime_cols=["signup"], config=cfg)
        assert schema.datetime == ("signup",)
        kinds = [d.kind for d in schema.derived]
        assert "datetime_year" in kinds
        assert "datetime_month" in kinds
        sources = {d.source for d in schema.derived}
        assert sources == {"signup"}

    def test_config_with_bool_passthrough(self) -> None:
        df = pd.DataFrame({"flag": [True, False, True]})
        cfg = PreprocessingConfig(bool_as_numeric=True)
        schema = build_feature_schema(df, [], [], target="y", bool_cols=["flag"], config=cfg)
        assert len(schema.derived) == 1
        assert schema.derived[0].kind == "bool_numeric"
        assert schema.derived[0].source == "flag"

    def test_config_uses_route_report_for_downgrade(self) -> None:
        df = pd.DataFrame({"city": [f"c_{i}" for i in range(10)]})
        cfg = PreprocessingConfig(
            categorical_encoding="onehot",
            highcard_threshold=5,
            highcard_auto_downgrade=True,
        )
        report = plan_categorical_routing(df, ["city"], cfg)
        schema = build_feature_schema(df, [], ["city"], target="y", config=cfg, route_report=report)
        # city 는 frequency 로 다운그레이드 → 단일 derived (kind=frequency)
        assert len(schema.derived) == 1
        assert schema.derived[0].kind == "frequency"
