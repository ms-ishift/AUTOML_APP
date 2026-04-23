"""ml.feature_engineering 단위 테스트 (IMPLEMENTATION_PLAN §9.4, FR-056)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.feature_engineering import BoolToNumeric, DatetimeDecomposer

# =========================================================== DatetimeDecomposer


class TestDatetimeDecomposer:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signup": pd.to_datetime(
                    ["2024-01-01 08:30", "2024-06-15 14:00", "2024-12-31 23:59"]
                ),
            }
        )

    def test_fit_records_feature_names(self) -> None:
        df = self._df()
        dd = DatetimeDecomposer(parts=("year", "month"))
        dd.fit(df)
        assert list(dd.feature_names_in_) == ["signup"]
        assert dd.n_features_in_ == 1

    def test_transform_extracts_year_month_day(self) -> None:
        df = self._df()
        dd = DatetimeDecomposer(parts=("year", "month", "day"))
        out = dd.fit_transform(df)
        assert out.shape == (3, 3)
        # 2024-01-01 → (2024, 1, 1)
        assert out[0, 0] == 2024
        assert out[0, 1] == 1
        assert out[0, 2] == 1
        # 2024-12-31
        assert out[2, 1] == 12
        assert out[2, 2] == 31

    def test_weekday_and_is_weekend(self) -> None:
        # 2024-01-01 = Monday(0), 2024-01-06 = Saturday(5), 2024-01-07 = Sunday(6)
        df = pd.DataFrame(
            {
                "d": pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-07"]),
            }
        )
        dd = DatetimeDecomposer(parts=("weekday", "is_weekend"))
        out = dd.fit_transform(df)
        assert out[0, 0] == 0  # Monday
        assert out[1, 0] == 5
        assert out[2, 0] == 6
        assert out[0, 1] == 0.0  # 평일
        assert out[1, 1] == 1.0  # 주말
        assert out[2, 1] == 1.0

    def test_hour_extracts_time_of_day(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2024-01-01 08:30", "2024-01-01 14:00", "2024-01-01 23:59"]),
            }
        )
        dd = DatetimeDecomposer(parts=("hour",))
        out = dd.fit_transform(df)
        assert list(out[:, 0]) == [8.0, 14.0, 23.0]

    def test_nat_propagates_to_nan(self) -> None:
        df = pd.DataFrame(
            {
                "d": [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-03-15")],
            }
        )
        dd = DatetimeDecomposer(parts=("year", "month"))
        out = dd.fit_transform(df)
        assert out[0, 0] == 2024
        assert np.isnan(out[1, 0])
        assert np.isnan(out[1, 1])

    def test_object_dtype_parsed_with_coerce(self) -> None:
        # pd.to_datetime(errors="coerce") 로 문자열도 파싱, 실패는 NaT → NaN
        df = pd.DataFrame({"d": ["2024-01-01", "not-a-date", "2024-02-15"]})
        dd = DatetimeDecomposer(parts=("year",))
        out = dd.fit_transform(df)
        assert out[0, 0] == 2024
        assert np.isnan(out[1, 0])
        assert out[2, 0] == 2024

    def test_multi_column_output_ordering(self) -> None:
        df = pd.DataFrame(
            {
                "a": pd.to_datetime(["2024-01-01", "2024-06-01"]),
                "b": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            }
        )
        dd = DatetimeDecomposer(parts=("year", "month"))
        out = dd.fit_transform(df)
        # 출력 순서: a_year, a_month, b_year, b_month
        assert out.shape == (2, 4)
        assert out[0, 0] == 2024  # a_year[0]
        assert out[0, 2] == 2023  # b_year[0]
        assert out[1, 3] == 6  # b_month[1]

    def test_feature_names_out_after_fit(self) -> None:
        df = self._df()
        dd = DatetimeDecomposer(parts=("year", "month")).fit(df)
        names = dd.get_feature_names_out()
        assert list(names) == ["signup_year", "signup_month"]

    def test_feature_names_out_with_explicit_input(self) -> None:
        dd = DatetimeDecomposer(parts=("year",))
        names = dd.get_feature_names_out(input_features=["a", "b"])
        assert list(names) == ["a_year", "b_year"]

    def test_invalid_part_raises(self) -> None:
        dd = DatetimeDecomposer(parts=("year", "INVALID"))
        with pytest.raises(ValueError, match="지원하지 않는 datetime part"):
            dd.fit(pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])}))

    def test_empty_parts_raises(self) -> None:
        dd = DatetimeDecomposer(parts=())
        with pytest.raises(ValueError, match="비어 있습니다"):
            dd.fit(pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])}))

    def test_sklearn_clone_preserves_parts(self) -> None:
        dd = DatetimeDecomposer(parts=("year", "is_weekend"))
        cloned = clone(dd)
        assert cloned.parts == ("year", "is_weekend")
        assert cloned is not dd

    def test_pipeline_with_simple_imputer_fills_nat(self) -> None:
        df = pd.DataFrame({"d": [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-03-01")]})
        pipe = Pipeline(
            steps=[
                ("decompose", DatetimeDecomposer(parts=("year", "month"))),
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        out = pipe.fit_transform(df)
        assert out.shape == (3, 2)
        assert not np.isnan(out).any()


# =========================================================== BoolToNumeric


class TestBoolToNumeric:
    def test_native_bool_dtype(self) -> None:
        df = pd.DataFrame({"flag": [True, False, True]})
        out = BoolToNumeric().fit_transform(df)
        assert out.shape == (3, 1)
        assert list(out[:, 0]) == [1.0, 0.0, 1.0]

    def test_int_0_1_values(self) -> None:
        df = pd.DataFrame({"flag": [0, 1, 1, 0]})
        out = BoolToNumeric().fit_transform(df)
        assert list(out[:, 0]) == [0.0, 1.0, 1.0, 0.0]

    def test_int_out_of_domain_maps_to_nan(self) -> None:
        df = pd.DataFrame({"flag": [0, 1, 2, 3]})
        out = BoolToNumeric().fit_transform(df)
        assert out[0, 0] == 0.0
        assert out[1, 0] == 1.0
        assert np.isnan(out[2, 0])
        assert np.isnan(out[3, 0])

    def test_yes_no_tokens(self) -> None:
        df = pd.DataFrame({"flag": ["yes", "no", "Yes", " NO "]})
        out = BoolToNumeric().fit_transform(df)
        assert list(out[:, 0]) == [1.0, 0.0, 1.0, 0.0]

    def test_true_false_tokens(self) -> None:
        df = pd.DataFrame({"flag": ["true", "false", "T", "F"]})
        out = BoolToNumeric().fit_transform(df)
        assert list(out[:, 0]) == [1.0, 0.0, 1.0, 0.0]

    def test_unknown_token_maps_to_nan(self) -> None:
        df = pd.DataFrame({"flag": ["maybe", "yes", None, "y"]})
        out = BoolToNumeric().fit_transform(df)
        assert np.isnan(out[0, 0])
        assert out[1, 0] == 1.0
        assert np.isnan(out[2, 0])
        assert out[3, 0] == 1.0

    def test_nan_in_object_maps_to_nan(self) -> None:
        df = pd.DataFrame({"flag": [np.nan, "yes", "no"]}, dtype=object)
        out = BoolToNumeric().fit_transform(df)
        assert np.isnan(out[0, 0])
        assert out[1, 0] == 1.0
        assert out[2, 0] == 0.0

    def test_multi_column_output(self) -> None:
        df = pd.DataFrame(
            {
                "a": [True, False, True],
                "b": ["y", "n", "y"],
            }
        )
        out = BoolToNumeric().fit_transform(df)
        assert out.shape == (3, 2)
        assert list(out[:, 0]) == [1.0, 0.0, 1.0]
        assert list(out[:, 1]) == [1.0, 0.0, 1.0]

    def test_custom_tokens_override_defaults(self) -> None:
        df = pd.DataFrame({"flag": ["on", "off", "ON"]})
        enc = BoolToNumeric(true_tokens=("on",), false_tokens=("off",))
        out = enc.fit_transform(df)
        assert list(out[:, 0]) == [1.0, 0.0, 1.0]

    def test_overlapping_tokens_raises(self) -> None:
        enc = BoolToNumeric(true_tokens=("yes", "shared"), false_tokens=("no", "shared"))
        with pytest.raises(ValueError, match="겹칩"):
            enc.fit(pd.DataFrame({"flag": ["yes"]}))

    def test_sklearn_clone_preserves_params(self) -> None:
        enc = BoolToNumeric(true_tokens=("on",), false_tokens=("off",))
        cloned = clone(enc)
        assert cloned.true_tokens == ("on",)
        assert cloned.false_tokens == ("off",)
        assert cloned is not enc

    def test_feature_names_out_after_fit(self) -> None:
        enc = BoolToNumeric().fit(pd.DataFrame({"flag": [True, False]}))
        names = enc.get_feature_names_out()
        assert list(names) == ["flag"]

    def test_pipeline_with_imputer_fills_unknown(self) -> None:
        df = pd.DataFrame({"flag": ["yes", "maybe", "no"]})
        pipe = Pipeline(
            steps=[
                ("normalize", BoolToNumeric()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        out = pipe.fit_transform(df)
        assert out.shape == (3, 1)
        assert not np.isnan(out).any()
