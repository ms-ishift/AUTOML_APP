from __future__ import annotations

import numpy as np
import pandas as pd

from ml.profiling import profile_dataframe, suggest_excluded
from ml.schemas import ColumnProfile, DatasetProfile


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [20, 30, np.nan, 40, 50],
            "city": ["A", "B", "A", "B", None],
            "label": [0, 1, 0, 1, 1],
        }
    )


def test_profile_basic_shape() -> None:
    df = _sample_df()
    profile = profile_dataframe(df)

    assert isinstance(profile, DatasetProfile)
    assert profile.n_rows == 5
    assert profile.n_cols == 4
    assert len(profile.columns) == 4
    assert all(isinstance(c, ColumnProfile) for c in profile.columns)


def test_profile_missing_and_unique() -> None:
    df = _sample_df()
    p = profile_dataframe(df)

    age = p.column("age")
    city = p.column("city")
    identifier = p.column("id")

    assert age is not None and age.n_missing == 1
    assert city is not None and city.n_missing == 1
    assert identifier is not None and identifier.n_unique == 5
    assert identifier.unique_ratio == 1.0


def test_profile_empty_df_is_safe() -> None:
    df = pd.DataFrame({"x": pd.Series(dtype="float64")})
    p = profile_dataframe(df)
    assert p.n_rows == 0
    assert p.n_cols == 1
    col = p.column("x")
    assert col is not None
    assert col.missing_ratio == 0.0


def test_suggest_excluded_flags_identifier() -> None:
    df = _sample_df()
    p = profile_dataframe(df)
    excluded = suggest_excluded(p)
    assert "id" in excluded
    assert "age" not in excluded


def test_suggest_excluded_skips_tiny_df() -> None:
    df = pd.DataFrame({"id": [1]})
    p = profile_dataframe(df)
    assert suggest_excluded(p) == []


def test_suggest_excluded_threshold_respected() -> None:
    df = pd.DataFrame(
        {
            "dup": [1, 1, 2, 2, 3, 3, 3, 3, 3, 3],  # 고유비율 0.3 → 제외 아님
        }
    )
    p = profile_dataframe(df)
    assert suggest_excluded(p, unique_ratio_threshold=0.9) == []
    assert suggest_excluded(p, unique_ratio_threshold=0.2) == ["dup"]
