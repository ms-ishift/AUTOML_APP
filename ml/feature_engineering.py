"""피처 공학 트랜스포머 (IMPLEMENTATION_PLAN §9.4, FR-056).

원칙:
- 순수 pandas/numpy/sklearn 의존. Streamlit/SQLAlchemy 미사용.
- 모든 트랜스포머는 ``sklearn.base.BaseEstimator + TransformerMixin`` 으로 구현해
  ``clone`` / ``ColumnTransformer`` / ``Pipeline`` 과 자연스럽게 합성된다.
- ``fit`` 은 학습 시점 메타정보(컬럼명/카디널리티 등)만 기억하며,
  결측 대치(imputation) 는 본 모듈에서 수행하지 않는다. 후단 ``SimpleImputer`` 가
  처리한다고 가정 (§9.3 에서 이미 파이프라인 결합).

제공 트랜스포머:
- ``DatetimeDecomposer(parts)`` — datetime 컬럼 → year/month/day/weekday/hour/is_weekend
- ``BoolToNumeric(true_tokens, false_tokens)`` — bool / 0·1 / Y·N 등 토큰 → {0.0, 1.0, NaN}

후속 이월(§9.11 에서 다룸):
- ``LogTransformer`` / ``YeoJohnsonTransformer`` — 왜도 높은 수치형 변환 (이번 스프린트 제외).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "DEFAULT_DATETIME_PARTS",
    "DEFAULT_FALSE_TOKENS",
    "DEFAULT_TRUE_TOKENS",
    "BoolToNumeric",
    "DatetimeDecomposer",
]


DEFAULT_DATETIME_PARTS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "is_weekend",
)

# BoolToNumeric 기본 토큰 (소문자·공백 제거 후 비교)
DEFAULT_TRUE_TOKENS: frozenset[str] = frozenset({"true", "t", "yes", "y", "1"})
DEFAULT_FALSE_TOKENS: frozenset[str] = frozenset({"false", "f", "no", "n", "0"})


# ============================================================ DatetimeDecomposer


_PART_EXTRACTORS: dict[str, Any] = {
    "year": lambda dt: dt.dt.year,
    "month": lambda dt: dt.dt.month,
    "day": lambda dt: dt.dt.day,
    "weekday": lambda dt: dt.dt.weekday,
    "hour": lambda dt: dt.dt.hour,
    # pandas weekday: Monday=0 ... Sunday=6 → weekend = {5, 6}
    "is_weekend": lambda dt: dt.dt.weekday.isin([5, 6]).astype("float"),
}


class DatetimeDecomposer(BaseEstimator, TransformerMixin):
    """datetime 컬럼을 ``parts`` 로 분해하는 트랜스포머.

    - ``fit`` 은 입력 컬럼명만 ``feature_names_in_`` 으로 기억한다.
      파이프라인 결합을 위해 표준 sklearn 속성(``n_features_in_``) 도 세팅.
    - ``transform`` 은 (n_rows, n_input_cols × len(parts)) 형태의 float numpy array
      를 반환한다. NaT 는 NaN 으로 전파되며, 후단 ``SimpleImputer`` 가 대치한다.
    - 출력 컬럼 순서는 입력 컬럼 우선(``col1_part1, col1_part2, ..., col2_part1, ...``).

    지원 parts: ``year, month, day, weekday, hour, is_weekend``.
    Unknown part 가 ``parts`` 에 섞이면 ``__init__`` 에서 즉시 ``ValueError``.
    (``PreprocessingConfig`` Literal 이 이미 검증하지만 직접 호출 보호).
    """

    def __init__(self, parts: tuple[str, ...] = DEFAULT_DATETIME_PARTS) -> None:
        self.parts = parts

    def _validate_parts(self) -> tuple[str, ...]:
        parts_tuple = tuple(self.parts)
        if not parts_tuple:
            raise ValueError("DatetimeDecomposer.parts 가 비어 있습니다.")
        unknown = [p for p in parts_tuple if p not in _PART_EXTRACTORS]
        if unknown:
            raise ValueError(
                f"지원하지 않는 datetime part: {unknown}. " f"허용값: {sorted(_PART_EXTRACTORS)}"
            )
        return parts_tuple

    def fit(self, X: Any, y: Any = None) -> DatetimeDecomposer:  # noqa: ARG002
        self._validate_parts()
        df = self._as_frame(X)
        self.feature_names_in_ = np.array([str(c) for c in df.columns], dtype=object)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        parts = self._validate_parts()
        df = self._as_frame(X)
        n_rows = df.shape[0]
        if df.shape[1] == 0:
            return np.empty((n_rows, 0), dtype=float)

        out_cols: list[np.ndarray] = []
        for col in df.columns:
            series = pd.to_datetime(df[col], errors="coerce")
            for part in parts:
                extractor = _PART_EXTRACTORS[part]
                values = extractor(series).to_numpy(dtype=float)
                out_cols.append(values)
        return np.column_stack(out_cols)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """sklearn ColumnTransformer 에서 파생 컬럼명을 조회할 때 사용."""
        parts = self._validate_parts()
        if input_features is None:
            if not hasattr(self, "feature_names_in_"):
                raise ValueError("get_feature_names_out 은 fit 이후에만 호출할 수 있습니다.")
            input_features = self.feature_names_in_
        names: list[str] = []
        for col in input_features:
            for part in parts:
                names.append(f"{col}_{part}")
        return np.array(names, dtype=object)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, pd.Series):
            return X.to_frame()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr)


# ============================================================ BoolToNumeric


class BoolToNumeric(BaseEstimator, TransformerMixin):
    """bool / 0·1 / Y·N 등 토큰 컬럼을 {0.0, 1.0} float 로 정규화.

    - 네이티브 bool dtype → ``astype(float)`` 로 {0.0, 1.0}.
    - 수치형(int/float) → 값이 0/1 이면 유지, 그 외는 NaN (훈련 도메인 밖).
    - object/문자열 → ``true_tokens`` / ``false_tokens`` 에 매핑. 소문자·공백 제거 비교.
      ``None`` / NaN / 정의되지 않은 토큰 → NaN (후단 SimpleImputer 가 대치).

    sklearn clone 은 ``__init__`` 하이퍼(true_tokens, false_tokens) 를 그대로 보존한다.
    """

    def __init__(
        self,
        true_tokens: frozenset[str] | set[str] | tuple[str, ...] = DEFAULT_TRUE_TOKENS,
        false_tokens: frozenset[str] | set[str] | tuple[str, ...] = DEFAULT_FALSE_TOKENS,
    ) -> None:
        self.true_tokens = true_tokens
        self.false_tokens = false_tokens

    def _normalized_tokens(self) -> tuple[frozenset[str], frozenset[str]]:
        t = frozenset(str(x).strip().lower() for x in self.true_tokens)
        f = frozenset(str(x).strip().lower() for x in self.false_tokens)
        overlap = t & f
        if overlap:
            raise ValueError(f"true_tokens 와 false_tokens 가 겹칩니다: {sorted(overlap)}")
        return t, f

    def fit(self, X: Any, y: Any = None) -> BoolToNumeric:  # noqa: ARG002
        self._normalized_tokens()  # 검증만
        df = self._as_frame(X)
        self.feature_names_in_ = np.array([str(c) for c in df.columns], dtype=object)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        true_set, false_set = self._normalized_tokens()
        df = self._as_frame(X)
        if df.shape[1] == 0:
            return np.empty((df.shape[0], 0), dtype=float)

        out_cols: list[np.ndarray] = []
        for col in df.columns:
            series = df[col]
            out_cols.append(self._normalize_series(series, true_set, false_set))
        return np.column_stack(out_cols)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            if not hasattr(self, "feature_names_in_"):
                raise ValueError("get_feature_names_out 은 fit 이후에만 호출할 수 있습니다.")
            input_features = self.feature_names_in_
        return np.array([str(c) for c in input_features], dtype=object)

    @staticmethod
    def _normalize_series(
        series: pd.Series,
        true_set: frozenset[str],
        false_set: frozenset[str],
    ) -> np.ndarray:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(float).to_numpy()

        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=float, na_value=np.nan)
            # 0/1 이외의 값은 NaN 으로 취급 (토큰 매핑 의도 밖)
            result = np.where(arr == 1.0, 1.0, np.where(arr == 0.0, 0.0, np.nan))
            return result.astype(float)

        # object / string
        def _map(value: Any) -> float:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return np.nan
            key = str(value).strip().lower()
            if key in true_set:
                return 1.0
            if key in false_set:
                return 0.0
            return np.nan

        return series.map(_map).to_numpy(dtype=float)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, pd.Series):
            return X.to_frame()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr)
