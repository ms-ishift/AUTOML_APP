"""전처리 파이프라인 (IMPLEMENTATION_PLAN §3.4 / §9.3, FR-050~053, FR-055, FR-056).

원칙:
- 이 모듈은 Streamlit/DB 에 의존하지 않는다.
- ColumnTransformer 는 새 알고리즘 학습마다 ``sklearn.base.clone`` 으로 복제해 사용한다
  (같은 인스턴스를 여러 Pipeline 에 재사용하면 fit 상태가 공유되어 재현성이 깨진다).

§9.3 확장 요약:
- ``config`` (PreprocessingConfig) 를 받으면 수치 imputer / scaler / 이상치 클립,
  범주형 encoder (onehot / ordinal / frequency, 고카디널리티 자동 폴백),
  bool passthrough 를 구성한다. ``config=None`` 이면 기존 기본 동작과 동일.
- 이상치 처리는 ``IQRClipper`` / ``Winsorizer`` 를 sklearn BaseEstimator 로 직접 구현.
- 범주형 고카디널리티 라우팅 결과는 ``PreprocessingRouteReport`` 로 별도 반환하며,
  동일 객체가 ``build_preprocessor`` 가 반환하는 ColumnTransformer 에
  ``_route_report_`` 속성으로도 부착된다 (UI 미리보기 §9.9 용).
- Datetime decompose / string-token bool 정규화는 §9.4 의 ``ml/feature_engineering``
  모듈이 필요하다. 본 §9.3 에서는 그 경로를 진입하려 하면 ``NotImplementedError``
  를 올려 명시적으로 막는다 (기본 경로는 영향 없음).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from ml.schemas import (
    DerivedFeature,
    FeatureSchema,
    PreprocessingConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "FrequencyEncoder",
    "IQRClipper",
    "PreprocessingRouteReport",
    "Winsorizer",
    "build_feature_schema",
    "build_preprocessor",
    "plan_categorical_routing",
    "prepare_xy",
    "split_feature_types",
    "split_feature_types_v2",
]


# ============================================================ route report


@dataclass(frozen=True, slots=True)
class PreprocessingRouteReport:
    """범주형 라우팅 결정 기록 (§9.3 / §9.7).

    - ``encoding_per_col``: 최종 적용된 encoding (col → ``"onehot" | "ordinal" | "frequency"``).
    - ``auto_downgraded``: ``onehot`` 요청이었으나 ``highcard_threshold`` 를
      넘겨서 ``frequency`` 로 자동 강등된 컬럼명 튜플.
    """

    encoding_per_col: dict[str, str] = field(default_factory=dict)
    auto_downgraded: tuple[str, ...] = ()


def plan_categorical_routing(
    df_sample: pd.DataFrame | None,
    cat_cols: Iterable[str],
    config: PreprocessingConfig,
) -> PreprocessingRouteReport:
    """범주형 인코딩 라우팅을 계산한다 (실제 fit 없이 메타데이터만).

    - 기본 인코딩은 ``config.categorical_encoding``.
    - ``config.categorical_encoding == "onehot"`` 이고
      ``config.highcard_auto_downgrade=True`` 이면 컬럼별 ``nunique`` 를 보고
      ``config.highcard_threshold`` 를 넘는 컬럼은 ``frequency`` 로 다운그레이드.
    - ``df_sample=None`` 이면 모든 컬럼을 요청 인코딩 그대로 라우팅 (강등 없음).
    """
    requested = config.categorical_encoding
    encoding: dict[str, str] = {}
    downgraded: list[str] = []
    for col in cat_cols:
        col_str = str(col)
        if (
            requested == "onehot"
            and config.highcard_auto_downgrade
            and df_sample is not None
            and col_str in df_sample.columns
        ):
            nu = int(df_sample[col_str].nunique(dropna=True))
            if nu > config.highcard_threshold:
                encoding[col_str] = "frequency"
                downgraded.append(col_str)
                continue
        encoding[col_str] = requested
    return PreprocessingRouteReport(
        encoding_per_col=encoding,
        auto_downgraded=tuple(downgraded),
    )


# ============================================================ transformers


class IQRClipper(BaseEstimator, TransformerMixin):
    """IQR-기반 이상치 clipper (§9.3).

    fit 시 컬럼별 Q1/Q3 를 저장하고, transform 시 ``[Q1 - k·IQR, Q3 + k·IQR]`` 로
    clip 한다. NaN 은 그대로 통과 — 후단 imputer 가 처리한다고 가정.
    sklearn ``clone`` 은 ``__init__`` 의 하이퍼파라미터를 그대로 보존하므로
    별도 ``__sklearn_clone__`` 정의 없이 호환된다.
    """

    def __init__(self, k: float = 1.5) -> None:
        self.k = k

    def fit(self, X: Any, y: Any = None) -> IQRClipper:  # noqa: ARG002
        arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = arr.shape[1] if arr.ndim == 2 else 1
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lower_, self.upper_)


class Winsorizer(BaseEstimator, TransformerMixin):
    """분위수 기반 Winsorize clipper (§9.3).

    fit 시 ``p`` / ``1-p`` 분위수를 저장하고 transform 시 clip.
    ``p`` 는 (0, 0.5) 범위이며 ``PreprocessingConfig.__post_init__`` 에서 이미 검증된다.
    """

    def __init__(self, p: float = 0.01) -> None:
        self.p = p

    def fit(self, X: Any, y: Any = None) -> Winsorizer:  # noqa: ARG002
        arr = np.asarray(X, dtype=float)
        lower = np.nanpercentile(arr, self.p * 100.0, axis=0)
        upper = np.nanpercentile(arr, (1.0 - self.p) * 100.0, axis=0)
        self.lower_ = lower
        self.upper_ = upper
        self.n_features_in_ = arr.shape[1] if arr.ndim == 2 else 1
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lower_, self.upper_)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """범주 → 학습 데이터 빈도 비율 (0.0~1.0) 로 치환하는 인코더 (§9.3).

    fit 시 컬럼별 ``value → freq_ratio`` 맵을 저장, transform 시 매핑.
    학습에 없던 범주는 0.0 으로 처리 (unseen ratio).
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> FrequencyEncoder:  # noqa: ARG002
        df = pd.DataFrame(X).astype(object)
        self.maps_: list[dict[Any, float]] = []
        for col in df.columns:
            series = df[col]
            denom = max(int(series.dropna().shape[0]), 1)
            counts = series.value_counts(dropna=True)
            self.maps_.append({k: float(v) / denom for k, v in counts.items()})
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        df = pd.DataFrame(X).astype(object)
        cols: list[np.ndarray] = []
        for idx, col in enumerate(df.columns):
            mapping = self.maps_[idx]
            values = df[col].map(lambda v, _m=mapping: _m.get(v, 0.0)).to_numpy(dtype=float)
            cols.append(values)
        return np.column_stack(cols) if cols else np.empty((df.shape[0], 0), dtype=float)


# ============================================================ internal builders


_NUMERIC_IMPUTER_STRATEGY = {
    "median": ("median", None),
    "mean": ("mean", None),
    "most_frequent": ("most_frequent", None),
    "constant_zero": ("constant", 0.0),
}


def _build_numeric_pipeline(config: PreprocessingConfig) -> Pipeline:
    """수치 파이프라인: impute → (optional outlier clip) → scaler."""
    steps: list[tuple[str, Any]] = []

    # 1) Impute. "drop_rows" 는 파이프라인 내에서 처리하지 않는다 (호출 측에서 사전 drop).
    if config.numeric_impute == "drop_rows":
        # 파이프라인 자체는 통과시키되, 호출자가 df 에서 이미 결측 row 를 제거했다고 가정.
        # 안전을 위해 최빈 imputer 를 두어 돌발 NaN 을 방어.
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    else:
        strategy, fill_value = _NUMERIC_IMPUTER_STRATEGY[config.numeric_impute]
        if fill_value is None:
            steps.append(("imputer", SimpleImputer(strategy=strategy)))
        else:
            steps.append(("imputer", SimpleImputer(strategy=strategy, fill_value=fill_value)))

    # 2) Outlier (optional)
    if config.outlier == "iqr_clip":
        steps.append(("outlier", IQRClipper(k=config.outlier_iqr_k)))
    elif config.outlier == "winsorize":
        steps.append(("outlier", Winsorizer(p=config.winsorize_p)))
    # "none" → skip

    # 3) Scaler
    if config.numeric_scale == "standard":
        steps.append(("scaler", StandardScaler()))
    elif config.numeric_scale == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif config.numeric_scale == "robust":
        steps.append(("scaler", RobustScaler()))
    # "none" → skip

    if not steps:
        # 방어: 최소 1개 스텝은 있어야 sklearn Pipeline 이 동작.
        steps.append(("passthrough", SimpleImputer(strategy="most_frequent")))

    return Pipeline(steps=steps)


def _coerce_to_object(X: Any) -> np.ndarray:
    """범주형 파이프라인 입력 전 dtype 정규화 (bool/int → object).

    SimpleImputer 는 bool dtype 을 거부하므로 bool 컬럼이 cat 그룹으로 편입되는
    경로(``bool_as_numeric=False``)에서 사전 변환이 필요하다.
    """
    arr = np.asarray(X)
    if arr.dtype.kind in ("O",):
        return arr
    # bool/int 등은 object 로 승격 (문자 비교 기반 impute/onehot 에 적합)
    return arr.astype(object)


def _build_categorical_pipeline_for_encoding(
    config: PreprocessingConfig,
    encoding: str,
) -> Pipeline:
    """범주형 파이프라인: (dtype 정규화) → impute → encoder."""
    if config.categorical_impute == "constant_missing":
        imputer: Any = SimpleImputer(strategy="constant", fill_value="__MISSING__")
    else:
        imputer = SimpleImputer(strategy="most_frequent")

    if encoding == "onehot":
        encoder: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    elif encoding == "frequency":
        encoder = FrequencyEncoder()
    else:
        raise ValueError(f"알 수 없는 categorical_encoding: {encoding}")

    return Pipeline(
        steps=[
            ("to_object", FunctionTransformer(_coerce_to_object, accept_sparse=False)),
            ("imputer", imputer),
            ("encoder", encoder),
        ]
    )


def _build_datetime_pipeline(
    config: PreprocessingConfig,
) -> Pipeline:
    """datetime 분해 파이프라인 — §9.4 에서 완성.

    ``config.datetime_decompose=True`` 인데 §9.4 의 ``DatetimeDecomposer`` 가 아직
    구현되지 않은 상태에서 본 함수를 호출하면 명시적으로 실패시킨다.
    """
    try:
        from ml.feature_engineering import DatetimeDecomposer
    except ImportError as exc:  # pragma: no cover - §9.4 완료 시 제거
        raise NotImplementedError(
            "Datetime 분해 파이프라인은 §9.4 (ml/feature_engineering.py) 에서 제공됩니다."
        ) from exc

    return Pipeline(
        steps=[
            ("decompose", DatetimeDecomposer(parts=tuple(config.datetime_parts))),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


def _build_bool_passthrough() -> str:
    """bool 컬럼 passthrough 마커.

    sklearn ColumnTransformer 는 ``"passthrough"`` 문자열을 그대로 지원한다.
    네이티브 bool dtype 은 passthrough 시 자동으로 0/1 numeric 으로 취급된다.
    문자 토큰 bool (Y/N) 정규화는 §9.4 의 ``BoolToNumeric`` 을 사용해야 한다.
    """
    return "passthrough"


# ============================================================ public API


def split_feature_types(
    df: pd.DataFrame,
    target: str,
    excluded: tuple[str, ...] | list[str] = (),
) -> tuple[list[str], list[str]]:
    """데이터프레임 컬럼을 수치/범주 두 그룹으로 나눈다. target·excluded 는 제외.

    반환 순서는 DataFrame 의 원본 컬럼 순서를 유지한다. **호환 유지**를 위해
    bool 과 datetime 은 범주로 분류된다. 4-way 분류는 ``split_feature_types_v2``.
    """
    excluded_set = set(excluded)
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in df.columns:
        if col == target or col in excluded_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            num_cols.append(str(col))
        else:
            cat_cols.append(str(col))
    return num_cols, cat_cols


def split_feature_types_v2(
    df: pd.DataFrame,
    target: str,
    excluded: tuple[str, ...] | list[str] = (),
) -> tuple[list[str], list[str], list[str], list[str]]:
    """컬럼을 ``(num, cat, datetime, bool)`` 4-tuple 로 분류한다 (§9.3).

    - ``datetime``: ``is_datetime64_any_dtype`` True
    - ``bool``: ``is_bool_dtype`` True (네이티브 bool만; 문자 토큰은 §9.4 에서
      `ml.type_inference.detect_bool_columns` 와 결합하여 보강)
    - ``num``: 수치형이면서 bool 아닌 경우
    - ``cat``: 위 어디에도 속하지 않는 나머지

    반환 순서는 원본 컬럼 순서를 유지한다. target/excluded 는 모두에서 제외.
    """
    excluded_set = set(excluded)
    num_cols: list[str] = []
    cat_cols: list[str] = []
    dt_cols: list[str] = []
    bool_cols: list[str] = []
    for col in df.columns:
        if col == target or col in excluded_set:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt_cols.append(str(col))
        elif pd.api.types.is_bool_dtype(s):
            bool_cols.append(str(col))
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(str(col))
        else:
            cat_cols.append(str(col))
    return num_cols, cat_cols, dt_cols, bool_cols


def _make_cat_transformers(
    cat_list: list[str],
    route_report: PreprocessingRouteReport,
    config: PreprocessingConfig,
) -> list[tuple[str, Any, list[str]]]:
    """encoding 별 그룹핑 후 (이름, Pipeline, 컬럼) 튜플 리스트를 만든다."""
    encoding_to_cols: dict[str, list[str]] = {}
    for col in cat_list:
        enc = route_report.encoding_per_col.get(col, config.categorical_encoding)
        encoding_to_cols.setdefault(enc, []).append(col)
    return [
        (f"cat_{enc}", _build_categorical_pipeline_for_encoding(config, enc), cols)
        for enc, cols in encoding_to_cols.items()
    ]


def build_preprocessor(
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
    *,
    config: PreprocessingConfig | None = None,
    df_sample: pd.DataFrame | None = None,
    datetime_cols: list[str] | tuple[str, ...] = (),
    bool_cols: list[str] | tuple[str, ...] = (),
) -> ColumnTransformer:
    """ColumnTransformer 를 구성한다.

    - ``config=None`` (기본): 기존 MVP 동작 — 수치 median→StandardScaler,
      범주 most_frequent→OneHotEncoder(handle_unknown=ignore).
    - ``config`` 지정: 전처리 전략에 따라 파이프라인 구성. 라우팅 결과는
      반환 ColumnTransformer 의 ``_route_report_`` 속성으로 부착된다.

    ``datetime_cols`` / ``bool_cols`` 는 ``config`` 가 주어졌을 때만 의미가 있다.
    ``config.datetime_decompose=False`` 면 datetime 컬럼은 파이프라인에서 제외되고,
    ``config.bool_as_numeric=False`` 면 bool 컬럼은 범주로 편입된다.
    """
    if config is None:
        return _build_preprocessor_default(num_cols, cat_cols)

    num_list = [str(c) for c in num_cols]
    cat_list = [str(c) for c in cat_cols]
    dt_list = [str(c) for c in datetime_cols]
    bool_list = [str(c) for c in bool_cols]

    if not config.bool_as_numeric:
        cat_list = cat_list + [c for c in bool_list if c not in cat_list]
        bool_list = []

    route_report = plan_categorical_routing(df_sample, cat_list, config)

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_list:
        transformers.append(("num", _build_numeric_pipeline(config), num_list))
    if cat_list:
        transformers.extend(_make_cat_transformers(cat_list, route_report, config))
    if dt_list and config.datetime_decompose:
        transformers.append(("datetime", _build_datetime_pipeline(config), dt_list))
    if bool_list:
        transformers.append(("bool", _build_bool_passthrough(), bool_list))

    if not transformers:
        raise ValueError(
            "수치/범주/datetime/bool 컬럼이 모두 비어 있습니다. 입력 피처를 확인하세요."
        )

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct._route_report_ = route_report
    return ct


def _build_preprocessor_default(
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
) -> ColumnTransformer:
    """기존 MVP 동작 (config=None 기본 경로, §3.4 원본)."""
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        transformers.append(("num", num_pipeline, list(num_cols)))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, list(cat_cols)))

    if not transformers:
        raise ValueError("수치/범주 컬럼이 모두 비어 있습니다. 입력 피처를 확인하세요.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _enumerate_derived_features(
    df: pd.DataFrame,
    cat_cols: Iterable[str],
    datetime_cols: Iterable[str],
    bool_cols: Iterable[str],
    config: PreprocessingConfig,
    route_report: PreprocessingRouteReport | None,
) -> list[DerivedFeature]:
    """config + df 카테고리 기반으로 ``DerivedFeature`` 목록을 구축.

    실제 fit 없이 메타데이터만으로 산출 — §9.7 preview 에서도 같은 함수 사용.
    """
    derived: list[DerivedFeature] = []

    cat_list = [str(c) for c in cat_cols]
    dt_list = [str(c) for c in datetime_cols]
    bool_list = [str(c) for c in bool_cols]

    # 범주형
    for col in cat_list:
        final_enc = (
            route_report.encoding_per_col.get(col, config.categorical_encoding)
            if route_report
            else config.categorical_encoding
        )
        if final_enc == "onehot" and col in df.columns:
            values = df[col].dropna().astype(str).unique().tolist()
            for v in sorted(values):
                derived.append(
                    DerivedFeature(
                        name=f"{col}__{v}",
                        source=col,
                        kind="onehot",
                    )
                )
        else:
            derived.append(DerivedFeature(name=col, source=col, kind=final_enc))

    # datetime 분해 (§9.4 의존; 여기서는 파트 이름만 나열)
    if config.datetime_decompose:
        for col in dt_list:
            for part in config.datetime_parts:
                derived.append(
                    DerivedFeature(
                        name=f"{col}_{part}",
                        source=col,
                        kind=f"datetime_{part}",
                    )
                )

    # bool passthrough (bool_as_numeric)
    if config.bool_as_numeric:
        for col in bool_list:
            derived.append(DerivedFeature(name=col, source=col, kind="bool_numeric"))

    return derived


def build_feature_schema(
    df: pd.DataFrame,
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
    target: str,
    *,
    datetime_cols: list[str] | tuple[str, ...] = (),
    bool_cols: list[str] | tuple[str, ...] = (),
    config: PreprocessingConfig | None = None,
    route_report: PreprocessingRouteReport | None = None,
) -> FeatureSchema:
    """학습에 실제 사용된 피처 구조를 스냅샷. 예측 시 입력 검증의 기준.

    - 기본 호출 (config=None, datetime_cols=(), bool_cols=()): 기존 동작과 동일.
    - ``config`` 지정 시 ``derived`` 목록이 encoding/decompose/bool_as_numeric
      규칙에 맞춰 자동 채워진다.
    """
    categories: dict[str, tuple[str, ...]] = {}
    for col in cat_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        categories[str(col)] = tuple(sorted(values))

    # bool_as_numeric=False 면 bool 컬럼은 범주로 포함 — 호출자가 cat_cols 에 병합해서 넘겨야 함.
    # build_feature_schema 는 넘겨받은 그룹을 신뢰한다 (단일 책임).

    derived: list[DerivedFeature] = []
    if config is not None:
        derived = _enumerate_derived_features(
            df,
            cat_cols=cat_cols,
            datetime_cols=datetime_cols,
            bool_cols=bool_cols,
            config=config,
            route_report=route_report,
        )

    return FeatureSchema(
        numeric=tuple(str(c) for c in num_cols),
        categorical=tuple(str(c) for c in cat_cols),
        target=target,
        categories=categories,
        datetime=tuple(str(c) for c in datetime_cols),
        derived=tuple(derived),
    )


def prepare_xy(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame 에서 (X, y) 를 분리한다.

    - ``target_column`` 이 데이터에 없으면 ``ValueError``.
    - ``excluded_columns`` 는 존재하는 컬럼만 제거 (없는 컬럼은 무시).
    """
    if config.target_column not in df.columns:
        raise ValueError(f"target_column({config.target_column})이 데이터에 없습니다.")

    drop_cols = {config.target_column, *config.excluded_columns}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[config.target_column]
    return X, y
