"""전처리 파이프라인 (IMPLEMENTATION_PLAN §3.4, FR-050~053).

원칙:
- 이 모듈은 Streamlit/DB 에 의존하지 않는다.
- ColumnTransformer 는 새 알고리즘 학습마다 ``sklearn.base.clone`` 으로 복제해 사용한다
  (같은 인스턴스를 여러 Pipeline 에 재사용하면 fit 상태가 공유되어 재현성이 깨진다).
- 결측 처리:
    - 수치형: 중앙값 대치
    - 범주형: 최빈값 대치 + OneHot(handle_unknown='ignore')
- 스케일링:
    - 수치형: StandardScaler
- remainder='drop' → 수치/범주 분류에 포함되지 않은 컬럼은 학습에 사용하지 않는다.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.schemas import FeatureSchema, TrainingConfig


def split_feature_types(
    df: pd.DataFrame,
    target: str,
    excluded: tuple[str, ...] | list[str] = (),
) -> tuple[list[str], list[str]]:
    """데이터프레임 컬럼을 수치/범주 두 그룹으로 나눈다. target·excluded 는 제외.

    반환 순서는 DataFrame 의 원본 컬럼 순서를 유지한다.
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


def build_preprocessor(
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
) -> ColumnTransformer:
    """수치/범주 컬럼 각각에 대한 ColumnTransformer 를 구성한다.

    반환된 transformer 는 **fit 되지 않은 상태**다. 학습 pipeline 에 결합된 뒤 fit 된다.
    """
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


def build_feature_schema(
    df: pd.DataFrame,
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
    target: str,
) -> FeatureSchema:
    """학습에 실제 사용된 피처 구조를 스냅샷. 예측 시 입력 검증의 기준이 된다."""
    categories: dict[str, tuple[str, ...]] = {}
    for col in cat_cols:
        values = df[col].dropna().astype(str).unique().tolist()
        categories[col] = tuple(sorted(values))

    return FeatureSchema(
        numeric=tuple(num_cols),
        categorical=tuple(cat_cols),
        target=target,
        categories=categories,
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
