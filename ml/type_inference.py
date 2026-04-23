"""타입 추론 강화 유틸 (IMPLEMENTATION_PLAN §9.2, FR-056).

전처리 UI/파이프라인에서 datetime · bool · 고카디널리티 범주형 컬럼을 자동
감지하기 위한 **순수 pandas** 유틸리티. Streamlit / SQLAlchemy / sklearn 의존 없음.

함수 계약 요약:
- ``detect_datetime_columns(df)`` — datetime64 타입 + object 컬럼 중 파싱 성공률
  ≥ :data:`DATETIME_PARSE_SUCCESS_RATIO` 인 컬럼.
- ``detect_bool_columns(df)`` — bool 타입 + int{0,1} + object bool 토큰 집합.
- ``detect_highcard_categorical(df, cols, ...)`` — ``nunique`` 또는
  ``nunique / n_rows`` 임계 초과.
- ``skew_report(df, num_cols, ...)`` — ``|skew| >= threshold`` 인 수치 컬럼 매핑
  (L3 후속 작업용 미리 배치, 기본 off — 호출 측에서 원할 때만 사용).

모든 함수는 **존재하지 않는 컬럼·빈 시리즈·모두 NaN** 케이스에서 예외를 던지지
않고 조용히 스킵한다. 행이 0건인 DataFrame 은 전부 빈 결과.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------- 상수
# object 컬럼을 datetime 으로 승격하기 위한 최소 파싱 성공률.
# 너무 낮추면 수치 문자열까지 오인 가능 → 0.95 로 고정.
DATETIME_PARSE_SUCCESS_RATIO: float = 0.95

# object 컬럼을 bool 로 승격하기 위한 토큰 허용 집합 (소문자 비교).
# 플랜 기본은 {"True","False",0,1,"Y","N"} 이며 실무 빈도를 고려해 약간 확장.
_BOOL_TOKENS: frozenset[str] = frozenset(
    {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}
)

# object 컬럼이 실제로는 수치 문자열("1","42","3.14") 일 때 datetime 오인 방지.
# 수치 coerce 성공률이 이 값 이상이면 datetime 후보에서 제외한다.
_NUMERIC_STRING_EXCLUSION_RATIO: float = 0.8


__all__ = [
    "DATETIME_PARSE_SUCCESS_RATIO",
    "detect_bool_columns",
    "detect_datetime_columns",
    "detect_highcard_categorical",
    "skew_report",
]


# ---------------------------------------------------------------- datetime


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """datetime 후보 컬럼을 반환한다.

    규칙:
    - ``is_datetime64_any_dtype`` True 면 무조건 포함.
    - ``object`` 컬럼은 ``pd.to_datetime(errors="coerce")`` 비결측 성공률이
      :data:`DATETIME_PARSE_SUCCESS_RATIO` 이상이면 포함.
    - 수치 문자열("1","42","3.14") 위주의 object 컬럼은 제외한다 —
      ``to_datetime`` 이 unit 해석으로 오인할 여지가 있음.
    - 모두 NaN 이거나 non-null 이 0 인 컬럼은 제외.

    반환 리스트의 순서는 ``df.columns`` 순서를 그대로 따른다.
    """
    if df.shape[0] == 0:
        # 행이 없으면 dtype 만 보고 판단 (is_datetime64_any_dtype 는 행 수 무관).
        return [str(c) for c in df.columns if is_datetime64_any_dtype(df[c])]

    out: list[str] = []
    for col in df.columns:
        s = df[col]
        if is_datetime64_any_dtype(s):
            out.append(str(col))
            continue
        if s.dtype != object:
            continue
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        # 수치 문자열 배제 (to_datetime 오인 방지)
        as_num = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = float(as_num.notna().mean())
        if numeric_ratio >= _NUMERIC_STRING_EXCLUSION_RATIO:
            continue
        # pandas 의 포맷 추정 경고는 감지 용도에서는 무시 — dateutil fallback 도 충분.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(non_null, errors="coerce")
        ratio = float(parsed.notna().sum()) / len(non_null)
        if ratio >= DATETIME_PARSE_SUCCESS_RATIO:
            out.append(str(col))
    return out


# ---------------------------------------------------------------- bool


def _is_bool_token_series(non_null: pd.Series) -> bool:
    """object series 의 non-null 고유값이 ``_BOOL_TOKENS`` 부분집합인지 검사."""
    try:
        uniques: set[str] = {str(v).strip().lower() for v in non_null.unique()}
    except Exception:
        return False
    if not uniques:
        return False
    return uniques.issubset(_BOOL_TOKENS)


def detect_bool_columns(df: pd.DataFrame) -> list[str]:
    """bool 후보 컬럼을 반환한다.

    규칙:
    - ``is_bool_dtype`` True → 포함.
    - 정수 컬럼 중 non-null 고유값이 ``{0, 1}`` 부분집합 → 포함.
    - object 컬럼 중 non-null 고유값이 ``_BOOL_TOKENS`` 부분집합 → 포함
      ("Y"/"N", "yes"/"no", "true"/"false", "0"/"1" 등을 허용).
    - 전부 NaN 이거나 non-null 이 0 인 컬럼은 제외.

    반환 리스트의 순서는 ``df.columns`` 순서를 그대로 따른다.
    """
    out: list[str] = []
    for col in df.columns:
        s = df[col]
        if is_bool_dtype(s):
            out.append(str(col))
            continue
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        # int 0/1
        if is_integer_dtype(non_null):
            uniques = {int(v) for v in non_null.unique()}
            if uniques and uniques.issubset({0, 1}):
                out.append(str(col))
            continue
        # object 토큰
        if s.dtype == object and _is_bool_token_series(non_null):
            out.append(str(col))
    return out


# ---------------------------------------------------------------- highcard


def detect_highcard_categorical(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    nunique_threshold: int = 50,
    unique_ratio_threshold: float = 0.3,
) -> list[str]:
    """주어진 ``cols`` 중 고카디널리티로 간주할 컬럼을 반환.

    판정 축 (둘 중 하나라도 초과하면 고카디널리티):
    - ``nunique`` > ``nunique_threshold``
    - (행 수 ≥ 2 일 때) ``nunique / n_rows`` > ``unique_ratio_threshold``

    존재하지 않는 컬럼은 조용히 스킵. 결과는 입력 ``cols`` 의 순회 순서를 유지한다.
    """
    n = int(df.shape[0])
    out: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        nu = int(s.nunique(dropna=True))
        if nu > nunique_threshold:
            out.append(str(col))
            continue
        if n >= 2 and (nu / n) > unique_ratio_threshold:
            out.append(str(col))
    return out


# ---------------------------------------------------------------- skew


def skew_report(
    df: pd.DataFrame,
    num_cols: Iterable[str],
    *,
    abs_skew_threshold: float = 1.0,
) -> dict[str, float]:
    """``|skew| >= abs_skew_threshold`` 인 수치 컬럼 → skew 값 매핑.

    L3(로그/Yeo-Johnson) 후속 작업용 **미리 배치 유틸**. 기본 off — UI 는
    사용자가 명시적으로 조회했을 때만 호출한다.

    - 수치가 아닌 컬럼 / 존재하지 않는 컬럼 / ``skew`` 가 비유한(NaN/Inf) 인
      컬럼 (상수 또는 거의 상수)은 조용히 스킵한다.
    - 반환 dict 의 순서는 입력 ``num_cols`` 순회 순서를 유지한다 (Python 3.7+ dict).
    - skew 값은 소수 6자리로 반올림.
    """
    out: dict[str, float] = {}
    for col in num_cols:
        if col not in df.columns:
            continue
        s = df[col]
        if not is_numeric_dtype(s):
            continue
        skew_val: Any
        try:
            skew_val = s.skew()
        except Exception:
            continue
        try:
            sk = float(skew_val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sk):
            continue
        if abs(sk) >= abs_skew_threshold:
            out[str(col)] = round(sk, 6)
    return out
