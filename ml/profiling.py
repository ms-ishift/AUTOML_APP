"""데이터 프로파일링 (IMPLEMENTATION_PLAN §3.3, FR-033).

Streamlit/DB 의존 없이 순수 pandas 기반 연산만 사용한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml.schemas import ColumnProfile, DatasetProfile

if TYPE_CHECKING:
    import pandas as pd


# 식별자 의심 컬럼 판정 기준 (Service 계층에서도 동일 임계를 노출 가능하도록 public)
ID_UNIQUE_RATIO_THRESHOLD = 0.95
_ID_UNIQUE_RATIO_THRESHOLD = ID_UNIQUE_RATIO_THRESHOLD  # 하위 호환용 내부 alias


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """pandas DataFrame 을 ``DatasetProfile`` 로 요약한다.

    - ``dtype`` 은 pandas dtype 문자열을 그대로 쓴다 (예: ``int64``, ``object``, ``float64``).
    - ``missing_ratio`` / ``unique_ratio`` 는 [0, 1] 정규화 값.
    - 행이 0개일 경우 비율은 0.0 으로 처리.
    """
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    denom = n_rows if n_rows > 0 else 1

    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                n_missing=n_missing,
                n_unique=n_unique,
                missing_ratio=round(n_missing / denom, 6),
                unique_ratio=round(n_unique / denom, 6),
            )
        )

    return DatasetProfile(n_rows=n_rows, n_cols=n_cols, columns=tuple(columns))


def suggest_excluded(
    profile: DatasetProfile,
    *,
    unique_ratio_threshold: float = _ID_UNIQUE_RATIO_THRESHOLD,
) -> list[str]:
    """식별자 의심 컬럼(고유값 비율이 임계 이상)을 반환.

    비율 기준이므로 행이 1건이거나 0건인 경우에는 항상 빈 리스트를 반환한다.
    """
    if profile.n_rows < 2:
        return []
    return [c.name for c in profile.columns if c.unique_ratio >= unique_ratio_threshold]
