"""데이터 미리보기/프로파일 렌더러 (IMPLEMENTATION_PLAN §5.2, FR-031~033).

책임:
- ``dataset_service.preview_dataset`` 결과(``list[dict]``)를 ``st.dataframe`` 으로 그린다.
- ``DatasetProfileDTO`` 를 컬럼 프로파일 테이블 + 요약 메트릭으로 렌더.

설계 메모:
- Service 에서 JSON 직렬화까지 마친 데이터만 받는다 (pd.DataFrame 이나 ORM 에 의존 금지).
- 이 컴포넌트는 데이터 소스를 모른다 — ``pages/02_dataset_upload.py`` 뿐 아니라
  §6.3 학습 페이지의 "데이터 확인" 스텝에서도 그대로 재사용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from services.dto import DatasetProfileDTO


def render_preview(
    rows: list[dict[str, Any]],
    *,
    caption: str | None = None,
    height: int | None = 260,
) -> None:
    """``preview_dataset`` 결과 레코드를 테이블로 렌더링.

    빈 리스트면 안내 문구를 표시. 컬럼 순서는 입력 순서를 유지한다.
    """
    if not rows:
        st.info("표시할 데이터가 없습니다.")
        return
    df = pd.DataFrame(rows)
    _render_df(df, height=height)
    if caption:
        st.caption(caption)


def _profile_rows(profile: DatasetProfileDTO) -> list[dict[str, Any]]:
    """``DatasetProfileDTO`` → 테이블 친화 레코드."""
    return [
        {
            "컬럼": c.name,
            "타입": c.dtype,
            "결측": c.n_missing,
            "결측비율": round(c.missing_ratio, 4),
            "고유값": c.n_unique,
            "고유비율": round(c.unique_ratio, 4),
        }
        for c in profile.columns
    ]


def render_profile(profile: DatasetProfileDTO, *, height: int | None = 320) -> None:
    """컬럼 프로파일 테이블 + 행/컬럼 수 요약 메트릭.

    레이아웃:
    - 상단: 2-컬럼 메트릭(행/컬럼 수)
    - 하단: 컬럼별 통계 테이블 (결측/고유 수 및 비율)
    """
    c1, c2 = st.columns(2)
    c1.metric("행 수", f"{profile.rows:,}")
    c2.metric("컬럼 수", profile.cols)
    records = _profile_rows(profile)
    if not records:
        st.info("프로파일 정보가 비어 있습니다.")
        return
    df = pd.DataFrame(records)
    _render_df(df, height=height)


def _render_df(df: pd.DataFrame, *, height: int | None) -> None:
    """``st.dataframe`` 호출 래퍼 — ``height=None`` 일 때 인자에서 제외해 mypy 오버로드를 만족시킨다."""
    if height is None:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, width="stretch", height=height, hide_index=True)


__all__ = ["render_preview", "render_profile"]
