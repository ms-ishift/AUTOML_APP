"""공용 레이아웃 컴포넌트 (IMPLEMENTATION_PLAN §5.1, FR-001~003).

- ``configure_page``: ``st.set_page_config`` 을 페이지별로 **반드시 가장 먼저 1회** 호출하도록 표준화.
- ``render_sidebar``: 현재 프로젝트 표시 + 페이지 네비 안내 + DB 상태 뱃지.
- ``render_page_header``: 페이지 상단 title + caption + flash 렌더의 일괄 처리.

규약 (``.cursor/rules/streamlit-ui.mdc``):
- 세션 상태 키는 ``SessionKey`` 상수로만 접근.
- Service 호출을 layout 레이어에서 직접 하지 않는다 — 프로젝트 조회는 **호출 측이 준비해서 주입**.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Literal

import streamlit as st

from pages.components.toast import render_flashes
from utils.session_utils import SessionKey, clear_state, get_state

if TYPE_CHECKING:
    from services.dto import ProjectDTO


NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("홈", "app.py"),
    ("프로젝트", "pages/01_projects.py"),
    ("데이터 업로드", "pages/02_dataset_upload.py"),
    ("학습", "pages/03_training.py"),
    ("결과 비교", "pages/04_results.py"),
    ("모델 관리", "pages/05_models.py"),
    ("예측", "pages/06_prediction.py"),
    ("이력/관리자", "pages/07_admin.py"),
)


def configure_page(title: str, *, layout: Literal["centered", "wide"] = "wide") -> None:
    """``st.set_page_config`` 래퍼. 페이지 title 과 기본 레이아웃을 강제한다.

    Streamlit 규약상 페이지당 **첫 Streamlit 호출**이어야 한다 — 반드시 각 page 파일의 최상단에서 호출.
    ``StreamlitAPIException`` (이미 설정된 경우) 은 조용히 무시해서 재실행 루프에 견딘다.
    """
    with suppress(st.errors.StreamlitAPIException):
        st.set_page_config(
            page_title=f"{title} · AutoML MVP",
            layout=layout,
            initial_sidebar_state="expanded",
        )


def render_sidebar(
    *,
    current_project: ProjectDTO | None,
    db_ready: bool,
) -> None:
    """사이드바 렌더링. 현재 프로젝트 표시 + 페이지 네비 + DB 상태.

    - ``current_project`` 는 호출자가 Service 경유로 조회해 주입.
    - ``db_ready`` 가 False 면 빨간 뱃지로 초기화 필요를 안내.
    """
    with st.sidebar:
        st.markdown("### AutoML MVP")
        st.caption("정형 데이터 AutoML 파이프라인")

        if not db_ready:
            st.error("DB가 아직 초기화되지 않았습니다.")
            st.caption(
                "터미널에서 `python scripts/init_db.py --seed` 를 실행한 뒤 페이지를 새로고침하세요."
            )
            st.divider()

        st.markdown("#### 현재 프로젝트")
        if current_project is None:
            st.info("선택된 프로젝트가 없습니다.")
            st.caption("프로젝트 메뉴에서 하나를 선택해 주세요.")
        else:
            st.success(f"**{current_project.name}** (id={current_project.id})")
            if current_project.description:
                st.caption(current_project.description)
            meta = (
                f"데이터셋 {current_project.dataset_count}건 · "
                f"저장 모델 {current_project.model_count}건"
            )
            st.caption(meta)
            if st.button("선택 해제", key="sidebar_clear_project", width="stretch"):
                clear_state(
                    SessionKey.CURRENT_PROJECT_ID,
                    SessionKey.CURRENT_DATASET_ID,
                    SessionKey.LAST_TRAINING_JOB_ID,
                )
                st.rerun()

        st.divider()
        st.markdown("#### 페이지")
        for label, _ in NAV_ITEMS:
            st.markdown(f"- {label}")
        st.caption("왼쪽 네비게이션에서 페이지를 선택하세요.")


def render_page_header(
    title: str,
    *,
    caption: str | None = None,
    show_flashes: bool = True,
) -> None:
    """페이지 상단 헤더. flash 메시지 렌더까지 한번에 처리."""
    st.title(title)
    if caption:
        st.caption(caption)
    if show_flashes:
        render_flashes()


def current_project_id() -> int | None:
    """세션에 저장된 현재 프로젝트 id (없으면 ``None``)."""
    pid = get_state(SessionKey.CURRENT_PROJECT_ID)
    return int(pid) if pid is not None else None


__all__ = [
    "NAV_ITEMS",
    "configure_page",
    "current_project_id",
    "render_page_header",
    "render_sidebar",
]
