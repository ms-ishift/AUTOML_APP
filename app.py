"""AutoML Streamlit MVP — 앱 진입점 + 홈 화면 (IMPLEMENTATION_PLAN §5.1).

페이지 네비게이션은 Streamlit 의 자동 멀티페이지(`pages/` 디렉터리) 규약을 따른다.
이 파일은 다음을 수행한다:

1. ``st.set_page_config`` 을 가장 먼저 호출 (공용 ``configure_page`` 경유)
2. DB 초기화 상태 점검 + 사용자 가이드
3. 홈 본문: 서비스 소개 / 최근 프로젝트 3개 / "프로젝트로 이동" CTA
4. 사이드바: 현재 프로젝트 · 페이지 네비 · DB 뱃지

규약 (``.cursor/rules/streamlit-ui.mdc``):
- 페이지/서비스는 ``Msg`` 상수와 세션 ``SessionKey`` 만 사용.
- Service 외의 비즈니스 로직을 UI 에서 수행하지 않는다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import ProjectDTO


HOME_TITLE = "AutoML 홈"
HOME_CAPTION = "정형 데이터 업로드 → AutoML 학습 → 모델 비교 → 예측까지 한 흐름."
RECENT_PROJECTS_LIMIT = 3


def _load_current_project(db_ready: bool) -> ProjectDTO | None:
    """현재 선택된 프로젝트를 Service 경유로 조회. 삭제된 경우 세션에서도 제거."""
    if not db_ready:
        return None
    project_id = get_state(SessionKey.CURRENT_PROJECT_ID)
    if project_id is None:
        return None
    try:
        return project_service.get_project(int(project_id))
    except NotFoundError:
        # 세션에 남은 stale id → 조용히 정리 후 홈에서 선택 유도
        st.session_state.pop(SessionKey.CURRENT_PROJECT_ID, None)
        flash("warning", "선택된 프로젝트가 더 이상 존재하지 않아 해제했습니다.")
        return None
    except AppError:
        return None


def _render_intro() -> None:
    st.markdown(
        """
        ### 이 앱은 무엇을 하나요?

        - **데이터 업로드**: CSV/XLSX 를 올리면 프로파일과 미리보기를 자동 생성합니다.
        - **AutoML 학습**: 타깃과 지표만 고르면 여러 알고리즘을 동시에 학습·비교합니다.
        - **모델 저장/예측**: 가장 마음에 드는 모델을 저장하고, 단건 폼 또는 파일로 예측할 수 있습니다.

        학습과 예측의 실행은 모두 서비스 계층에서 오케스트레이션되며, 이 화면에서는 결과만 보여줍니다.
        """
    )


def _render_recent_projects(projects: list[ProjectDTO]) -> None:
    """최근 프로젝트 카드 N개 + "프로젝트 페이지로 이동" CTA."""
    st.subheader("최근 프로젝트")
    if not projects:
        st.info(Msg.PROJECT_REQUIRED)
        st.caption("아래 버튼으로 새 프로젝트를 만들어 시작할 수 있습니다.")
        return

    cols = st.columns(min(len(projects), RECENT_PROJECTS_LIMIT))
    for col, project in zip(cols, projects[:RECENT_PROJECTS_LIMIT], strict=False):
        with col, st.container(border=True):
            st.markdown(f"**{project.name}**")
            if project.description:
                st.caption(project.description)
            st.caption(f"데이터셋 {project.dataset_count} · 모델 {project.model_count}")
            st.caption(f"업데이트: {project.updated_at:%Y-%m-%d %H:%M}")
            if st.button(
                "선택하기",
                key=f"home_select_project_{project.id}",
                width="stretch",
            ):
                set_state(SessionKey.CURRENT_PROJECT_ID, project.id)
                flash("success", f"'{project.name}' 프로젝트를 선택했습니다.")
                st.rerun()


def _render_cta(db_ready: bool) -> None:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        disabled = not db_ready
        if st.button(
            "프로젝트 페이지로 이동",
            type="primary",
            width="stretch",
            disabled=disabled,
            help=None if db_ready else "DB 초기화가 필요합니다.",
        ):
            # pages/01_projects.py 는 §6.1 에서 구현 예정. 현재는 안내만.
            flash(
                "info",
                "프로젝트 페이지는 단계 6.1 에서 추가됩니다. "
                "지금은 사이드바 네비게이션이 비어 있을 수 있습니다.",
            )
            st.rerun()
    with c2:
        if st.button("문서 보기", width="stretch"):
            st.toast("AutoML_Streamlit_MVP.md / IMPLEMENTATION_PLAN.md 를 참고하세요.")


def _load_recent_projects(db_ready: bool) -> list[ProjectDTO]:
    if not db_ready:
        return []
    try:
        return project_service.list_projects()[:RECENT_PROJECTS_LIMIT]
    except AppError:
        return []


def main() -> None:
    configure_page(HOME_TITLE)

    db_ready = is_db_initialized()
    current = _load_current_project(db_ready)

    render_sidebar(current_project=current, db_ready=db_ready)
    render_page_header(HOME_TITLE, caption=HOME_CAPTION)

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        st.caption("초기화 후 페이지를 새로고침하세요.")
        return

    _render_intro()
    st.divider()
    _render_recent_projects(_load_recent_projects(db_ready))
    _render_cta(db_ready)


if __name__ == "__main__":
    main()
