"""프로젝트 관리 페이지 (IMPLEMENTATION_PLAN §6.1, FR-020~024).

책임:
- 생성/목록/선택/수정/삭제 UX 를 ``project_service`` 호출로 오케스트레이션
- 선택 시 ``SessionKey.CURRENT_PROJECT_ID`` 를 갱신하여 사이드바에 즉시 반영
- 삭제는 cascade 옵션을 명시적으로 확인 받는 2-step 플로우

UX 결정:
- ``st.dialog`` 대신 **세션 상태 기반 인라인 컨테이너**로 수정/삭제 확인을 구현.
  이유: ``streamlit.testing.v1.AppTest`` 환경에서 dialog 내부 위젯 조작이 버전별로 불안정한데 반해,
  세션 플래그 + 인라인 렌더는 버튼 클릭 하나로 같은 UX 를 제공하고 테스트가 견고하다.
- 목록은 ``st.dataframe`` 대신 row per-container 로 그려 ``[선택][수정][삭제]`` 3개 버튼을 같은 행에 배치.

규약 (``.cursor/rules/streamlit-ui.mdc``):
- 한글 리터럴은 ``utils.messages.Msg`` 에서 가져온다. Service 예외는 ``AppError`` 로 잡아 ``flash`` 로 전달.
- 세션 상태 접근은 ``SessionKey`` 상수 경유.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

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


PAGE_TITLE: Final[str] = "프로젝트 관리"
PAGE_CAPTION: Final[str] = "프로젝트를 만들고 전환하며 관리합니다."

# 페이지 전용 세션 키 (위젯 인스턴스/플로우 상태). 공식 상수 밖이지만 로컬 위젯 key 는 자유.
EDIT_TARGET_KEY: Final[str] = "projects_edit_target_id"
DELETE_TARGET_KEY: Final[str] = "projects_delete_target_id"


# --------------------------------------------------------------- data helpers


def _load_projects() -> list[ProjectDTO]:
    try:
        return project_service.list_projects()
    except AppError as err:
        flash("error", str(err))
        return []


def _load_current_project(db_ready: bool) -> ProjectDTO | None:
    if not db_ready:
        return None
    pid = get_state(SessionKey.CURRENT_PROJECT_ID)
    if pid is None:
        return None
    try:
        return project_service.get_project(int(pid))
    except NotFoundError:
        st.session_state.pop(SessionKey.CURRENT_PROJECT_ID, None)
        return None
    except AppError:
        return None


# ------------------------------------------------------------------ actions


def _select_project(project: ProjectDTO) -> None:
    set_state(SessionKey.CURRENT_PROJECT_ID, project.id)
    flash("success", f"'{project.name}' 프로젝트를 선택했습니다.")


def _enter_edit_mode(project: ProjectDTO) -> None:
    st.session_state[EDIT_TARGET_KEY] = project.id
    st.session_state.pop(DELETE_TARGET_KEY, None)


def _enter_delete_mode(project: ProjectDTO) -> None:
    st.session_state[DELETE_TARGET_KEY] = project.id
    st.session_state.pop(EDIT_TARGET_KEY, None)


def _cancel_flow() -> None:
    st.session_state.pop(EDIT_TARGET_KEY, None)
    st.session_state.pop(DELETE_TARGET_KEY, None)


# ------------------------------------------------------------------ renders


def _render_create_form() -> None:
    with st.expander("새 프로젝트 만들기", expanded=False), st.form("create_project_form"):
        name = st.text_input("프로젝트명", max_chars=100, key="create_name")
        description = st.text_area("설명 (선택)", max_chars=500, height=80, key="create_desc")
        submitted = st.form_submit_button("생성", type="primary")
        if not submitted:
            return
        try:
            project = project_service.create_project(name, description or None)
        except AppError as err:
            flash("error", str(err))
            st.rerun()
        else:
            flash("success", Msg.PROJECT_CREATED)
            _select_project(project)
            st.rerun()


def _render_project_row(project: ProjectDTO, *, is_current: bool) -> None:
    with st.container(border=True):
        info_col, meta_col, action_col = st.columns([5, 3, 4])
        with info_col:
            prefix = "★ " if is_current else ""
            st.markdown(f"**{prefix}{project.name}**")
            if project.description:
                st.caption(project.description)
            else:
                st.caption("_설명 없음_")
        with meta_col:
            st.caption(f"데이터셋 {project.dataset_count} · 모델 {project.model_count}")
            st.caption(f"업데이트 {project.updated_at:%Y-%m-%d %H:%M}")
            st.caption(f"id={project.id}")
        with action_col:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "선택" if not is_current else "선택됨",
                    key=f"project_select_{project.id}",
                    width="stretch",
                    disabled=is_current,
                ):
                    _select_project(project)
                    st.rerun()
            with b2:
                if st.button(
                    "수정",
                    key=f"project_edit_{project.id}",
                    width="stretch",
                ):
                    _enter_edit_mode(project)
                    st.rerun()
            with b3:
                if st.button(
                    "삭제",
                    key=f"project_delete_{project.id}",
                    width="stretch",
                ):
                    _enter_delete_mode(project)
                    st.rerun()


def _render_list(projects: list[ProjectDTO], *, current_id: int | None) -> None:
    st.subheader(f"프로젝트 목록 ({len(projects)})")
    if not projects:
        st.info("아직 생성된 프로젝트가 없습니다. 위 폼으로 첫 프로젝트를 만들어 보세요.")
        return
    for project in projects:
        _render_project_row(project, is_current=project.id == current_id)


def _render_edit_flow(projects: list[ProjectDTO]) -> None:
    target_id = st.session_state.get(EDIT_TARGET_KEY)
    if target_id is None:
        return
    target = next((p for p in projects if p.id == int(target_id)), None)
    if target is None:
        st.session_state.pop(EDIT_TARGET_KEY, None)
        return

    with st.container(border=True):
        st.markdown(f"#### 프로젝트 수정 — `{target.name}` (id={target.id})")
        with st.form("edit_project_form"):
            name = st.text_input("프로젝트명", value=target.name, max_chars=100, key="edit_name")
            description = st.text_area(
                "설명",
                value=target.description or "",
                max_chars=500,
                height=80,
                key="edit_desc",
            )
            c1, c2 = st.columns(2)
            submitted = c1.form_submit_button("저장", type="primary")
            canceled = c2.form_submit_button("취소")

        if canceled:
            _cancel_flow()
            st.rerun()
        if submitted:
            try:
                project_service.update_project(
                    target.id,
                    name=name,
                    description=description,
                )
            except AppError as err:
                flash("error", str(err))
                st.rerun()
            else:
                flash("success", Msg.PROJECT_UPDATED)
                _cancel_flow()
                st.rerun()


def _render_delete_flow(projects: list[ProjectDTO]) -> None:
    target_id = st.session_state.get(DELETE_TARGET_KEY)
    if target_id is None:
        return
    target = next((p for p in projects if p.id == int(target_id)), None)
    if target is None:
        st.session_state.pop(DELETE_TARGET_KEY, None)
        return

    has_children = bool(target.dataset_count or target.model_count)
    with st.container(border=True):
        st.markdown(f"#### 프로젝트 삭제 확인 — `{target.name}` (id={target.id})")
        st.caption(Msg.DELETE_CONFIRM)
        if has_children:
            st.warning(
                f"연결된 리소스가 있습니다: 데이터셋 {target.dataset_count}, "
                f"저장 모델 {target.model_count}."
            )
        cascade = st.checkbox(
            "연결된 데이터셋·학습·모델까지 함께 삭제",
            value=has_children,
            key="delete_cascade_flag",
            help="체크를 해제하고 연결된 리소스가 있으면 삭제가 거부됩니다.",
        )
        c1, c2 = st.columns(2)
        do_delete = c1.button("삭제 실행", type="primary", key="delete_confirm_btn")
        do_cancel = c2.button("취소", key="delete_cancel_btn")

        if do_cancel:
            _cancel_flow()
            st.rerun()
        if do_delete:
            try:
                project_service.delete_project(target.id, cascade=cascade)
            except AppError as err:
                # 확인 블록은 유지 → 사용자가 cascade 체크 후 바로 재시도 가능
                flash("error", str(err))
                st.rerun()
            else:
                flash("success", Msg.PROJECT_DELETED)
                # 현재 선택된 프로젝트가 삭제 대상이면 세션에서도 해제.
                if get_state(SessionKey.CURRENT_PROJECT_ID) == target.id:
                    st.session_state.pop(SessionKey.CURRENT_PROJECT_ID, None)
                _cancel_flow()
                st.rerun()


# ------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current = _load_current_project(db_ready)
    render_sidebar(current_project=current, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return

    _render_create_form()

    projects = _load_projects()
    _render_edit_flow(projects)
    _render_delete_flow(projects)
    _render_list(
        projects,
        current_id=current.id if current is not None else None,
    )


if __name__ == "__main__":
    main()
