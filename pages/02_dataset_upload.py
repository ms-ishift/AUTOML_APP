"""데이터 업로드 페이지 (IMPLEMENTATION_PLAN §6.2, FR-030~035).

책임:
- 현재 선택된 프로젝트에 CSV/XLSX 를 업로드하고, 성공 시 자동으로 미리보기 + 프로파일을 표시한다.
- 기존 업로드 이력을 최신순으로 나열하고 삭제 버튼을 제공 (cascade 는 Service 에서 보장).
- 업로드/삭제 결과는 flash 로 전달, 화면 선택 상태는 ``SessionKey.CURRENT_DATASET_ID`` 에 반영.

UX 결정:
- Streamlit ``st.file_uploader`` 는 매 rerun 마다 같은 객체를 반환하므로 폼 내부에서 ``clear_on_submit=True``
  를 사용해 다중 업로드 시 중복 submit 을 방지.
- 삭제 확인은 프로젝트 페이지와 동일하게 세션 플래그 + 인라인 컨펌 블록 패턴 (AppTest 친화).
- 목록 클릭(선택)은 "미리보기 대상 지정" 개념. `SessionKey.CURRENT_DATASET_ID` 로 §6.3 학습 페이지와 연동.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from pages.components.data_preview import render_preview, render_profile
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import dataset_service, project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.file_utils import ALLOWED_EXTENSIONS
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import DatasetDTO, DatasetProfileDTO, ProjectDTO


PAGE_TITLE: Final[str] = "데이터 업로드"
PAGE_CAPTION: Final[str] = "CSV/XLSX 파일을 업로드하고 컬럼 프로파일을 확인합니다."

UPLOAD_FORM_KEY: Final[str] = "dataset_upload_form"
UPLOADER_KEY: Final[str] = "dataset_file_uploader"
DELETE_TARGET_KEY: Final[str] = "datasets_delete_target_id"
PREVIEW_ROWS: Final[int] = 50


# ------------------------------------------------------------------ loaders


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


def _load_datasets(project_id: int) -> list[DatasetDTO]:
    try:
        return dataset_service.list_datasets(project_id)
    except AppError as err:
        flash("error", str(err))
        return []


def _load_profile(dataset_id: int) -> DatasetProfileDTO | None:
    try:
        return dataset_service.get_dataset_profile(dataset_id)
    except AppError as err:
        flash("error", str(err))
        return None


def _load_preview(dataset_id: int) -> list[dict]:
    try:
        return dataset_service.preview_dataset(dataset_id, n=PREVIEW_ROWS)
    except AppError as err:
        flash("error", str(err))
        return []


# ----------------------------------------------------------------- renders


def _render_project_required_guard() -> None:
    st.warning(Msg.PROJECT_REQUIRED)
    st.caption("홈 또는 프로젝트 페이지에서 먼저 프로젝트를 선택해 주세요.")


def _render_upload_form(project: ProjectDTO) -> None:
    st.subheader(f"새 파일 업로드 — `{project.name}`")
    with st.form(UPLOAD_FORM_KEY, clear_on_submit=True):
        uploaded = st.file_uploader(
            "CSV / XLSX 파일을 선택하세요.",
            type=list(ALLOWED_EXTENSIONS),
            accept_multiple_files=False,
            key=UPLOADER_KEY,
        )
        submitted = st.form_submit_button("업로드", type="primary")

    if not submitted:
        return
    if uploaded is None:
        flash("warning", "업로드할 파일을 선택해 주세요.")
        st.rerun()
        return

    try:
        dataset = dataset_service.upload_dataset(project.id, uploaded)
    except AppError as err:
        flash("error", str(err))
        st.rerun()
    else:
        flash("success", Msg.UPLOAD_SUCCESS)
        set_state(SessionKey.CURRENT_DATASET_ID, dataset.id)
        st.rerun()


def _render_preview_section(dataset_id: int, datasets: list[DatasetDTO]) -> None:
    target = next((d for d in datasets if d.id == dataset_id), None)
    if target is None:
        # stale id → 세션 정리 후 조용히 반환
        st.session_state.pop(SessionKey.CURRENT_DATASET_ID, None)
        return

    st.subheader(f"미리보기 — `{target.file_name}` (id={target.id})")
    st.caption(
        f"행 {target.row_count:,} · 컬럼 {target.column_count} · "
        f"업로드 {target.created_at:%Y-%m-%d %H:%M}"
    )

    tab_preview, tab_profile = st.tabs(["샘플 데이터", "컬럼 프로파일"])
    with tab_preview:
        rows = _load_preview(dataset_id)
        render_preview(
            rows,
            caption=f"상위 {len(rows)}행 (최대 {PREVIEW_ROWS}행)",
        )
    with tab_profile:
        profile = _load_profile(dataset_id)
        if profile is not None:
            render_profile(profile)


def _render_dataset_row(
    dataset: DatasetDTO,
    *,
    is_current: bool,
) -> None:
    with st.container(border=True):
        info_col, meta_col, action_col = st.columns([5, 3, 4])
        with info_col:
            prefix = "★ " if is_current else ""
            st.markdown(f"**{prefix}{dataset.file_name}**")
            st.caption(f"id={dataset.id}")
        with meta_col:
            st.caption(f"행 {dataset.row_count:,} · 컬럼 {dataset.column_count}")
            st.caption(f"업로드 {dataset.created_at:%Y-%m-%d %H:%M}")
        with action_col:
            b1, b2 = st.columns(2)
            with b1:
                if st.button(
                    "선택" if not is_current else "선택됨",
                    key=f"dataset_select_{dataset.id}",
                    width="stretch",
                    disabled=is_current,
                ):
                    set_state(SessionKey.CURRENT_DATASET_ID, dataset.id)
                    flash("success", f"'{dataset.file_name}' 미리보기로 전환했습니다.")
                    st.rerun()
            with b2:
                if st.button(
                    "삭제",
                    key=f"dataset_delete_{dataset.id}",
                    width="stretch",
                ):
                    st.session_state[DELETE_TARGET_KEY] = dataset.id
                    st.rerun()


def _render_list(datasets: list[DatasetDTO], *, current_id: int | None) -> None:
    st.subheader(f"업로드된 데이터셋 ({len(datasets)})")
    if not datasets:
        st.info(Msg.DATASET_REQUIRED)
        return
    for dataset in datasets:
        _render_dataset_row(dataset, is_current=dataset.id == current_id)


def _render_delete_flow(datasets: list[DatasetDTO]) -> None:
    target_id = st.session_state.get(DELETE_TARGET_KEY)
    if target_id is None:
        return
    target = next((d for d in datasets if d.id == int(target_id)), None)
    if target is None:
        st.session_state.pop(DELETE_TARGET_KEY, None)
        return

    with st.container(border=True):
        st.markdown(f"#### 데이터셋 삭제 확인 — `{target.file_name}` (id={target.id})")
        st.caption(Msg.DELETE_CONFIRM)
        st.warning("이 데이터셋에 연결된 학습 이력과 모델이 있으면 함께 삭제됩니다.")
        c1, c2 = st.columns(2)
        do_delete = c1.button("삭제 실행", type="primary", key="dataset_delete_confirm_btn")
        do_cancel = c2.button("취소", key="dataset_delete_cancel_btn")

        if do_cancel:
            st.session_state.pop(DELETE_TARGET_KEY, None)
            st.rerun()
        if do_delete:
            try:
                dataset_service.delete_dataset(target.id)
            except AppError as err:
                flash("error", str(err))
                st.rerun()
            else:
                flash("success", f"'{target.file_name}' 를 삭제했습니다.")
                if get_state(SessionKey.CURRENT_DATASET_ID) == target.id:
                    st.session_state.pop(SessionKey.CURRENT_DATASET_ID, None)
                st.session_state.pop(DELETE_TARGET_KEY, None)
                st.rerun()


# -------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)
    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return
    if current_project is None:
        _render_project_required_guard()
        return

    _render_upload_form(current_project)

    datasets = _load_datasets(current_project.id)
    _render_delete_flow(datasets)

    current_dataset_id = get_state(SessionKey.CURRENT_DATASET_ID)
    # 프로젝트에 속하지 않는 stale id 는 렌더 전에 제거
    if current_dataset_id is not None and not any(
        d.id == int(current_dataset_id) for d in datasets
    ):
        st.session_state.pop(SessionKey.CURRENT_DATASET_ID, None)
        current_dataset_id = None

    if current_dataset_id is not None:
        _render_preview_section(int(current_dataset_id), datasets)
        st.divider()

    _render_list(
        datasets,
        current_id=int(current_dataset_id) if current_dataset_id is not None else None,
    )


if __name__ == "__main__":
    main()
