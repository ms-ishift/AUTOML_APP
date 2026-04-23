"""모델 관리 페이지 (IMPLEMENTATION_PLAN §6.5, FR-074·FR-075).

화면 구성:
1) 프로젝트의 저장 모델 목록(알고리즘/생성일/주 지표/베스트 배지)
2) 필터: 저장된 것만 보기(is_best=True) 토글
3) 모델 상세 — 선택 시 `feature_schema` + `metrics_summary` 를 아래에 전개
4) 액션: `예측하러 가기` (→ §6.6 + `SessionKey.CURRENT_MODEL_ID`) / `삭제` (세션 플래그 인라인 컨펌)

결정 기록:
- `st.dialog` 대신 세션 플래그 기반 인라인 컨펌 — §6.1·§6.2 와 동일한 패턴으로 AppTest 친화 유지
- 목록은 `st.container(border=True)` 행 + 4컬럼(이름/지표/액션x2) 레이아웃. 저장된 모델이 많지 않은 MVP 이라
  페이지네이션은 생략.
- 모델 상세 로드는 `st.session_state` 로 선택된 model_id 만 관리 (expander 기본 펼침)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from pages.components.help import render_help
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import model_service, project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import ModelDetailDTO, ModelDTO, ProjectDTO


PAGE_TITLE: Final[str] = "모델 관리"
PAGE_CAPTION: Final[str] = "프로젝트에 저장된 모델을 확인하고 예측에 활용하세요."

# 세션 키
SELECTED_MODEL_KEY: Final[str] = "models_selected_detail_id"
DELETE_TARGET_KEY: Final[str] = "models_delete_target_id"
BEST_ONLY_KEY: Final[str] = "models_filter_best_only"


# ------------------------------------------------------------- data loaders


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


def _load_models(project_id: int) -> list[ModelDTO]:
    try:
        return model_service.list_models(project_id)
    except AppError as err:
        flash("error", str(err))
        return []


def _load_detail(model_id: int) -> ModelDetailDTO | None:
    try:
        return model_service.get_model_detail(model_id)
    except AppError as err:
        flash("error", str(err))
        return None


# ------------------------------------------------------------------ actions


def _go_predict(model_id: int) -> None:
    set_state(SessionKey.CURRENT_MODEL_ID, int(model_id))
    try:
        st.switch_page("pages/06_prediction.py")
    except Exception:  # noqa: BLE001 - §6.6 미구현 환경에서는 안내 flash 로 폴백
        flash(
            "info",
            "예측 페이지는 §6.6 에서 구현됩니다. 선택된 모델은 `CURRENT_MODEL_ID` 에 저장되었습니다.",
        )
        st.rerun()


def _apply_delete(model_id: int) -> None:
    try:
        model_service.delete_model(model_id)
    except AppError as err:
        flash("error", str(err))
    else:
        flash("success", Msg.MODEL_DELETED)
        # 현재 선택 상세가 삭제 대상이면 정리
        if st.session_state.get(SELECTED_MODEL_KEY) == model_id:
            st.session_state.pop(SELECTED_MODEL_KEY, None)
        if st.session_state.get(SessionKey.CURRENT_MODEL_ID) == model_id:
            st.session_state.pop(SessionKey.CURRENT_MODEL_ID, None)
    st.session_state.pop(DELETE_TARGET_KEY, None)
    st.rerun()


# ------------------------------------------------------------------- render


def _format_model_row_caption(model: ModelDTO) -> str:
    score_part = f" · score={model.metric_score:.4f}" if model.metric_score is not None else ""
    return (
        f"학습잡 {model.training_job_id} · " f"생성 {model.created_at:%Y-%m-%d %H:%M}{score_part}"
    )


def _render_filter_bar() -> bool:
    best_only = st.toggle(
        "저장된 모델(베스트)만 보기",
        value=bool(get_state(BEST_ONLY_KEY, False)),
        key=BEST_ONLY_KEY,
        help="학습 직후에는 모든 알고리즘의 모델이 DB 에 남습니다. "
        "결과 페이지에서 '이 모델 저장' 을 한 모델만 보려면 체크하세요.",
    )
    return bool(best_only)


def _render_model_row(model: ModelDTO, *, current_detail_id: int | None) -> None:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1:
            prefix = "★ " if model.is_best else ""
            st.markdown(f"**{prefix}{model.algo_name}**  ·  `model_id={model.id}`")
            st.caption(_format_model_row_caption(model))
        with c2:
            if model.metric_score is not None:
                st.metric("주 지표", f"{model.metric_score:.4f}")
            else:
                st.caption("지표 없음(실패 모델일 수 있음)")
        with c3:
            already_selected = current_detail_id == model.id
            if st.button(
                "상세" if not already_selected else "닫기",
                key=f"models_detail_btn_{model.id}",
                width="stretch",
            ):
                if already_selected:
                    st.session_state.pop(SELECTED_MODEL_KEY, None)
                else:
                    set_state(SELECTED_MODEL_KEY, int(model.id))
                st.rerun()
        with c4:
            if st.button(
                "삭제",
                key=f"models_delete_btn_{model.id}",
                width="stretch",
                type="secondary",
            ):
                set_state(DELETE_TARGET_KEY, int(model.id))
                st.rerun()

        # 현재 행이 삭제 대상이면 확인 블록을 인라인 표시
        if get_state(DELETE_TARGET_KEY) == model.id:
            _render_delete_confirm(model)

        # 상세가 선택된 행이면 아래에 펼침
        if current_detail_id == model.id:
            _render_detail_block(model)


def _render_delete_confirm(model: ModelDTO) -> None:
    with st.container(border=True):
        st.warning(
            f"모델 **{model.algo_name}** (id={model.id}) 을(를) 삭제할까요? "
            "연결된 예측 이력과 아티팩트 파일도 함께 제거됩니다."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "삭제 확정",
                key=f"models_delete_confirm_{model.id}",
                type="primary",
                width="stretch",
            ):
                _apply_delete(model.id)
        with c2:
            if st.button(
                "취소",
                key=f"models_delete_cancel_{model.id}",
                width="stretch",
            ):
                st.session_state.pop(DELETE_TARGET_KEY, None)
                st.rerun()


def _render_detail_block(model: ModelDTO) -> None:
    detail = _load_detail(model.id)
    if detail is None:
        return
    schema = detail.feature_schema
    summary = detail.metrics_summary

    st.markdown("**입력 스키마**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"수치형 ({len(schema.numeric)})")
        if schema.numeric:
            st.write(", ".join(schema.numeric))
        else:
            st.caption("—")
    with c2:
        st.caption(f"범주형 ({len(schema.categorical)})")
        if schema.categorical:
            st.write(", ".join(schema.categorical))
        else:
            st.caption("—")
    if schema.target:
        st.caption(f"타깃: **{schema.target}**")
    if schema.categories:
        with st.expander("범주형 컬럼 값 목록", expanded=False):
            for col, values in schema.categories.items():
                st.write(f"- **{col}**: {', '.join(map(str, values))}")

    st.markdown("**metrics_summary**")
    if summary:
        metric_cols = st.columns(min(len(summary), 4))
        for metric_col, (metric_key, value) in zip(metric_cols, summary.items(), strict=False):
            metric_col.metric(metric_key, f"{value:.4f}")
    else:
        st.caption("이 모델에는 기록된 지표가 없습니다 (실패 모델일 수 있음).")

    if st.button(
        "예측하러 가기",
        key=f"models_goto_predict_{model.id}",
        type="primary",
    ):
        _go_predict(model.id)


def _filter_models(models: list[ModelDTO], *, best_only: bool) -> list[ModelDTO]:
    if best_only:
        return [m for m in models if m.is_best]
    return list(models)


# ------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)
    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)
    render_help("05_models")

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return
    if current_project is None:
        st.warning(Msg.PROJECT_REQUIRED)
        return

    models = _load_models(current_project.id)
    best_only = _render_filter_bar()
    filtered = _filter_models(models, best_only=best_only)

    if not models:
        st.info("이 프로젝트에는 아직 저장된 모델이 없습니다.")
        st.caption("학습 페이지에서 먼저 학습을 실행한 뒤 결과 비교 페이지에서 모델을 저장하세요.")
        return
    if not filtered:
        st.info("선택된 필터 조건에 맞는 모델이 없습니다. 필터를 해제해 보세요.")
        return

    current_detail_id_raw = get_state(SELECTED_MODEL_KEY)
    current_detail_id = int(current_detail_id_raw) if current_detail_id_raw is not None else None
    # stale 정리: 상세 선택이 목록에 없으면 해제
    if current_detail_id is not None and not any(m.id == current_detail_id for m in filtered):
        st.session_state.pop(SELECTED_MODEL_KEY, None)
        current_detail_id = None

    st.caption(f"총 {len(filtered)}건 (프로젝트 전체 {len(models)}건)")
    for model in filtered:
        _render_model_row(model, current_detail_id=current_detail_id)


if __name__ == "__main__":
    main()
