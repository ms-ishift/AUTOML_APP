"""이력/관리자 페이지 (IMPLEMENTATION_PLAN §6.7, FR-090~093).

화면 구성:
1) 상단 통계 카드: 프로젝트/데이터셋/학습 잡/모델/예측 수 + 학습·예측 실패 건수
2) 공통 필터: 프로젝트(전체/개별) + 기간(최근 7·30·90·전체) + 상태(전체/completed/failed/running)
3) 탭
   - **학습 이력**: `admin_service.list_training_history` → `st.dataframe`. 실행 시간/성공·실패 모델 수/베스트 표시.
   - **예측 이력**: `admin_service.list_prediction_history`. input_type(form/file) + 결과 파일 경로.
   - **최근 실패**: `admin_service.list_recent_failures` → action_type + detail.

결정 기록:
- 프로젝트 가드는 두지 않는다 (관리자 뷰 — 여러 프로젝트 가로질러 조회). DB 가드만 유지.
- 필터 값은 `st.session_state` 에 두고 페이지 전환 시 보존. 탭 간에도 필터가 공유되도록
  `sidebar` 또는 페이지 상단 공통 영역에 배치.
- 테이블은 `st.dataframe(use_container_width ... )` 대신 `width="stretch"` 로 §6.5 와 동일한 패턴 유지.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final

import pandas as pd
import streamlit as st

from pages.components.help import render_help
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import admin_service, project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.session_utils import SessionKey, flash, get_state

if TYPE_CHECKING:
    from services.dto import (
        AdminStatsDTO,
        AuditLogEntryDTO,
        PredictionHistoryRowDTO,
        ProjectDTO,
        TrainingHistoryRowDTO,
    )


PAGE_TITLE: Final[str] = "이력 / 관리자"
PAGE_CAPTION: Final[str] = "모든 프로젝트의 학습·예측 이력과 실패 로그를 한 눈에 확인합니다."

# 필터 세션 키
FILTER_PROJECT_KEY: Final[str] = "admin_filter_project_id"
FILTER_STATUS_KEY: Final[str] = "admin_filter_status"
FILTER_PERIOD_KEY: Final[str] = "admin_filter_period"

PERIOD_OPTIONS: Final[tuple[tuple[str, int | None], ...]] = (
    ("최근 7일", 7),
    ("최근 30일", 30),
    ("최근 90일", 90),
    ("전체", None),
)
STATUS_OPTIONS: Final[tuple[str, ...]] = (
    "전체",
    "completed",
    "failed",
    "running",
    "pending",
)


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


def _load_projects() -> list[ProjectDTO]:
    try:
        return project_service.list_projects()
    except AppError as err:
        flash("error", str(err))
        return []


def _load_stats() -> AdminStatsDTO | None:
    try:
        return admin_service.get_stats()
    except AppError as err:
        flash("error", str(err))
        return None


# ---------------------------------------------------------------- filters


def _resolve_period_since(period_label: str) -> datetime | None:
    for label, days in PERIOD_OPTIONS:
        if label == period_label and days is not None:
            return datetime.utcnow() - timedelta(days=days)
    return None


def _render_filter_bar(projects: list[ProjectDTO]) -> tuple[int | None, str | None, str]:
    """프로젝트·상태·기간 필터. 반환: (project_id, status, period_label)."""
    project_options: list[tuple[str, int | None]] = [("전체 프로젝트", None)]
    project_options.extend((f"{p.name} (id={p.id})", p.id) for p in projects)
    labels = [label for label, _ in project_options]

    default_pid = get_state(FILTER_PROJECT_KEY)
    try:
        default_idx = next(i for i, (_, pid) in enumerate(project_options) if pid == default_pid)
    except StopIteration:
        default_idx = 0

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        selected_label = st.selectbox(
            "프로젝트",
            options=labels,
            index=default_idx,
            key="admin_project_picker",
        )
        selected_pid = next(pid for label, pid in project_options if label == selected_label)
        st.session_state[FILTER_PROJECT_KEY] = selected_pid
    with c2:
        status_value = st.selectbox(
            "상태",
            options=STATUS_OPTIONS,
            index=STATUS_OPTIONS.index(
                get_state(FILTER_STATUS_KEY, "전체")
                if get_state(FILTER_STATUS_KEY, "전체") in STATUS_OPTIONS
                else "전체"
            ),
            key="admin_status_picker",
        )
        st.session_state[FILTER_STATUS_KEY] = status_value
    with c3:
        period_labels = [label for label, _ in PERIOD_OPTIONS]
        default_period = get_state(FILTER_PERIOD_KEY, "최근 30일")
        period_idx = period_labels.index(default_period) if default_period in period_labels else 1
        period_label = st.selectbox(
            "기간",
            options=period_labels,
            index=period_idx,
            key="admin_period_picker",
        )
        st.session_state[FILTER_PERIOD_KEY] = period_label

    status_filter = None if status_value == "전체" else status_value
    return selected_pid, status_filter, period_label


# ------------------------------------------------------------- render tabs


def _render_stats(stats: AdminStatsDTO) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("프로젝트", stats.projects)
    c2.metric("데이터셋", stats.datasets)
    c3.metric("학습 잡", stats.training_jobs)
    c4.metric("모델", stats.models)
    c5.metric("예측", stats.predictions)
    c6, c7 = st.columns(2)
    c6.metric(
        "학습 실패",
        stats.training_failures,
        delta_color="inverse",
    )
    c7.metric(
        "예측 실패",
        stats.prediction_failures,
        delta_color="inverse",
    )


def _training_rows_to_df(rows: list[TrainingHistoryRowDTO]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append(
            {
                "job_id": r.id,
                "project": r.project_name,
                "task": r.task_type,
                "target": r.target_column,
                "metric": r.metric_key,
                "status": r.status,
                "성공": r.n_models_success,
                "실패": r.n_models_failed,
                "best_algo": r.best_algo or "",
                "best_score": r.best_metric,
                "소요(ms)": r.duration_ms,
                "시작시각": r.started_at,
                "종료시각": r.ended_at,
            }
        )
    return pd.DataFrame.from_records(records)


def _prediction_rows_to_df(rows: list[PredictionHistoryRowDTO]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append(
            {
                "pred_id": r.id,
                "project": r.project_name,
                "model": f"{r.algorithm_name} (id={r.model_id})",
                "input_type": r.input_type,
                "status": r.status,
                "created_at": r.created_at,
                "입력파일": r.input_file_path or "",
                "결과파일": r.result_path or "",
            }
        )
    return pd.DataFrame.from_records(records)


def _failure_rows_to_df(rows: list[AuditLogEntryDTO]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append(
            {
                "log_id": r.id,
                "action_type": r.action_type,
                "target_type": r.target_type or "",
                "target_id": r.target_id,
                "time": r.action_time,
                "detail": r.detail,
            }
        )
    return pd.DataFrame.from_records(records)


def _render_training_tab(
    project_id: int | None,
    status: str | None,
    period_label: str,
) -> None:
    since = _resolve_period_since(period_label)
    try:
        rows = admin_service.list_training_history(
            project_id=project_id,
            status=status,
            since=since,
        )
    except AppError as err:
        st.error(str(err))
        return

    if not rows:
        st.info("선택한 조건에 해당하는 학습 이력이 없습니다.")
        return

    st.caption(f"총 {len(rows)}건")
    df = _training_rows_to_df(rows)
    st.dataframe(
        df,
        width="stretch",
        height=420,
        column_config={
            "best_score": st.column_config.NumberColumn(format="%.4f"),
            "소요(ms)": st.column_config.NumberColumn(format="%d"),
        },
    )


def _render_prediction_tab(
    project_id: int | None,
    status: str | None,
    period_label: str,
) -> None:
    since = _resolve_period_since(period_label)
    try:
        rows = admin_service.list_prediction_history(
            project_id=project_id,
            status=status,
            since=since,
        )
    except AppError as err:
        st.error(str(err))
        return

    if not rows:
        st.info("선택한 조건에 해당하는 예측 이력이 없습니다.")
        return

    st.caption(f"총 {len(rows)}건")
    df = _prediction_rows_to_df(rows)
    st.dataframe(df, width="stretch", height=420)


def _render_failure_tab(period_label: str) -> None:
    since = _resolve_period_since(period_label)
    try:
        rows = admin_service.list_recent_failures(since=since)
    except AppError as err:
        st.error(str(err))
        return

    if not rows:
        st.success("기록된 실패 이벤트가 없습니다.")
        return

    st.caption(f"총 {len(rows)}건 (action_type 이 '_failed' 로 끝나는 감사 로그)")
    df = _failure_rows_to_df(rows)
    st.dataframe(df, width="stretch", height=360)


# ------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)
    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)
    render_help("07_admin")

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return

    stats = _load_stats()
    if stats is not None:
        _render_stats(stats)

    projects = _load_projects()
    st.divider()
    project_id, status, period_label = _render_filter_bar(projects)

    tab_training, tab_prediction, tab_failures = st.tabs(["학습 이력", "예측 이력", "최근 실패"])
    with tab_training:
        _render_training_tab(project_id, status, period_label)
    with tab_prediction:
        _render_prediction_tab(project_id, status, period_label)
    with tab_failures:
        _render_failure_tab(period_label)


if __name__ == "__main__":
    main()
