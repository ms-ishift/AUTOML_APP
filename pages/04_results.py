"""결과 비교 페이지 (IMPLEMENTATION_PLAN §6.4, FR-070~073).

화면 구성:
1) 학습 잡 선택(최근순) — 기본값은 ``SessionKey.LAST_TRAINING_JOB_ID`` 또는 리스트 최상단
2) 요약 카드(시도/성공/실패 + 베스트 배지)
3) 성능 비교표 — 기준 지표 방향 기준 정렬, 실패 행은 ``—`` 로 표기, is_best 는 ``★`` 배지
4) 플롯 섹션 — 알고리즘 선택(성공 모델 중) → 분류: 혼동행렬 heatmap / 회귀: 예측 vs 실제 scatter
5) "이 모델 저장"(베스트) / "다른 모델 저장"(드롭다운) 버튼 → ``model_service.save_model`` 로 is_best pin
6) "모델 관리로 이동" CTA

결정 기록:
- 플롯은 학습 시점에 생성된 ``<model_dir>/plot_data.json`` 을 로드(재학습 비용 제거).
- ``st.dataframe`` + ``column_config`` 로 is_best 는 "★ best", status 는 Check/X 이모지. 커스텀 스타일링
  최소화 — AppTest 친화 유지.
- 실패 행도 비교표에 유지(FR-066). 지표 컬럼은 ``None`` → pandas 가 자동으로 ``—`` 처럼 표시되도록 ``object`` 로 캐스팅.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import streamlit as st

from ml.evaluators import METRIC_DIRECTIONS
from pages.components.help import render_help
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import model_service, project_service, training_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import (
        ModelComparisonRowDTO,
        ProjectDTO,
        TrainingJobDTO,
        TrainingResultDTO,
    )


PAGE_TITLE: Final[str] = "결과 비교"
PAGE_CAPTION: Final[str] = "학습된 모델들의 성능을 비교하고 베스트 모델을 저장하세요."


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


def _load_jobs(project_id: int) -> list[TrainingJobDTO]:
    try:
        return training_service.list_training_jobs(project_id)
    except AppError as err:
        flash("error", str(err))
        return []


def _load_result(job_id: int) -> TrainingResultDTO | None:
    try:
        return training_service.get_training_result(job_id)
    except AppError as err:
        flash("error", str(err))
        return None


# ------------------------------------------------------------------ render


def _sort_rows(result: TrainingResultDTO) -> list[ModelComparisonRowDTO]:
    """기준 지표 방향에 맞춰 정렬. 실패 행은 맨 아래로."""
    direction = METRIC_DIRECTIONS.get(result.metric_key, "max")
    reverse = direction == "max"

    success = [r for r in result.rows if r.status == "success"]
    failed = [r for r in result.rows if r.status != "success"]

    success.sort(
        key=lambda r: r.metrics.get(result.metric_key, float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )
    return success + failed


def _build_table(result: TrainingResultDTO) -> list[dict[str, Any]]:
    """``st.dataframe`` 에 넘길 레코드 리스트를 생성. 실패 행의 지표는 ``None``."""
    all_metrics: list[str] = []
    seen: set[str] = set()
    for row in result.rows:
        for k in row.metrics:
            if k not in seen:
                seen.add(k)
                all_metrics.append(k)
    # 기준 지표를 첫 컬럼으로 끌어올림
    if result.metric_key in all_metrics:
        all_metrics.remove(result.metric_key)
        all_metrics.insert(0, result.metric_key)

    records: list[dict[str, Any]] = []
    for row in _sort_rows(result):
        base: dict[str, Any] = {
            "is_best": "★ best" if row.is_best else "",
            "algorithm": row.algo_name,
            "status": "success" if row.status == "success" else "failed",
            "train_time_ms": row.train_time_ms,
        }
        for m in all_metrics:
            base[m] = row.metrics.get(m) if row.status == "success" else None
        base["error"] = row.error or ""
        records.append(base)
    return records


def _render_job_picker(jobs: list[TrainingJobDTO], *, current_id: int | None) -> int | None:
    if not jobs:
        return None
    labels = {
        j.id: (
            f"[{j.id}] {j.task_type} · target={j.target_column} · " f"{j.metric_key} · {j.status}"
        )
        for j in jobs
    }
    ids = list(labels.keys())
    default_idx = ids.index(current_id) if current_id in labels else 0
    selected = st.selectbox(
        "학습 잡 선택",
        options=ids,
        format_func=lambda i: labels[i],
        index=default_idx,
        key="results_job_pick",
    )
    if selected != current_id:
        set_state(SessionKey.LAST_TRAINING_JOB_ID, int(selected))
    return int(selected)


def _render_summary(result: TrainingResultDTO) -> None:
    success = [r for r in result.rows if r.status == "success"]
    failed = [r for r in result.rows if r.status != "success"]
    best = next((r for r in result.rows if r.is_best), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("학습 시도", len(result.rows))
    c2.metric("성공", len(success))
    c3.metric("실패", len(failed))

    if best is not None:
        score = best.metrics.get(result.metric_key)
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        direction = METRIC_DIRECTIONS.get(result.metric_key, "max")
        arrow = "↑" if direction == "max" else "↓"
        st.success(f"베스트 모델: **{best.algo_name}** · {result.metric_key} {arrow} {score_text}")
    else:
        st.warning("성공한 모델이 없습니다. 학습 이력을 확인해 주세요.")


def _render_comparison_table(result: TrainingResultDTO) -> None:
    records = _build_table(result)
    # column_config 로 is_best 를 짧은 폭 배지 컬럼으로
    column_config = {
        "is_best": st.column_config.TextColumn("베스트", width="small"),
        "algorithm": st.column_config.TextColumn("알고리즘"),
        "status": st.column_config.TextColumn("상태", width="small"),
        "train_time_ms": st.column_config.NumberColumn("학습 시간(ms)", format="%d"),
        "error": st.column_config.TextColumn("에러", width="medium"),
    }
    # 메트릭 컬럼은 소수점 4자리 고정
    for key in METRIC_DIRECTIONS:
        column_config[key] = st.column_config.NumberColumn(key, format="%.4f")

    st.dataframe(
        records,
        column_config=column_config,
        hide_index=True,
        width="stretch",
    )


def _render_plot(task_type: str, plot_data: dict[str, Any]) -> None:
    """``plot_data`` dict 기반으로 플롯을 렌더.

    - 분류: 혼동행렬은 plotly heatmap 으로 표시. plotly 가 없거나 실패하면 dataframe 폴백.
    - 회귀: 예측 vs 실제 scatter (plotly). 대각선(y=x) 참고선 포함.
    """
    kind = plot_data.get("kind")
    if kind == "confusion_matrix":
        labels = plot_data.get("labels", [])
        matrix = plot_data.get("matrix", [])
        try:
            import plotly.express as px

            fig = px.imshow(
                matrix,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "예측", "y": "실제", "color": "count"},
            )
            fig.update_layout(
                margin={"l": 40, "r": 20, "t": 30, "b": 40},
                title="혼동행렬",
            )
            st.plotly_chart(fig, width="stretch")
        except Exception:  # noqa: BLE001 - plotly 실패 시 안전 폴백
            st.caption("plotly 렌더 실패 — 원시 행렬을 대신 표시합니다.")
            st.dataframe(
                {label: row for label, row in zip(labels, matrix, strict=False)},
                width="stretch",
            )
    elif kind == "regression_scatter":
        y_true = plot_data.get("y_true", [])
        y_pred = plot_data.get("y_pred", [])
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    name="예측",
                    marker={"size": 6, "opacity": 0.6},
                )
            )
            if y_true:
                lo = min(min(y_true), min(y_pred))
                hi = max(max(y_true), max(y_pred))
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[lo, hi],
                        mode="lines",
                        name="y = x",
                        line={"dash": "dash", "color": "gray"},
                    )
                )
            fig.update_layout(
                xaxis_title="실제",
                yaxis_title="예측",
                title="예측 vs 실제",
                margin={"l": 40, "r": 20, "t": 30, "b": 40},
            )
            st.plotly_chart(fig, width="stretch")
        except Exception:  # noqa: BLE001
            st.caption("plotly 렌더 실패 — 표로 대신 표시합니다.")
            st.dataframe(
                {"y_true": y_true, "y_pred": y_pred},
                width="stretch",
            )
    else:
        st.info(f"알 수 없는 플롯 유형입니다: kind={kind!r} (task_type={task_type})")


def _render_feature_influence_section(result: TrainingResultDTO) -> None:
    """FR-094~095: 순열 중요도 + (가능 시) 전처리 후 내장 트리 중요도."""
    success_rows = [r for r in result.rows if r.status == "success" and r.model_id is not None]
    if not success_rows:
        return
    best_idx = next((i for i, r in enumerate(success_rows) if r.is_best), 0)
    with st.expander("특성 영향도 (전역)", expanded=False):
        st.caption(Msg.INFLUENCE_DISCLAIMER)
        pick = st.selectbox(
            "대상 모델",
            options=list(range(len(success_rows))),
            format_func=lambda i: (
                f"{'★ ' if success_rows[i].is_best else ''}{success_rows[i].algo_name}"
            ),
            index=best_idx,
            key="results_influence_model_pick",
        )
        row = success_rows[int(pick)]
        assert row.model_id is not None
        if st.button(Msg.INFLUENCE_COMPUTE_BUTTON, key="results_influence_btn"):
            try:
                inf = model_service.get_feature_influence(row.model_id)
            except AppError as err:
                flash("error", str(err))
            else:
                st.caption(
                    Msg.INFLUENCE_ROWS_CAPTION.format(
                        used=inf.n_rows_used,
                        total=inf.n_test_rows,
                    )
                )
                st.caption(f"scoring: **{inf.scoring}**")
                top_perm = list(inf.permutation_rows)[:20]
                if top_perm:
                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure(
                            go.Bar(
                                x=[r.permutation_mean for r in reversed(top_perm)],
                                y=[r.feature_name for r in reversed(top_perm)],
                                orientation="h",
                                error_x=dict(
                                    type="data",
                                    array=[r.permutation_std for r in reversed(top_perm)],
                                    visible=True,
                                ),
                            )
                        )
                        fig.update_layout(
                            title="순열 중요도 (상위 20개, mean ± std)",
                            margin={"l": 160, "r": 20, "t": 40, "b": 40},
                            xaxis_title="importance",
                        )
                        st.plotly_chart(fig, width="stretch")
                    except Exception:  # noqa: BLE001
                        pass
                    st.dataframe(
                        [
                            {
                                "feature": r.feature_name,
                                "mean": r.permutation_mean,
                                "std": r.permutation_std,
                            }
                            for r in inf.permutation_rows
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                if inf.builtin_rows:
                    st.subheader(Msg.INFLUENCE_BUILTIN_SECTION)
                    st.caption(
                        "이 표는 **전처리 후** 피처 이름(one-hot 등)을 사용합니다. "
                        "위 순열 중요도와 열 이름·스케일이 다를 수 있습니다."
                    )
                    top_b = list(inf.builtin_rows)[:20]
                    try:
                        import plotly.graph_objects as go

                        fig2 = go.Figure(
                            go.Bar(
                                x=[r.importance for r in reversed(top_b)],
                                y=[r.feature_name for r in reversed(top_b)],
                                orientation="h",
                            )
                        )
                        fig2.update_layout(
                            title="내장 중요도 (상위 20개)",
                            margin={"l": 200, "r": 20, "t": 40, "b": 40},
                            xaxis_title="importance",
                        )
                        st.plotly_chart(fig2, width="stretch")
                    except Exception:  # noqa: BLE001
                        pass
                    st.dataframe(
                        [
                            {"feature": r.feature_name, "importance": r.importance}
                            for r in inf.builtin_rows
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info(Msg.INFLUENCE_BUILTIN_NONE)


def _render_plot_section(result: TrainingResultDTO) -> None:
    success_rows = [r for r in result.rows if r.status == "success" and r.model_id is not None]
    if not success_rows:
        st.info("플롯은 성공한 모델에 대해서만 제공됩니다. 이번 학습에는 성공 모델이 없습니다.")
        return

    best_idx = next((i for i, r in enumerate(success_rows) if r.is_best), 0)
    selected = st.selectbox(
        "플롯 대상 모델",
        options=list(range(len(success_rows))),
        format_func=lambda i: (
            f"{'★ ' if success_rows[i].is_best else ''}{success_rows[i].algo_name}"
        ),
        index=best_idx,
        key="results_plot_pick",
    )
    selected_row = success_rows[int(selected)]
    assert selected_row.model_id is not None  # success_rows 필터

    plot_data = model_service.get_model_plot_data(selected_row.model_id)
    if plot_data is None:
        st.caption("이 모델에는 저장된 플롯 데이터가 없습니다 (구버전 아티팩트일 수 있음).")
        return
    _render_plot(result.task_type or "", plot_data)


def _render_save_actions(result: TrainingResultDTO) -> None:
    success_rows = [r for r in result.rows if r.status == "success" and r.model_id is not None]
    if not success_rows:
        return

    st.subheader("모델 저장")
    st.caption(
        "선택한 모델을 해당 학습 잡의 베스트로 고정합니다. 모델 관리/예측 페이지에서 이 모델을 기본으로 사용합니다."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        options = [
            f"{'★ ' if r.is_best else ''}{r.algo_name} (model_id={r.model_id})"
            for r in success_rows
        ]
        idx = next((i for i, r in enumerate(success_rows) if r.is_best), 0)
        picked = st.selectbox(
            "저장할 모델",
            options=list(range(len(success_rows))),
            format_func=lambda i: options[i],
            index=idx,
            key="results_save_pick",
        )
    picked_row = success_rows[int(picked)]
    assert picked_row.model_id is not None

    with c2:
        disabled = picked_row.is_best
        label = "베스트로 고정됨" if disabled else "이 모델 저장"
        if st.button(
            label,
            type="primary",
            width="stretch",
            disabled=disabled,
            key="results_save_btn",
        ):
            try:
                model_service.save_model(picked_row.model_id)
            except AppError as err:
                flash("error", f"{Msg.SAVE_FAILED} — {err}")
            else:
                flash("success", Msg.MODEL_SAVED)
            st.rerun()


def _render_nav_cta() -> None:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "모델 관리로 이동",
            width="stretch",
            key="results_goto_models",
        ):
            try:
                st.switch_page("pages/05_models.py")
            except Exception:  # noqa: BLE001 - §6.5 미구현 폴백
                flash("info", "모델 관리 페이지는 §6.5 에서 구현됩니다.")
                st.rerun()
    with c2:
        if st.button(
            "다시 학습하기",
            width="stretch",
            key="results_goto_training",
        ):
            try:
                st.switch_page("pages/03_training.py")
            except Exception:  # noqa: BLE001
                flash("info", "학습 페이지로 이동할 수 없습니다.")
                st.rerun()


# ------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)
    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)
    render_help("04_results")

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return
    if current_project is None:
        st.warning(Msg.PROJECT_REQUIRED)
        return

    jobs = _load_jobs(current_project.id)
    if not jobs:
        st.warning(Msg.TRAINING_RESULT_REQUIRED)
        st.caption("학습 페이지에서 먼저 학습을 실행하세요.")
        return

    last_id = get_state(SessionKey.LAST_TRAINING_JOB_ID)
    picked = _render_job_picker(
        jobs,
        current_id=int(last_id) if last_id is not None else None,
    )
    if picked is None:
        return

    result = _load_result(picked)
    if result is None:
        return

    st.divider()
    _render_summary(result)

    st.subheader("성능 비교표")
    _render_comparison_table(result)

    st.subheader("플롯")
    _render_plot_section(result)

    _render_feature_influence_section(result)

    _render_save_actions(result)
    _render_nav_cta()


if __name__ == "__main__":
    main()
