"""학습 설정/실행 페이지 (IMPLEMENTATION_PLAN §6.3, FR-040~066).

화면 구성:
1) 데이터 확인 섹션 — `data_preview` 컴포넌트로 선택 데이터셋 프로파일 재사용 렌더
2) 학습 설정 폼 — 문제 유형 / 타깃 / 제외 컬럼 / 테스트 비율 / 기준 지표 / 학습명
3) 실행 버튼 → `st.status` 로 진행률(`on_progress(stage, ratio)`) 단계별 표시
4) 완료 시 요약(베스트 / 성공·실패 건수) + 결과 페이지 이동 CTA, `SessionKey.LAST_TRAINING_JOB_ID` 세팅

UX 결정:
- 폼은 **변경 즉시 rerun 되는 개별 위젯** + 하단 ``st.button("실행")`` 구성.
  ``st.form`` 안에 넣으면 타깃 변경 시 excluded 선택지 갱신이 불가능하므로 분리.
- 실행 중 UI 가 멈추는 동안도 ``st.status`` 가 단계/ratio 를 보여준다. ``run_training`` 은 동기 호출.
- 오류는 Service 예외(``AppError``) → 카드 내부 error 렌더, 성공은 summary + 결과 페이지 이동 버튼.
- 결과 페이지(§6.4) 는 아직 구현 전이라 CTA 는 `st.switch_page` 시도 후 실패 시 flash 안내로 폴백.

규약 (``.cursor/rules/streamlit-ui.mdc``):
- Streamlit 위젯과 Service 간 DTO 이외의 값은 오가지 않는다 (ml ImportError 방지를 위해 TrainingConfig
  는 명시적으로 ml.schemas 에서만 import, 본 페이지에서는 run 호출 직전 조립).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Final

import streamlit as st

from config.settings import settings
from ml.evaluators import (
    CLASSIFICATION_METRICS,
    METRIC_DIRECTIONS,
    REGRESSION_METRICS,
)
from ml.schemas import TrainingConfig
from pages.components.data_preview import render_profile
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import dataset_service, project_service, training_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import (
        DatasetDTO,
        DatasetProfileDTO,
        ProjectDTO,
        TrainingResultDTO,
    )


PAGE_TITLE: Final[str] = "학습 설정 / 실행"
PAGE_CAPTION: Final[str] = "타깃과 기준 지표를 고르고 여러 알고리즘을 한 번에 학습합니다."

# 위젯 key
FORM_TASK_KEY: Final[str] = "training_task_type"
FORM_TARGET_KEY: Final[str] = "training_target_col"
FORM_EXCLUDED_KEY: Final[str] = "training_excluded_cols"
FORM_TEST_SIZE_KEY: Final[str] = "training_test_size"
FORM_METRIC_KEY: Final[str] = "training_metric_key"
FORM_JOB_NAME_KEY: Final[str] = "training_job_name"


# --------------------------------------------------------------- data loaders


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


def _load_suggested_excluded(dataset_id: int) -> list[str]:
    try:
        return dataset_service.suggest_excluded_columns(dataset_id)
    except AppError:
        return []


# ------------------------------------------------------------------ render


def _metric_options(task_type: str) -> tuple[str, ...]:
    return CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS


def _describe_metric(metric_key: str) -> str:
    direction = METRIC_DIRECTIONS.get(metric_key, "max")
    return "↑ 클수록 좋음" if direction == "max" else "↓ 작을수록 좋음"


def _render_dataset_picker(datasets: list[DatasetDTO], *, current_id: int | None) -> int | None:
    """데이터 확인 섹션에서 사용할 데이터셋 선택 위젯."""
    labels = {
        d.id: f"[{d.id}] {d.file_name} — {d.row_count:,}행 × {d.column_count}열" for d in datasets
    }
    ids = list(labels.keys())
    default_idx = ids.index(current_id) if current_id in labels else 0
    selected = st.selectbox(
        "대상 데이터셋",
        options=ids,
        format_func=lambda i: labels[i],
        index=default_idx,
        key="training_dataset_pick",
    )
    if selected != current_id:
        set_state(SessionKey.CURRENT_DATASET_ID, int(selected))
    return int(selected)


def _render_config_form(
    profile: DatasetProfileDTO, suggested_excluded: list[str]
) -> TrainingConfig | None:
    """학습 설정 폼. 제출 시 ``TrainingConfig`` 를 반환, 아니면 ``None``."""
    col_names = [c.name for c in profile.columns]
    if not col_names:
        st.warning("이 데이터셋에는 컬럼이 없습니다. 다른 파일을 선택해 주세요.")
        return None

    task_type = st.radio(
        "문제 유형",
        options=("classification", "regression"),
        format_func=lambda v: (
            "분류 (classification)" if v == "classification" else "회귀 (regression)"
        ),
        horizontal=True,
        key=FORM_TASK_KEY,
    )

    target = st.selectbox(
        "예측 대상 (타깃) 컬럼",
        options=col_names,
        key=FORM_TARGET_KEY,
    )

    default_exclude = [c for c in suggested_excluded if c != target]
    exclude_options = [c for c in col_names if c != target]
    excluded = st.multiselect(
        "제외할 컬럼",
        options=exclude_options,
        default=default_exclude,
        key=FORM_EXCLUDED_KEY,
        help=(
            "고유값 비율이 0.95 이상인 컬럼은 식별자로 추정되어 기본 선택됩니다. "
            "필요 시 해제하거나 추가 컬럼을 지정하세요."
        ),
    )

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider(
            "테스트 비율",
            min_value=0.05,
            max_value=0.5,
            value=float(settings.DEFAULT_TEST_SIZE),
            step=0.05,
            key=FORM_TEST_SIZE_KEY,
        )
    with c2:
        metric_options = _metric_options(str(task_type))
        metric_key = st.selectbox(
            "기준 지표",
            options=metric_options,
            index=0,
            key=FORM_METRIC_KEY,
            help=_describe_metric(metric_options[0]),
        )

    job_name = st.text_input(
        "학습명 (선택)",
        max_chars=100,
        placeholder="예: 1차_베이스라인",
        key=FORM_JOB_NAME_KEY,
    )

    submitted = st.button("학습 실행", type="primary", key="training_submit_btn")
    if not submitted:
        return None

    try:
        return TrainingConfig(
            dataset_id=int(st.session_state["training_dataset_pick"]),
            task_type=str(task_type),  # type: ignore[arg-type]
            target_column=str(target),
            excluded_columns=tuple(excluded),
            test_size=float(test_size),
            metric_key=str(metric_key),
            job_name=job_name.strip() or None,
        )
    except ValueError as err:
        flash("error", str(err))
        st.rerun()
        return None


def _run_with_status(config: TrainingConfig) -> TrainingResultDTO | None:
    """``st.status`` 내부에서 ``run_training`` 실행 + 단계별 갱신."""
    with st.status("학습 진행 중...", expanded=True) as status:
        progress_bar = st.progress(0.0, text="준비 중")
        stages_log: list[str] = []

        def _on_progress(stage: str, ratio: float) -> None:
            ratio = max(0.0, min(1.0, float(ratio)))
            progress_bar.progress(ratio, text=f"{stage} ({ratio:.0%})")
            stages_log.append(f"[{time.strftime('%H:%M:%S')}] {stage} · {ratio:.0%}")
            # 너무 많은 줄은 잘라서 표시
            st.caption(" / ".join(stages_log[-6:]))

        try:
            result = training_service.run_training(config, on_progress=_on_progress)
        except AppError as err:
            status.update(label=f"학습 실패: {err}", state="error", expanded=True)
            flash("error", f"{Msg.TRAINING_FAILED} — {err}")
            return None
        else:
            status.update(label="학습 완료", state="complete", expanded=False)
            return result


def _summarize_result(result: TrainingResultDTO) -> None:
    success_rows = [r for r in result.rows if r.status == "success"]
    failed_rows = [r for r in result.rows if r.status != "success"]
    best = next((r for r in result.rows if r.is_best), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("학습 시도", len(result.rows))
    c2.metric("성공", len(success_rows))
    c3.metric("실패", len(failed_rows))

    if best is not None:
        score = best.metrics.get(result.metric_key)
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        st.success(f"베스트 모델: **{best.algo_name}** · {result.metric_key}={score_text}")
    else:
        st.warning("성공한 모델이 없습니다. 실패 원인을 확인해 주세요.")

    if failed_rows:
        with st.expander(f"실패 {len(failed_rows)}건 상세", expanded=False):
            for row in failed_rows:
                st.write(f"- **{row.algo_name}** — {row.error or '원인 미상'}")


def _render_post_run_cta(result: TrainingResultDTO) -> None:
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "결과 비교 페이지로 이동",
            type="primary",
            width="stretch",
            key="training_goto_results",
        ):
            try:
                st.switch_page("pages/04_results.py")
            except Exception:  # noqa: BLE001 - §6.4 구현 전 폴백
                flash("info", "결과 비교 페이지는 §6.4 에서 구현됩니다.")
                st.rerun()
    with c2:
        st.caption(
            f"`LAST_TRAINING_JOB_ID` = {result.job_id} — 결과 페이지는 이 값을 기본으로 사용합니다."
        )


def _render_training_flow(
    project: ProjectDTO,
    datasets: list[DatasetDTO],
    current_dataset_id: int,
) -> None:
    profile = _load_profile(current_dataset_id)
    if profile is None:
        return

    with st.expander("데이터 확인", expanded=False):
        render_profile(profile)

    st.subheader("학습 설정")
    suggested = _load_suggested_excluded(current_dataset_id)
    config = _render_config_form(profile, suggested)

    last_result_id = get_state(SessionKey.LAST_TRAINING_JOB_ID)
    if config is not None:
        result = _run_with_status(config)
        if result is not None:
            set_state(SessionKey.LAST_TRAINING_JOB_ID, result.job_id)
            flash("success", Msg.TRAINING_COMPLETED)
            st.divider()
            st.subheader("결과 요약")
            _summarize_result(result)
            _render_post_run_cta(result)
    elif last_result_id is not None:
        # 직전 실행 결과가 있으면 요약만 다시 보여준다
        try:
            result = training_service.get_training_result(int(last_result_id))
        except AppError:
            st.session_state.pop(SessionKey.LAST_TRAINING_JOB_ID, None)
        else:
            st.divider()
            st.subheader(f"최근 학습 결과 (job_id={result.job_id})")
            _summarize_result(result)
            _render_post_run_cta(result)


# ------------------------------------------------------------------- main


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
        st.warning(Msg.PROJECT_REQUIRED)
        st.caption("프로젝트 페이지에서 먼저 프로젝트를 선택해 주세요.")
        return

    datasets = _load_datasets(current_project.id)
    if not datasets:
        st.warning(Msg.DATASET_REQUIRED)
        st.caption("데이터 업로드 페이지에서 먼저 CSV/XLSX 파일을 업로드해 주세요.")
        return

    current_dataset_id = get_state(SessionKey.CURRENT_DATASET_ID)
    # stale id 정리
    if current_dataset_id is not None and not any(
        d.id == int(current_dataset_id) for d in datasets
    ):
        st.session_state.pop(SessionKey.CURRENT_DATASET_ID, None)
        current_dataset_id = None

    picked = _render_dataset_picker(
        datasets,
        current_id=int(current_dataset_id) if current_dataset_id is not None else None,
    )
    st.divider()
    if picked is not None:
        _render_training_flow(current_project, datasets, picked)


if __name__ == "__main__":
    main()
