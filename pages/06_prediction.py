"""예측 페이지 (IMPLEMENTATION_PLAN §6.6, FR-080~085, §10.4).

화면 구성:
1) 모델 선택 selectbox — 프로젝트의 **저장 가능 모델**(학습 성공 + 아티팩트 존재). 기본값은 `SessionKey.CURRENT_MODEL_ID`
   → 없으면 `is_best=True` 모델 → 여전히 없으면 목록 첫 번째
2) 모델 상세 요약(알고리즘/주 지표/입력 컬럼 수)
3) 탭: **단건 입력** / **파일 예측**
   - 단건: `feature_schema` 기반 입력 폼 자동 생성 (numeric → `number_input`, categorical → `selectbox`)
   - 파일: CSV/XLSX 업로드 → `predict_batch` → 결과 미리보기 + CSV 다운로드 (FR-085)
4) 누락 컬럼은 `PredictionInputError` 로 차단되어 flash error 로 노출 (§10.4)

결정 기록:
- 업로드 파일은 `<predictions_dir>/_inputs/<uuid>.<ext>` 로 임시 저장 후 `predict_batch` 에 경로 전달.
  Service 가 결과 CSV 만 `<predictions_dir>/<job_id>.csv` 에 쓰므로 `_inputs/` 는 즉시 정리해도 안전하지만,
  예측 이력 재현을 위해 보존한다(§7.1 감사 로그와 정합).
- 폼 key 는 `form_input_<model_id>_<column>` 으로 모델 전환 시 자동 초기화.
- `predict_proba` 컬럼은 결과 표에만 표시하고 단건에서는 상위 3개만 별도 카드로 강조.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import streamlit as st

from config.settings import settings
from pages.components.help import render_help
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import model_service, prediction_service, project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError, ValidationError
from utils.file_utils import ALLOWED_EXTENSIONS, extract_extension
from utils.messages import Msg
from utils.session_utils import SessionKey, flash, get_state, set_state

if TYPE_CHECKING:
    from services.dto import (
        FeatureSchemaDTO,
        ModelDetailDTO,
        ModelDTO,
        PredictionResultDTO,
        ProjectDTO,
    )


PAGE_TITLE: Final[str] = "예측"
PAGE_CAPTION: Final[str] = "저장된 모델로 단건·파일 예측을 실행하세요."

SINGLE_RESULT_KEY: Final[str] = "prediction_single_result"
BATCH_RESULT_KEY: Final[str] = "prediction_batch_result"
SELECTED_MODEL_KEY: Final[str] = "prediction_selected_model_id"


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


def _load_predictable_models(project_id: int) -> list[ModelDTO]:
    """학습 성공(metric_score 존재) 모델만 노출. 아티팩트 부재는 Service 가 보호."""
    try:
        all_models = model_service.list_models(project_id)
    except AppError as err:
        flash("error", str(err))
        return []
    return [m for m in all_models if m.metric_score is not None]


def _load_detail(model_id: int) -> ModelDetailDTO | None:
    try:
        return model_service.get_model_detail(model_id)
    except AppError as err:
        flash("error", str(err))
        return None


# ------------------------------------------------------------- model picker


def _resolve_default_model_id(models: list[ModelDTO]) -> int:
    """기본 선택 모델 id 우선순위: CURRENT_MODEL_ID → is_best → 첫 행."""
    candidate = get_state(SessionKey.CURRENT_MODEL_ID)
    if candidate is not None:
        cid = int(candidate)
        if any(m.id == cid for m in models):
            return cid
    best = next((m for m in models if m.is_best), None)
    if best is not None:
        return best.id
    return models[0].id


def _render_model_picker(models: list[ModelDTO]) -> ModelDTO:
    default_id = _resolve_default_model_id(models)
    default_idx = next(
        (i for i, m in enumerate(models) if m.id == default_id),
        0,
    )

    def _format(model: ModelDTO) -> str:
        badge = "★ " if model.is_best else ""
        score = f" · score={model.metric_score:.4f}" if model.metric_score is not None else ""
        return f"{badge}{model.algo_name} (id={model.id}){score}"

    selected = st.selectbox(
        "예측에 사용할 모델",
        options=models,
        index=default_idx,
        format_func=_format,
        key="prediction_model_picker",
    )
    if selected is None:  # pragma: no cover - selectbox 는 항상 값을 반환
        selected = models[0]
    # 다음 렌더 사이클에 기억 (페이지 재진입 시 동일 모델 선택 유지)
    set_state(SessionKey.CURRENT_MODEL_ID, int(selected.id))
    set_state(SELECTED_MODEL_KEY, int(selected.id))
    return selected


# ---------------------------------------------------------- single-form tab


def _build_input_form(schema: FeatureSchemaDTO, model_id: int) -> dict[str, Any]:
    """스키마 기반 입력 위젯 렌더 → 사용자 입력 dict."""
    payload: dict[str, Any] = {}
    columns = list(schema.numeric) + list(schema.categorical)
    # 2열 그리드로 컴팩트하게
    cols = st.columns(2)
    for idx, name in enumerate(columns):
        holder = cols[idx % 2]
        widget_key = f"form_input_{model_id}_{name}"
        if name in schema.numeric:
            payload[name] = holder.number_input(
                name,
                value=0.0,
                format="%.6f",
                key=widget_key,
            )
        else:
            options = schema.categories.get(name, [])
            if options:
                payload[name] = holder.selectbox(
                    name,
                    options=options,
                    key=widget_key,
                )
            else:
                # 카테고리 목록이 비어 있으면 자유 입력으로 폴백
                payload[name] = holder.text_input(
                    name,
                    key=widget_key,
                )
    return payload


def _run_single_prediction(model_id: int, payload: dict[str, Any]) -> None:
    try:
        result = prediction_service.predict_single(model_id, payload)
    except AppError as err:
        flash("error", str(err))
        st.session_state.pop(SINGLE_RESULT_KEY, None)
        st.rerun()
        return
    set_state(SINGLE_RESULT_KEY, {"model_id": model_id, "result": result})
    flash("success", Msg.PREDICTION_COMPLETED)
    st.rerun()


def _render_single_result(result: PredictionResultDTO) -> None:
    if not result.rows:
        st.info("예측 결과가 비어 있습니다.")
        return
    row = result.rows[0]
    pred_value = row.get(prediction_service.PREDICTION_COLUMN)
    st.metric("예측 결과", f"{pred_value}")

    # 확률 상위 카드 (분류일 때만 유의미)
    proba_items = sorted(
        (
            (k[len(prediction_service.PROBABILITY_PREFIX) :], v)
            for k, v in row.items()
            if isinstance(k, str)
            and k.startswith(prediction_service.PROBABILITY_PREFIX)
            and isinstance(v, int | float)
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if proba_items:
        st.markdown("**클래스별 확률 (상위 3개)**")
        proba_cols = st.columns(min(len(proba_items), 3))
        for col, (cls, prob) in zip(proba_cols, proba_items[:3], strict=False):
            col.metric(str(cls), f"{prob:.4f}")

    with st.expander("상세 입력/결과", expanded=False):
        st.dataframe(pd.DataFrame(result.rows), width="stretch")

    for warning in result.warnings:
        st.warning(warning)


def _render_single_tab(detail: ModelDetailDTO) -> None:
    schema = detail.feature_schema
    if not schema.numeric and not schema.categorical:
        st.warning("이 모델에는 입력 스키마가 남아 있지 않아 단건 예측을 생성할 수 없습니다.")
        return

    with st.form(key=f"prediction_single_form_{detail.base.id}"):
        st.caption("각 컬럼에 값을 입력한 뒤 '예측 실행'을 눌러 주세요.")
        payload = _build_input_form(schema, detail.base.id)
        submit = st.form_submit_button("예측 실행", type="primary")
    if submit:
        _run_single_prediction(detail.base.id, payload)

    cached = get_state(SINGLE_RESULT_KEY)
    if cached and cached.get("model_id") == detail.base.id:
        st.divider()
        st.markdown("### 최근 단건 예측")
        _render_single_result(cached["result"])


# ----------------------------------------------------------------- batch tab


def _stash_uploaded_file(uploaded: Any, project_id: int) -> Path:
    """업로드 파일을 ``<predictions_dir>/_inputs/<project_id>/<uuid>.<ext>`` 로 저장."""
    ext = extract_extension(uploaded.name)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(Msg.FILE_PARSE_FAILED)
    root = settings.predictions_dir / "_inputs" / str(project_id)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{uuid.uuid4().hex}.{ext}"
    dest.write_bytes(uploaded.getvalue())
    return dest


def _run_batch_prediction(model_id: int, project_id: int, uploaded: Any) -> None:
    try:
        path = _stash_uploaded_file(uploaded, project_id)
    except AppError as err:
        flash("error", str(err))
        st.rerun()
        return
    try:
        result = prediction_service.predict_batch(model_id, path)
    except AppError as err:
        flash("error", str(err))
        st.session_state.pop(BATCH_RESULT_KEY, None)
        st.rerun()
        return
    set_state(
        BATCH_RESULT_KEY,
        {"model_id": model_id, "result": result, "source_name": uploaded.name},
    )
    flash("success", Msg.PREDICTION_COMPLETED)
    st.rerun()


def _render_batch_result(result: PredictionResultDTO, source_name: str) -> None:
    st.caption(
        f"원본: **{source_name}** · 예측 잡 id=`{result.job_id}` · "
        f"행수={len(result.rows)} (미리보기 최대 {prediction_service.BATCH_PREVIEW_MAX_ROWS})"
    )
    for warning in result.warnings:
        st.warning(warning)

    df = pd.DataFrame(result.rows)
    st.dataframe(df, width="stretch", height=360)

    if result.result_path:
        result_path = Path(result.result_path)
        if result_path.exists():
            try:
                payload = result_path.read_bytes()
            except OSError as exc:  # pragma: no cover - 드문 경우
                st.error(f"결과 파일을 읽지 못했습니다: {exc}")
            else:
                st.download_button(
                    "결과 CSV 다운로드",
                    data=payload,
                    file_name=f"prediction_{result.job_id}.csv",
                    mime="text/csv",
                    type="primary",
                )
        else:
            st.info("결과 파일이 서버에서 이동되었을 수 있습니다. 다시 실행해 주세요.")


def _render_batch_tab(model: ModelDTO, project_id: int) -> None:
    st.caption(
        "CSV 또는 XLSX 를 업로드하면 학습 시 사용한 입력 스키마에 맞게 예측이 실행됩니다. "
        "누락된 컬럼이 있으면 안내 후 실행을 차단합니다."
    )
    uploaded = st.file_uploader(
        "예측 입력 파일",
        type=list(ALLOWED_EXTENSIONS),
        key=f"prediction_batch_upload_{model.id}",
    )
    run = st.button(
        "파일 예측 실행",
        key=f"prediction_batch_run_{model.id}",
        type="primary",
        disabled=uploaded is None,
    )
    if run and uploaded is not None:
        _run_batch_prediction(model.id, project_id, uploaded)

    cached = get_state(BATCH_RESULT_KEY)
    if cached and cached.get("model_id") == model.id:
        st.divider()
        st.markdown("### 최근 파일 예측 결과")
        _render_batch_result(cached["result"], cached["source_name"])


# ---------------------------------------------------------- model summary


def _render_model_summary(detail: ModelDetailDTO) -> None:
    base = detail.base
    schema = detail.feature_schema
    summary = detail.metrics_summary

    c1, c2, c3 = st.columns(3)
    c1.metric("알고리즘", base.algo_name)
    if base.metric_score is not None:
        c2.metric("주 지표", f"{base.metric_score:.4f}")
    else:
        c2.metric("주 지표", "—")
    c3.metric("입력 컬럼", f"{len(schema.numeric) + len(schema.categorical)}")

    if summary:
        with st.expander("metrics_summary", expanded=False):
            metric_cols = st.columns(min(len(summary), 4))
            for col, (key, value) in zip(metric_cols, summary.items(), strict=False):
                col.metric(key, f"{value:.4f}")
    if base.is_best:
        st.caption("★ 이 모델은 현재 학습 잡의 베스트로 고정되어 있습니다.")


# ------------------------------------------------------------------- main


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)
    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)
    render_help("06_prediction")

    if not db_ready:
        st.error("DB가 아직 초기화되지 않았습니다.")
        st.code("python scripts/init_db.py --seed", language="bash")
        return
    if current_project is None:
        st.warning(Msg.PROJECT_REQUIRED)
        return

    models = _load_predictable_models(current_project.id)
    if not models:
        st.info(Msg.MODEL_REQUIRED)
        st.caption("학습 페이지에서 학습을 실행한 뒤 결과 비교 페이지에서 모델을 저장하세요.")
        return

    selected_model = _render_model_picker(models)
    detail = _load_detail(selected_model.id)
    if detail is None:
        return

    _render_model_summary(detail)

    st.divider()
    tab_single, tab_batch = st.tabs(["단건 입력", "파일 예측"])
    with tab_single:
        _render_single_tab(detail)
    with tab_batch:
        _render_batch_tab(selected_model, current_project.id)


if __name__ == "__main__":
    main()
