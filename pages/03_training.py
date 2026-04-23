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
from ml import (
    balancing as ml_balancing,  # SMOTE_AVAILABLE 을 테스트에서 monkeypatch 하기 쉽게 모듈 참조
)
from ml.evaluators import (
    CLASSIFICATION_METRICS,
    METRIC_DIRECTIONS,
    REGRESSION_METRICS,
)
from ml.feature_engineering import DEFAULT_DATETIME_PARTS
from ml.registry import optional_backends_status
from ml.schemas import PreprocessingConfig, TrainingConfig
from pages.components.data_preview import render_profile
from pages.components.help import render_help
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
        FeaturePreviewDTO,
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

# 고급 전처리 위젯 key (§9.9, FR-055~058)
PP_NUM_IMPUTE_KEY: Final[str] = "pp_numeric_impute"
PP_NUM_SCALE_KEY: Final[str] = "pp_numeric_scale"
PP_OUTLIER_KEY: Final[str] = "pp_outlier"
PP_IQR_K_KEY: Final[str] = "pp_outlier_iqr_k"
PP_WINSORIZE_KEY: Final[str] = "pp_winsorize_p"
PP_CAT_IMPUTE_KEY: Final[str] = "pp_categorical_impute"
PP_CAT_ENCODING_KEY: Final[str] = "pp_categorical_encoding"
PP_HC_AUTO_KEY: Final[str] = "pp_highcard_auto_downgrade"
PP_HC_THRESH_KEY: Final[str] = "pp_highcard_threshold"
PP_DT_DECOMPOSE_KEY: Final[str] = "pp_datetime_decompose"
PP_DT_PARTS_KEY: Final[str] = "pp_datetime_parts"
PP_BOOL_NUM_KEY: Final[str] = "pp_bool_as_numeric"
PP_IMBALANCE_KEY: Final[str] = "pp_imbalance"
PP_SMOTE_K_KEY: Final[str] = "pp_smote_k_neighbors"
PP_PREVIEW_BTN_KEY: Final[str] = "pp_preview_btn"
PP_PREVIEW_RESULT_KEY: Final[str] = "pp_preview_result"

# 전처리 옵션 튜플 (ml.schemas.Literal 과 키 1:1 대응)
_NUM_IMPUTE_OPTIONS: Final[tuple[str, ...]] = (
    "median",
    "mean",
    "most_frequent",
    "constant_zero",
)
_NUM_SCALE_OPTIONS: Final[tuple[str, ...]] = ("standard", "minmax", "robust", "none")
_OUTLIER_OPTIONS: Final[tuple[str, ...]] = ("none", "iqr_clip", "winsorize")
_CAT_IMPUTE_OPTIONS: Final[tuple[str, ...]] = ("most_frequent", "constant_missing")
_CAT_ENCODING_OPTIONS: Final[tuple[str, ...]] = ("onehot", "ordinal", "frequency")
_IMBALANCE_OPTIONS_CLASSIFICATION: Final[tuple[str, ...]] = ("none", "class_weight", "smote")
_IMBALANCE_OPTIONS_REGRESSION: Final[tuple[str, ...]] = ("none",)


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


def _render_optional_backend_notice() -> None:
    """xgboost / lightgbm 이 런타임 누락으로 스킵됐다면 사용자에게 사유를 한번 고지한다.

    조용히 후보에서 빠지면 "왜 성능 좋은 모델이 후보에 없지?" 혼란이 생기므로,
    `ml.registry.optional_backends_status()` 결과를 읽어 `st.info` 로 노출한다.
    """
    skipped = [s for s in optional_backends_status() if not s.available]
    if not skipped:
        return
    lines = [f"- **{s.name}** — {s.reason}" for s in skipped]
    st.info(
        "일부 부스팅 백엔드가 현재 환경에서 비활성화돼 학습 후보에서 제외됩니다. "
        "필요하면 아래 원인을 해결 후 앱을 재시작하세요.\n\n" + "\n".join(lines)
    )


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


def _imbalance_options(task_type: str) -> tuple[tuple[str, ...], str | None]:
    """task_type 과 imblearn 가용성에 따른 불균형 옵션 튜플 + 비활성 사유 반환.

    - 분류 + imblearn 설치: ("none", "class_weight", "smote"), 사유 None
    - 분류 + imblearn 미설치: ("none", "class_weight"), 사유 = SMOTE_UNAVAILABLE
    - 회귀: ("none",), 사유 = SMOTE_CLASSIFICATION_ONLY
    """
    if task_type == "regression":
        return _IMBALANCE_OPTIONS_REGRESSION, Msg.PREPROCESSING_SMOTE_CLASSIFICATION_ONLY
    if not ml_balancing.SMOTE_AVAILABLE:
        return (
            tuple(o for o in _IMBALANCE_OPTIONS_CLASSIFICATION if o != "smote"),
            Msg.PREPROCESSING_SMOTE_UNAVAILABLE,
        )
    return _IMBALANCE_OPTIONS_CLASSIFICATION, None


def _selectbox_index(options: tuple[str, ...], current: str, fallback: int = 0) -> int:
    """현재 값이 옵션에 있으면 그 인덱스, 없으면 fallback 반환 (옵션 축소 시 안전한 기본값)."""
    try:
        return options.index(current)
    except ValueError:
        return fallback


def _collect_preprocessing_config(task_type: str) -> PreprocessingConfig | None:
    """세션 상태의 전처리 위젯 값을 모아 ``PreprocessingConfig`` 로 조립.

    - 유효성 위반(예: datetime_decompose=True + 파트 미선택) 시 ``flash("error", ...)`` +
      ``None`` 반환 → 호출자는 학습 실행을 건너뛰도록 한다.
    - 모든 값이 기본값이면 여전히 ``PreprocessingConfig()`` 를 반환하지만,
      Service 는 `is_default` 를 통해 `preprocessing_config.json` 저장/감사 로그를 생략한다.
    """
    state = st.session_state
    # 회귀일 때는 SMOTE 를 강제로 "none" 으로 낮춘다 (UI 옵션에서 smote 가 사라져도 세션이 남아있을 수 있음).
    imbalance_raw = state.get(PP_IMBALANCE_KEY, "none")
    if task_type == "regression" and imbalance_raw == "smote":
        imbalance_raw = "none"
        state[PP_IMBALANCE_KEY] = "none"

    dt_parts_raw = tuple(state.get(PP_DT_PARTS_KEY, ()) or ())

    try:
        return PreprocessingConfig(
            numeric_impute=state.get(PP_NUM_IMPUTE_KEY, "median"),
            numeric_scale=state.get(PP_NUM_SCALE_KEY, "standard"),
            outlier=state.get(PP_OUTLIER_KEY, "none"),
            outlier_iqr_k=float(state.get(PP_IQR_K_KEY, 1.5)),
            winsorize_p=float(state.get(PP_WINSORIZE_KEY, 0.01)),
            categorical_impute=state.get(PP_CAT_IMPUTE_KEY, "most_frequent"),
            categorical_encoding=state.get(PP_CAT_ENCODING_KEY, "onehot"),
            highcard_threshold=int(state.get(PP_HC_THRESH_KEY, 50)),
            highcard_auto_downgrade=bool(state.get(PP_HC_AUTO_KEY, True)),
            datetime_decompose=bool(state.get(PP_DT_DECOMPOSE_KEY, False)),
            datetime_parts=dt_parts_raw,
            bool_as_numeric=bool(state.get(PP_BOOL_NUM_KEY, True)),
            imbalance=imbalance_raw,
            smote_k_neighbors=int(state.get(PP_SMOTE_K_KEY, 5)),
        )
    except ValueError as err:
        flash("error", str(err))
        return None


def _render_preview_result(preview: FeaturePreviewDTO) -> None:
    """미리보기 카드 렌더. n_cols_in / n_cols_out 메트릭 + 파생 테이블 + 다운그레이드 고지."""
    st.markdown(f"**{Msg.PREPROCESSING_PREVIEW_TITLE}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("원본 열", preview.n_cols_in)
    c2.metric("변환 후 열", preview.n_cols_out)
    c3.metric("파생 피처", len(preview.derived))

    if preview.auto_downgraded:
        st.info(
            f"{Msg.PREPROCESSING_PREVIEW_AUTO_DOWNGRADED} "
            f"({', '.join(preview.auto_downgraded)})"
        )

    if preview.derived:
        rows = [
            {"source": source, "kind": kind, "derived_name": name}
            for source, name, kind in preview.derived
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("파생 피처가 생성되지 않았습니다 (기본 수치 경로).")


def _render_numeric_section() -> None:
    """결측 · 스케일 · 이상치 섹션."""
    st.markdown("**결측 · 스케일 · 이상치**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox(
            "수치 결측 대치",
            options=_NUM_IMPUTE_OPTIONS,
            index=_selectbox_index(
                _NUM_IMPUTE_OPTIONS,
                st.session_state.get(PP_NUM_IMPUTE_KEY, "median"),
            ),
            key=PP_NUM_IMPUTE_KEY,
        )
    with c2:
        st.selectbox(
            "수치 스케일",
            options=_NUM_SCALE_OPTIONS,
            index=_selectbox_index(
                _NUM_SCALE_OPTIONS,
                st.session_state.get(PP_NUM_SCALE_KEY, "standard"),
            ),
            key=PP_NUM_SCALE_KEY,
        )
    with c3:
        st.selectbox(
            "이상치 처리",
            options=_OUTLIER_OPTIONS,
            index=_selectbox_index(
                _OUTLIER_OPTIONS,
                st.session_state.get(PP_OUTLIER_KEY, "none"),
            ),
            key=PP_OUTLIER_KEY,
        )

    outlier_choice = st.session_state.get(PP_OUTLIER_KEY, "none")
    if outlier_choice == "iqr_clip":
        st.number_input(
            "IQR 배수 (k)",
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            value=float(st.session_state.get(PP_IQR_K_KEY, 1.5)),
            key=PP_IQR_K_KEY,
        )
    elif outlier_choice == "winsorize":
        st.number_input(
            "Winsorize 비율 (p)",
            min_value=0.001,
            max_value=0.2,
            step=0.005,
            format="%0.3f",
            value=float(st.session_state.get(PP_WINSORIZE_KEY, 0.01)),
            key=PP_WINSORIZE_KEY,
        )


def _render_categorical_section() -> None:
    """범주형 결측 / 인코딩 / 고카디널리티 자동 강등."""
    st.markdown("**범주형**")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "범주 결측 대치",
            options=_CAT_IMPUTE_OPTIONS,
            index=_selectbox_index(
                _CAT_IMPUTE_OPTIONS,
                st.session_state.get(PP_CAT_IMPUTE_KEY, "most_frequent"),
            ),
            key=PP_CAT_IMPUTE_KEY,
        )
    with c2:
        st.selectbox(
            "인코딩",
            options=_CAT_ENCODING_OPTIONS,
            index=_selectbox_index(
                _CAT_ENCODING_OPTIONS,
                st.session_state.get(PP_CAT_ENCODING_KEY, "onehot"),
            ),
            key=PP_CAT_ENCODING_KEY,
        )
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox(
            "고카디널리티 자동 감지",
            value=bool(st.session_state.get(PP_HC_AUTO_KEY, True)),
            key=PP_HC_AUTO_KEY,
            help="onehot 선택 시 nunique 가 임계값을 초과하는 컬럼을 frequency 로 자동 강등합니다.",
        )
    with c2:
        st.number_input(
            "고카디널리티 임계 (nunique)",
            min_value=2,
            max_value=10000,
            step=1,
            value=int(st.session_state.get(PP_HC_THRESH_KEY, 50)),
            key=PP_HC_THRESH_KEY,
        )


def _render_advanced_types_section() -> None:
    """datetime 분해 + bool 수치 통과 옵션."""
    st.markdown("**고급 타입 (datetime · bool)**")
    st.caption("datetime 컬럼은 학습 직전 자동 감지됩니다. 미리보기로 확인하세요.")
    st.checkbox(
        "datetime 컬럼 분해",
        value=bool(st.session_state.get(PP_DT_DECOMPOSE_KEY, False)),
        key=PP_DT_DECOMPOSE_KEY,
    )
    if st.session_state.get(PP_DT_DECOMPOSE_KEY, False):
        st.multiselect(
            "datetime 파트",
            options=DEFAULT_DATETIME_PARTS,
            default=list(st.session_state.get(PP_DT_PARTS_KEY, DEFAULT_DATETIME_PARTS))
            or list(DEFAULT_DATETIME_PARTS),
            key=PP_DT_PARTS_KEY,
        )
    st.checkbox(
        "bool 컬럼을 수치(0/1) 로 통과",
        value=bool(st.session_state.get(PP_BOOL_NUM_KEY, True)),
        key=PP_BOOL_NUM_KEY,
        help="끄면 bool 컬럼도 범주형 인코딩 경로로 합류합니다.",
    )


def _render_imbalance_section(task_type: str) -> None:
    """불균형 대응 전략 선택 (분류 + imblearn 설치 시만 smote 노출)."""
    st.markdown("**불균형 대응**")
    options, disabled_reason = _imbalance_options(task_type)
    if disabled_reason:
        st.caption(disabled_reason)
    current = st.session_state.get(PP_IMBALANCE_KEY, "none")
    if current not in options:
        current = "none"
        st.session_state[PP_IMBALANCE_KEY] = "none"
    st.radio(
        "전략",
        options=options,
        index=options.index(current),
        key=PP_IMBALANCE_KEY,
        horizontal=True,
    )
    if st.session_state.get(PP_IMBALANCE_KEY) == "smote":
        st.number_input(
            "SMOTE k_neighbors",
            min_value=1,
            max_value=20,
            step=1,
            value=int(st.session_state.get(PP_SMOTE_K_KEY, 5)),
            key=PP_SMOTE_K_KEY,
        )


def _handle_preview_click(task_type: str, dataset_id: int) -> None:
    """미리보기 버튼 클릭 시: TrainingConfig 조립 → preview_preprocessing 호출."""
    pp_cfg = _collect_preprocessing_config(task_type)
    target = st.session_state.get(FORM_TARGET_KEY)
    excluded = tuple(st.session_state.get(FORM_EXCLUDED_KEY, ()) or ())
    if pp_cfg is None or not target:
        set_state(PP_PREVIEW_RESULT_KEY, None)
        return
    try:
        preview_cfg = TrainingConfig(
            dataset_id=dataset_id,
            task_type=task_type,  # type: ignore[arg-type]
            target_column=str(target),
            excluded_columns=excluded,
            preprocessing=pp_cfg,
        )
        preview = training_service.preview_preprocessing(dataset_id, preview_cfg)
        set_state(PP_PREVIEW_RESULT_KEY, preview)
    except (AppError, ValueError) as err:
        flash("error", str(err))
        set_state(PP_PREVIEW_RESULT_KEY, None)


def _render_advanced_preprocessing_expander(
    task_type: str, dataset_id: int
) -> PreprocessingConfig | None:
    """§9.9: 고급 전처리 expander + 미리보기 버튼.

    반환값은 현재 위젯 상태로 조립된 ``PreprocessingConfig``. 값 조합이 잘못되면 ``None``.
    """
    # 커스텀 여부를 expander 라벨에 반영. expander 닫힌 상태에서도 "변경됨" 시각 단서 제공.
    expander_label = Msg.PREPROCESSING_ADVANCED_TITLE
    current_cfg = _collect_preprocessing_config(task_type)
    if current_cfg is not None and not current_cfg.is_default:
        expander_label = f"{expander_label} · {Msg.PREPROCESSING_CUSTOM_BADGE}"

    with st.expander(expander_label, expanded=False):
        st.caption("모든 옵션은 기본값에서 변경된 축만 저장/감사 로그에 기록됩니다.")
        _render_numeric_section()
        _render_categorical_section()
        _render_advanced_types_section()
        _render_imbalance_section(task_type)

        st.divider()
        st.caption(Msg.PREPROCESSING_PREVIEW_HINT)
        if st.button("미리보기", key=PP_PREVIEW_BTN_KEY):
            _handle_preview_click(task_type, dataset_id)

        preview = get_state(PP_PREVIEW_RESULT_KEY)
        if preview is not None:
            _render_preview_result(preview)

    return _collect_preprocessing_config(task_type)


def _render_config_form(
    profile: DatasetProfileDTO, suggested_excluded: list[str], dataset_id: int
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

    # §9.9: 고급 전처리 expander + 미리보기.
    pp_cfg = _render_advanced_preprocessing_expander(str(task_type), dataset_id)
    if pp_cfg is not None and not pp_cfg.is_default:
        st.caption(Msg.PREPROCESSING_CUSTOM_BADGE)

    submitted = st.button("학습 실행", type="primary", key="training_submit_btn")
    if not submitted:
        return None

    # 위젯 조합이 유효하지 않으면 (flash 는 이미 큐잉됨) 학습 실행을 건너뛴다.
    if pp_cfg is None:
        st.rerun()
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
            preprocessing=pp_cfg if not pp_cfg.is_default else None,
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
    _render_optional_backend_notice()
    suggested = _load_suggested_excluded(current_dataset_id)
    config = _render_config_form(profile, suggested, current_dataset_id)

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
    render_help("03_training")

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
