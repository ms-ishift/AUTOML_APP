"""Prediction Service (IMPLEMENTATION_PLAN §4.5, FR-080~085, §10.4).

책임:
- 저장된 모델 아티팩트를 불러와 **단건(form) / 파일(batch)** 예측을 수행한다.
- 입력 스키마 검증(``ml.artifacts.validate_prediction_input``) + 도메인 규칙(§10.4: 누락 차단, 추가 컬럼 경고) 을 모두 통과시킨다.
- ``PredictionJob`` 을 기록하고 (batch 는 결과 CSV 를 ``storage/predictions/<job_id>.csv`` 로 저장).
- 반환은 ``PredictionResultDTO`` — UI 는 이 DTO 만을 소비.

규약 (``.cursor/rules/service-layer.mdc``):
- Streamlit 타입/함수 참조 금지
- ``utils.errors`` 계층의 도메인 예외로 변환 (``PredictionInputError``/``NotFoundError``/``StorageError``)
- 실패 경로에서 PredictionJob 은 ``failed`` 로 상태 전이 + 감사 로그
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from config.settings import settings
from ml.artifacts import load_model_bundle, validate_prediction_input
from repositories import (
    audit_repository,
    model_repository,
    prediction_repository,
    training_repository,
)
from repositories.base import session_scope
from services.dto import PredictionResultDTO
from utils.errors import (
    AppError,
    NotFoundError,
    PredictionInputError,
    StorageError,
    ValidationError,
)
from utils.events import Event
from utils.file_utils import read_tabular
from utils.log_utils import get_logger, log_event
from utils.messages import Msg, entity_not_found

if TYPE_CHECKING:
    import numpy as np

    from ml.artifacts import ModelBundle

logger = get_logger(__name__)

PREDICTION_COLUMN = "prediction"
PROBABILITY_PREFIX = "prob_"
BATCH_PREVIEW_MAX_ROWS = 1000


# ---------------------------------------------------------------- internals


def _load_model_context(model_id: int) -> tuple[ModelBundle, str, int]:
    """Model 상세 조회 + ModelBundle 로드 + 상위 TrainingJob.task_type 획득.

    반환: ``(bundle, task_type, model_id)``. ``bundle.metrics`` 에 ``task_type`` 이 있으면
    그걸 신뢰하고 (불필요한 DB 왕복 제거), 없으면 DB 에서 보조 조회.
    """
    with session_scope() as session:
        model = model_repository.get(session, model_id)
        if model is None:
            raise NotFoundError(entity_not_found("모델", model_id))
        if not model.model_path:
            raise NotFoundError(
                f"저장된 아티팩트가 없는 모델입니다 (model_id={model_id}).",
            )
        training_job = training_repository.get(session, model.training_job_id)
        if training_job is None:  # pragma: no cover - FK 보장
            raise NotFoundError(entity_not_found("학습 잡", model.training_job_id))
        task_type = training_job.task_type
        model_dir = Path(model.model_path).parent

    try:
        bundle = load_model_bundle(model_dir)
    except FileNotFoundError as exc:
        raise StorageError(
            f"모델 아티팩트를 찾을 수 없습니다 (dir={model_dir}).",
            cause=exc,
        ) from exc
    # bundle.metrics 이 없거나 task_type 키가 없을 수 있음 → DB 값을 신뢰
    return bundle, task_type, model_id


def _collect_input_warnings(df: pd.DataFrame, bundle: ModelBundle) -> list[str]:
    """추가 컬럼 + unseen 카테고리 경고 (§10.4, FR-083 보조 피드백).

    - 누락 컬럼은 ``validate_prediction_input`` 이 ValueError 로 차단 → 여기서는 수집 안 함.
    - 추가 컬럼은 단순히 무시되지만 사용자에게는 명시적으로 알린다.
    - OneHotEncoder(handle_unknown='ignore') 가 처리하지만 어떤 값이 unseen 인지 알리면
      사용자가 입력 품질을 점검할 수 있다.
    """
    warnings: list[str] = []
    expected = set(bundle.schema.input_columns)
    extras = [str(c) for c in df.columns if str(c) not in expected]
    if extras:
        warnings.append(f"학습에 사용되지 않은 컬럼은 무시합니다: {', '.join(extras)}")

    for col, seen_values in bundle.schema.categories.items():
        if col not in df.columns:
            continue
        seen_set = set(map(str, seen_values))
        values = df[col].dropna().astype(str).unique().tolist()
        unseen = [v for v in values if v not in seen_set]
        if unseen:
            # 너무 길면 앞 10개만 표시
            preview = ", ".join(unseen[:10]) + (" ..." if len(unseen) > 10 else "")
            warnings.append(f"학습 시 보지 못한 '{col}' 값이 포함되어 있습니다: {preview}")
    return warnings


def _run_predict(
    bundle: ModelBundle,
    cleaned: pd.DataFrame,
    task_type: str,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """``estimator.predict`` 호출 + (분류일 때) ``predict_proba`` 보조.

    반환: ``(예측값, [(클래스명, 확률배열), ...])``. 확률 계산 실패는 경고 없이 빈 리스트.
    """
    try:
        y_pred = bundle.estimator.predict(cleaned)
    except Exception as exc:  # noqa: BLE001 - 예측 실행 실패는 도메인 예외로 변환
        raise PredictionInputError(
            f"예측 실행 중 오류가 발생했습니다: {exc}",
            cause=exc,
        ) from exc

    proba_columns: list[tuple[str, np.ndarray]] = []
    if task_type == "classification" and hasattr(bundle.estimator, "predict_proba"):
        try:
            proba = bundle.estimator.predict_proba(cleaned)
            classes = getattr(bundle.estimator, "classes_", None)
            if classes is not None and proba.ndim == 2 and proba.shape[1] == len(classes):
                for idx, cls in enumerate(classes):
                    proba_columns.append((f"{PROBABILITY_PREFIX}{cls}", proba[:, idx]))
        except Exception:  # noqa: BLE001 - 확률은 보조 정보
            logger.debug("prediction.predict_proba_failed", extra={"task": task_type})

    return y_pred, proba_columns


def _build_result_df(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    proba_columns: list[tuple[str, np.ndarray]],
) -> pd.DataFrame:
    """원본 입력에 ``prediction`` + ``prob_*`` 컬럼을 붙인 결과 DataFrame."""
    out = df.copy()
    out[PREDICTION_COLUMN] = y_pred
    for name, values in proba_columns:
        out[name] = values
    return out


def _df_to_records(df: pd.DataFrame, *, limit: int | None) -> list[dict[str, Any]]:
    """DataFrame 을 JSON 직렬화 가능한 ``list[dict]`` 로. NaN → None."""
    if limit is not None:
        df = df.head(limit)
    import json as _json

    payload = df.to_json(orient="records", date_format="iso", force_ascii=False)
    return list(_json.loads(payload))


def _sanitize_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise PredictionInputError("단건 예측 입력이 비어 있습니다.")
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, float) and math.isnan(v):
            cleaned[str(k)] = None
        else:
            cleaned[str(k)] = v
    return cleaned


def _finalize_failure(
    prediction_job_id: int | None,
    model_id: int,
    input_type: str,
    error: AppError,
) -> None:
    """실패 경로: PredictionJob 을 failed 로 확정 + 감사 로그."""
    try:
        with session_scope() as session:
            if prediction_job_id is not None:
                prediction_repository.update_status(session, prediction_job_id, "failed")
            audit_repository.write(
                session,
                action_type=Event.PREDICTION_FAILED,
                target_type="PredictionJob",
                target_id=prediction_job_id or 0,
                detail={
                    "model_id": model_id,
                    "input_type": input_type,
                    "error": str(error),
                },
            )
    except Exception:  # pragma: no cover - 실패 기록 자체의 실패
        logger.exception(
            "prediction.fail_record_failed",
            extra={"model_id": model_id, "job_id": prediction_job_id},
        )
    log_event(
        logger,
        Event.PREDICTION_FAILED,
        model_id=model_id,
        input_type=input_type,
        error=str(error),
    )


def _insert_prediction_job(
    *,
    model_id: int,
    input_type: str,
    input_file_path: str | None,
) -> int:
    """PredictionJob insert + running 전이. 시작 감사 로그 기록."""
    with session_scope() as session:
        job = prediction_repository.insert(
            session,
            model_id=model_id,
            input_type=input_type,
            input_file_path=input_file_path,
            status="running",
        )
        audit_repository.write(
            session,
            action_type=Event.PREDICTION_STARTED,
            target_type="PredictionJob",
            target_id=job.prediction_job_id,
            detail={"model_id": model_id, "input_type": input_type},
        )
        log_event(
            logger,
            Event.PREDICTION_STARTED,
            job_id=job.prediction_job_id,
            model_id=model_id,
            input_type=input_type,
        )
        return job.prediction_job_id


def _complete_prediction_job(
    *,
    prediction_job_id: int,
    model_id: int,
    input_type: str,
    result_path: str | None,
    n_rows: int,
) -> None:
    with session_scope() as session:
        prediction_repository.update_status(
            session,
            prediction_job_id,
            "completed",
            result_path=result_path,
        )
        audit_repository.write(
            session,
            action_type=Event.PREDICTION_COMPLETED,
            target_type="PredictionJob",
            target_id=prediction_job_id,
            detail={
                "model_id": model_id,
                "input_type": input_type,
                "n_rows": n_rows,
                "result_path": result_path,
            },
        )
        log_event(
            logger,
            Event.PREDICTION_COMPLETED,
            job_id=prediction_job_id,
            model_id=model_id,
            input_type=input_type,
            n_rows=n_rows,
        )


def _validate_input(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """``ValueError`` → ``PredictionInputError`` 변환 래퍼."""
    try:
        return validate_prediction_input(df, bundle.schema)
    except ValueError as exc:
        raise PredictionInputError(str(exc), cause=exc) from exc


# --------------------------------------------------------- Public use-cases


def predict_single(model_id: int, payload: dict[str, Any]) -> PredictionResultDTO:
    """FR-080, FR-083: 입력 폼 dict → 단건 예측.

    ``payload`` 는 feature 컬럼명과 값의 매핑. 키 누락 시 ``PredictionInputError``.
    결과 CSV 는 생성하지 않고 DTO 에만 담아 반환한다.
    """
    cleaned_payload = _sanitize_payload(payload)
    bundle, task_type, _ = _load_model_context(model_id)
    df = pd.DataFrame([cleaned_payload])

    prediction_job_id: int | None = None
    try:
        warnings = _collect_input_warnings(df, bundle)
        cleaned = _validate_input(df, bundle)
        prediction_job_id = _insert_prediction_job(
            model_id=model_id, input_type="form", input_file_path=None
        )
        y_pred, proba_columns = _run_predict(bundle, cleaned, task_type)
        result_df = _build_result_df(df, y_pred, proba_columns)
        rows = _df_to_records(result_df, limit=None)
        _complete_prediction_job(
            prediction_job_id=prediction_job_id,
            model_id=model_id,
            input_type="form",
            result_path=None,
            n_rows=len(rows),
        )
        return PredictionResultDTO(
            job_id=prediction_job_id,
            rows=rows,
            result_path=None,
            warnings=warnings,
        )
    except AppError as err:
        _finalize_failure(prediction_job_id, model_id, "form", err)
        raise
    except Exception as exc:  # 예기치 못한 실패도 도메인 예외로 포장
        wrapped = PredictionInputError(Msg.PREDICTION_FAILED, cause=exc)
        _finalize_failure(prediction_job_id, model_id, "form", wrapped)
        raise wrapped from exc


def predict_batch(model_id: int, file_path: Path | str) -> PredictionResultDTO:
    """FR-081, FR-083, FR-084, FR-085: 업로드 파일 → 대량 예측.

    처리 순서:
    1) 입력 파일 파싱 (read_tabular: 헤더/빈파일/중복컬럼 검증 포함)
    2) 추가/unseen 경고 수집
    3) 스키마 검증(누락 컬럼 차단)
    4) PredictionJob insert(running)
    5) 예측 + 결과 DataFrame 구성
    6) ``<predictions_dir>/<job_id>.csv`` 저장
    7) PredictionJob completed + 감사 로그

    UI 는 DTO 의 ``rows`` (상위 ``BATCH_PREVIEW_MAX_ROWS`` 행) 를 표로 그리고,
    ``result_path`` 로 원본 CSV 를 다운로드한다 (FR-085).
    """
    source_path = Path(file_path)
    if not source_path.exists():
        raise ValidationError(Msg.FILE_PARSE_FAILED)

    bundle, task_type, _ = _load_model_context(model_id)

    prediction_job_id: int | None = None
    try:
        df = read_tabular(source_path)
        warnings = _collect_input_warnings(df, bundle)
        cleaned = _validate_input(df, bundle)

        prediction_job_id = _insert_prediction_job(
            model_id=model_id,
            input_type="file",
            input_file_path=str(source_path),
        )

        y_pred, proba_columns = _run_predict(bundle, cleaned, task_type)
        result_df = _build_result_df(df, y_pred, proba_columns)

        predictions_dir = settings.predictions_dir
        predictions_dir.mkdir(parents=True, exist_ok=True)
        result_path = predictions_dir / f"{prediction_job_id}.csv"
        try:
            result_df.to_csv(result_path, index=False)
        except OSError as exc:
            raise StorageError(
                f"예측 결과 파일을 저장하지 못했습니다: {result_path}",
                cause=exc,
            ) from exc

        _complete_prediction_job(
            prediction_job_id=prediction_job_id,
            model_id=model_id,
            input_type="file",
            result_path=str(result_path),
            n_rows=len(result_df),
        )
        return PredictionResultDTO(
            job_id=prediction_job_id,
            rows=_df_to_records(result_df, limit=BATCH_PREVIEW_MAX_ROWS),
            result_path=str(result_path),
            warnings=warnings,
        )
    except AppError as err:
        _finalize_failure(prediction_job_id, model_id, "file", err)
        raise
    except Exception as exc:
        wrapped = PredictionInputError(Msg.PREDICTION_FAILED, cause=exc)
        _finalize_failure(prediction_job_id, model_id, "file", wrapped)
        raise wrapped from exc


__all__ = [
    "BATCH_PREVIEW_MAX_ROWS",
    "PREDICTION_COLUMN",
    "PROBABILITY_PREFIX",
    "predict_single",
    "predict_batch",
]
