"""Model Service (IMPLEMENTATION_PLAN §4.4, FR-073~075).

책임:
- 프로젝트 소속 모델의 목록/상세 조회, 수동 저장(= ``is_best`` 핀 고정), 삭제.
- 학습 시점(``training_service.run_training``)에 이미 모든 모델은 DB/디스크에 저장되므로
  이 서비스는 **사후 관리(view/pin/delete)** 를 담당한다.

규약 (``.cursor/rules/service-layer.mdc``):
- Streamlit 타입 금지, ORM 은 DTO 변환 후 반환
- 사용자 문구는 ``utils.messages.Msg`` / ``entity_not_found`` 에서만 가져옴
- 파일 정리(models 디렉터리 삭제)는 DB 트랜잭션 커밋 이후 best-effort 로 수행
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import settings
from repositories import audit_repository, model_repository, training_repository
from repositories.base import session_scope
from services.dto import FeatureSchemaDTO, ModelDetailDTO, ModelDTO
from utils.errors import NotFoundError
from utils.events import Event
from utils.log_utils import get_logger, log_event
from utils.messages import entity_not_found

if TYPE_CHECKING:
    from repositories.models import Model

logger = get_logger(__name__)


# ---------------------------------------------------------------- internals


def _feature_schema_dto(data: dict[str, Any] | None) -> FeatureSchemaDTO:
    """``Model.feature_schema_json`` → ``FeatureSchemaDTO``.

    학습 시 실패 모델은 ``feature_schema_json`` 이 ``None`` 으로 저장되므로
    이 경우 빈 스키마를 돌려준다 (UI 는 ``input_columns`` 공집합으로 대응).
    """
    data = data or {}
    return FeatureSchemaDTO(
        numeric=list(data.get("numeric", [])),
        categorical=list(data.get("categorical", [])),
        target=str(data.get("target", "")),
        categories={k: list(v) for k, v in (data.get("categories") or {}).items()},
    )


def _metrics_summary_dict(data: dict[str, Any] | None) -> dict[str, float]:
    """``metric_summary_json`` → 순수 metric 맵. success/failed 포장은 벗긴다."""
    data = data or {}
    metrics = data.get("metrics") or {}
    cleaned: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            cleaned[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return cleaned


def _model_dir(model_id: int) -> Path:
    return settings.models_dir / str(model_id)


def _artifacts_exist(model: Model) -> bool:
    """아티팩트 파일이 실제로 있는지 확인.

    학습 실패 모델은 ``model_path`` 가 ``None`` (DB 에만 존재). 저장 가능 여부 판별에 사용.
    """
    if not model.model_path:
        return False
    return Path(model.model_path).exists()


# --------------------------------------------------------- Public use-cases


def list_models(project_id: int) -> list[ModelDTO]:
    """FR-075: 프로젝트 소속 모델 목록(최신순).

    학습 실패 행(metric_score=None, feature_schema_json=None) 도 함께 포함한다.
    UI 는 상태별 필터링을 자체적으로 처리한다.
    """
    with session_scope() as session:
        rows = model_repository.list_by_project(session, project_id)
        return [ModelDTO.from_orm(m) for m in rows]


def get_model_detail(model_id: int) -> ModelDetailDTO:
    """FR-074: 단건 상세. ``feature_schema`` + ``metrics_summary`` 를 DB JSON 에서 재구성."""
    with session_scope() as session:
        model = model_repository.get(session, model_id)
        if model is None:
            raise NotFoundError(entity_not_found("모델", model_id))
        return ModelDetailDTO(
            base=ModelDTO.from_orm(model),
            feature_schema=_feature_schema_dto(model.feature_schema_json),
            metrics_summary=_metrics_summary_dict(model.metric_summary_json),
        )


def save_model(model_id: int) -> ModelDTO:
    """FR-073: 사용자가 지정한 모델을 **저장된 최적 모델**로 수동 고정.

    구현 세부:
    - 동일 TrainingJob 내에서 ``model_repository.mark_best`` 로 is_best=True 로 전환
      (다른 모델의 is_best 는 모두 False 로 초기화됨).
    - 아티팩트 파일이 실제로 존재해야 저장 대상이 될 수 있다 (학습 실패 모델 차단).
    - 감사 로그 ``model.saved`` 에 ``{"manual": True}`` 로 구분.
    """
    with session_scope() as session:
        model = model_repository.get(session, model_id)
        if model is None:
            raise NotFoundError(entity_not_found("모델", model_id))
        if not _artifacts_exist(model):
            raise NotFoundError(
                f"저장된 아티팩트가 없는 모델은 선택할 수 없습니다 (model_id={model_id}).",
            )

        updated = model_repository.mark_best(session, model.training_job_id, model_id)
        if updated is None:  # pragma: no cover - mark_best 는 FK 로 안정적이지만 방어 코드
            raise NotFoundError(entity_not_found("모델", model_id))

        audit_repository.write(
            session,
            action_type=Event.MODEL_SAVED,
            target_type="Model",
            target_id=model_id,
            detail={
                "manual": True,
                "training_job_id": model.training_job_id,
                "algorithm_name": model.algorithm_name,
            },
        )
        log_event(
            logger,
            Event.MODEL_SAVED,
            model_id=model_id,
            training_job_id=model.training_job_id,
            manual=True,
        )
        return ModelDTO.from_orm(updated)


def delete_model(model_id: int) -> None:
    """모델 삭제. DB 레코드 제거 후 ``<models_dir>/<model_id>/`` 디렉터리를 정리.

    ORM cascade 로 연결된 ``PredictionJob`` 도 함께 삭제된다. 예측 결과 CSV 도 best-effort 로 정리.
    """
    with session_scope() as session:
        model = model_repository.get(session, model_id)
        if model is None:
            raise NotFoundError(entity_not_found("모델", model_id))

        training_job_id = model.training_job_id
        algorithm_name = model.algorithm_name

        prediction_result_paths: list[Path] = [
            Path(pj.result_path) for pj in model.prediction_jobs if pj.result_path
        ]
        if model.model_path:
            prediction_result_paths.append(Path(model.model_path))

        model_repository.delete(session, model_id)
        audit_repository.write(
            session,
            action_type=Event.MODEL_DELETED,
            target_type="Model",
            target_id=model_id,
            detail={
                "training_job_id": training_job_id,
                "algorithm_name": algorithm_name,
            },
        )
        log_event(
            logger,
            Event.MODEL_DELETED,
            model_id=model_id,
            training_job_id=training_job_id,
        )

    model_dir = _model_dir(model_id)
    _cleanup_model_assets(model_dir, prediction_result_paths)


def _cleanup_model_assets(model_dir: Path, extra_files: list[Path]) -> None:
    """커밋 이후의 파일 정리. 롤백 시 파일이 남더라도 DB 무결성에는 영향이 없다."""
    try:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
    except OSError:
        logger.exception("model.cleanup_dir_failed", extra={"path": str(model_dir)})

    for p in extra_files:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            logger.exception("model.cleanup_file_failed", extra={"path": str(p)})


# ---------------------------------------------------- helpers for others


def get_model_plot_data(model_id: int) -> dict[str, Any] | None:
    """FR-071/FR-072: 결과 비교 페이지용 플롯 데이터를 로드.

    학습 시 ``_persist_and_save`` 가 성공 모델에만 ``plot_data.json`` 을 남기므로
    실패 모델 / 파일이 없는 경우 ``None`` 을 반환한다 (UI 가 "플롯 없음" 안내).
    잘못된 JSON 은 역시 ``None``. 페이지가 플롯 없이도 렌더되어야 하기 때문.
    """
    path = _model_dir(model_id) / "plot_data.json"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        logger.debug("model.plot_data.read_failed", extra={"model_id": model_id})
        return None
    if not isinstance(data, dict):
        return None
    return data


def find_best_model(training_job_id: int) -> ModelDTO | None:
    """TrainingJob 의 현재 best 모델 (테스트/UI 편의용 조회 헬퍼).

    없으면 ``None``. 서비스 경계에서 is_best 가 0개인 실패 Job 을 쉽게 다룰 수 있도록 노출.
    """
    with session_scope() as session:
        job = training_repository.get(session, training_job_id)
        if job is None:
            return None
        for m in model_repository.list_by_training_job(session, training_job_id):
            if m.is_best:
                return ModelDTO.from_orm(m)
    return None


__all__ = [
    "list_models",
    "get_model_detail",
    "save_model",
    "delete_model",
    "find_best_model",
    "get_model_plot_data",
]
