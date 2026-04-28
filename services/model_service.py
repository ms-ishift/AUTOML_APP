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
from ml.artifacts import load_model_bundle
from ml.feature_influence import (
    DEFAULT_MAX_ROWS,
    DEFAULT_N_REPEATS,
    compute_permutation_importance,
    extract_builtin_transformed_importances,
    scoring_for_permutation,
)
from ml.schemas import TrainingConfig
from ml.trainers import split_dataset
from repositories import audit_repository, dataset_repository, model_repository, training_repository
from repositories.base import session_scope
from services.dto import (
    FeatureInfluenceBuiltinRowDTO,
    FeatureInfluencePermutationRowDTO,
    FeatureInfluenceResultDTO,
    FeatureSchemaDTO,
    ModelDetailDTO,
    ModelDTO,
)
from services.training_service import (
    assert_valid_training_target,
    rebuild_xy_for_influence_analysis,
)
from utils.errors import AppError, NotFoundError, StorageError, ValidationError
from utils.events import Event
from utils.file_utils import read_tabular
from utils.log_utils import get_logger, log_event
from utils.messages import Msg, entity_not_found

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


def _load_influence_bundle_and_config(  # noqa: C901 — DB 조회 + 번들/설정/DataFrame 로드 오케스트레이션
    model_id: int,
) -> tuple[Any, TrainingConfig, Any, int, str, str, str]:
    """DB+디스크에서 모델 번들, ``TrainingConfig``, 원시 DataFrame 을 로드."""
    with session_scope() as session:
        model = model_repository.get(session, model_id)
        if model is None:
            raise NotFoundError(entity_not_found("모델", model_id))
        if not model.model_path:
            raise NotFoundError(
                entity_not_found("모델 아티팩트", model_id) + " (학습 실패 행에는 파일이 없습니다.)"
            )
        job = training_repository.get(session, model.training_job_id)
        if job is None:
            raise NotFoundError(entity_not_found("학습 잡", model.training_job_id))
        dataset = dataset_repository.get(session, job.dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", job.dataset_id))
        model_dir = Path(model.model_path).parent
        dataset_path = Path(dataset.file_path)
        job_id = int(job.training_job_id)
        dataset_id = int(job.dataset_id)
        task_type = str(job.task_type)
        target_column = str(job.target_column)
        metric_key = str(job.metric_key)
        excluded_columns = tuple(job.excluded_columns_json or ())
        algorithm_name = str(model.algorithm_name)

    if not model_dir.is_dir():
        raise NotFoundError(f"모델 디렉터리가 없습니다: {model_dir}")
    if not dataset_path.exists():
        raise StorageError(Msg.FILE_PARSE_FAILED)

    try:
        bundle = load_model_bundle(model_dir)
    except FileNotFoundError as exc:
        raise NotFoundError(f"모델 번들을 읽을 수 없습니다 (model_id={model_id}).") from exc

    metrics = bundle.metrics or {}
    n_train = int(metrics.get("n_train") or 0)
    n_test_m = int(metrics.get("n_test") or 0)
    if n_train > 0 and n_test_m > 0:
        test_size = n_test_m / (n_train + n_test_m)
    else:
        test_size = settings.DEFAULT_TEST_SIZE

    pp = bundle.preprocessing
    preprocessing = None if pp.is_default else pp
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type=task_type,  # type: ignore[arg-type]
        target_column=target_column,
        excluded_columns=excluded_columns,
        test_size=float(test_size),
        metric_key=metric_key,
        preprocessing=preprocessing,
    )

    try:
        df = read_tabular(dataset_path)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StorageError(Msg.FILE_PARSE_FAILED, cause=exc) from exc

    return bundle, config, df, job_id, algorithm_name, task_type, metric_key


def get_feature_influence(model_id: int) -> FeatureInfluenceResultDTO:
    """FR-094, FR-095: 저장된 모델에 대해 테스트 분할 부분 표본으로 순열 중요도 및 내장 트리 중요도.

    - 테스트 분할: ``metrics.json`` 의 ``n_train``/``n_test`` 로 ``test_size`` 를 복원한 뒤
      ``split_dataset`` 으로 학습 시와 동일한 ``random_state`` 분할을 재현한다.
    - 순열 평가 행 상한: ``ml.feature_influence.DEFAULT_MAX_ROWS``.
    """
    bundle, config, df, job_id, algorithm_name, task_type, metric_key = (
        _load_influence_bundle_and_config(model_id)
    )

    assert_valid_training_target(df, config)
    X, y = rebuild_xy_for_influence_analysis(df, config)
    _x_train, X_test, _y_train, y_test = split_dataset(
        X, y, test_size=config.test_size, task_type=config.task_type
    )

    scoring = scoring_for_permutation(task_type, metric_key)
    try:
        rows_raw, n_used, n_test_total = compute_permutation_importance(
            bundle.estimator,
            X_test,
            y_test,
            task_type=task_type,
            metric_key=metric_key,
            n_repeats=DEFAULT_N_REPEATS,
            random_state=settings.RANDOM_SEED,
            n_jobs=1,
            max_rows=DEFAULT_MAX_ROWS,
        )
    except ValueError as exc:
        raise ValidationError(Msg.INFLUENCE_FAILED, cause=exc) from exc

    perm_rows = tuple(
        FeatureInfluencePermutationRowDTO(feature_name=name, permutation_mean=m, permutation_std=s)
        for name, m, s in rows_raw
    )
    built = extract_builtin_transformed_importances(bundle.estimator)
    builtin_rows = tuple(
        FeatureInfluenceBuiltinRowDTO(feature_name=n, importance=v) for n, v in (built or [])
    )

    with session_scope() as session:
        audit_repository.write(
            session,
            action_type=Event.MODEL_INFLUENCE_COMPUTED,
            target_type="Model",
            target_id=model_id,
            detail={
                "training_job_id": job_id,
                "algorithm_name": algorithm_name,
                "n_rows_used": n_used,
                "n_test_rows": n_test_total,
                "n_builtin": len(builtin_rows),
                "scoring": scoring,
            },
        )
    log_event(
        logger,
        Event.MODEL_INFLUENCE_COMPUTED,
        model_id=model_id,
        training_job_id=job_id,
        n_rows_used=n_used,
    )

    return FeatureInfluenceResultDTO(
        permutation_rows=perm_rows,
        builtin_rows=builtin_rows,
        n_rows_used=n_used,
        n_test_rows=n_test_total,
        scoring=scoring,
    )


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
    "get_feature_influence",
    "save_model",
    "delete_model",
    "find_best_model",
    "get_model_plot_data",
]
