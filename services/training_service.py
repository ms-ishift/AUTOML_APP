"""Training Service (IMPLEMENTATION_PLAN §4.3, FR-060~066, §4.3a).

책임:
- ``TrainingConfig`` 를 받아 전체 학습 유스케이스를 오케스트레이션한다.
  (프로젝트/데이터셋 조회 → 전처리 → split → 다중 학습 → 평가 → best 선정 → DB/파일 저장)
- ``TrainingJob`` 의 상태 전이와 ``run_log`` append 를 소유.
- 모델 레코드는 **성공·실패 전부** 저장하지만, **아티팩트 파일은 성공 모델만** 저장한다.
- 진행률 콜백 ``on_progress(stage, ratio)`` 를 동기적으로 호출 (Streamlit rerun 모델 호환).

§4.3a 보상 로직 (파일 저장 실패 → 전체 롤백):
1) ``model_repository.bulk_insert`` → ``session.flush()`` 로 ``model_id`` 확정
2) ``save_model_bundle(<models_dir>/<model_id>)`` 순차 저장, 성공 경로 누적
3) 중간 실패 시: 지금까지 쓴 디렉터리 정리 → ``session.rollback()`` (``session_scope`` 가 자동)
   → ``TrainingJob.status = 'failed'`` 를 **별도 트랜잭션**으로 저장 + ``MLTrainingError`` raise

전역 규칙 (``.cursor/rules/service-layer.mdc``):
- Streamlit 타입 import 금지
- ORM 은 UI 에 직접 노출하지 않음 (DTO 변환)
- 사용자 노출 문구는 ``utils.messages.Msg`` 에서만 가져옴
"""

from __future__ import annotations

import json
import shutil
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import settings
from ml.artifacts import save_model_bundle
from ml.evaluators import (
    METRIC_DIRECTIONS,
    build_plot_data,
    score_models,
    select_best,
)
from ml.preprocess import (
    build_feature_schema,
    build_preprocessor,
    prepare_xy,
    split_feature_types,
)
from ml.registry import get_specs
from ml.trainers import split_dataset, train_all
from repositories import (
    audit_repository,
    dataset_repository,
    model_repository,
    training_repository,
)
from repositories.base import session_scope
from services.dto import (
    ModelComparisonRowDTO,
    TrainingJobDTO,
    TrainingResultDTO,
)
from utils.errors import (
    AppError,
    MLTrainingError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from utils.events import Event
from utils.file_utils import read_tabular
from utils.log_utils import get_logger, log_event
from utils.messages import Msg, entity_not_found

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml.schemas import FeatureSchema, ScoredModel, TrainingConfig
    from ml.trainers import TrainedModel

    ProgressCallback = Callable[[str, float], None]

logger = get_logger(__name__)

_ERROR_TRUNCATE_LEN = 500
# 학습 수명주기 단계 가중치. ratio 계산에 쓰인다.
_STAGE_RATIO = {
    "preprocessing": 0.05,
    "split": 0.10,
    "train_start": 0.20,
    "train_end": 0.70,
    "score": 0.80,
    "save": 0.90,
    "completed": 1.0,
}


# --------------------------------------------------------------------- helpers


def _emit(cb: ProgressCallback | None, stage: str, ratio: float) -> None:
    """콜백 호출. 콜백 내부 예외가 학습 전체를 중단시키지 않도록 격리."""
    if cb is None:
        return
    with suppress(Exception):
        cb(stage, max(0.0, min(1.0, ratio)))


def _append_log(session: Any, job_id: int, message: str) -> None:
    """run_log 에 타임스탬프 prefix 로 한 줄 append."""
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    training_repository.append_run_log(session, job_id, f"[{ts}] {message}")


def _resolve_metric_key(task_type: str, requested: str) -> str:
    """metric_key 확정. 명시값이 없으면 task 기본값, 알려진 키가 아니면 ``ValidationError``."""
    if requested:
        if requested not in METRIC_DIRECTIONS:
            raise ValidationError(
                f"허용되지 않는 평가 지표입니다: {requested} "
                f"(사용 가능: {', '.join(sorted(METRIC_DIRECTIONS))})"
            )
        return requested
    specs = get_specs(task_type)  # type: ignore[arg-type]
    if specs and specs[0].default_metric:
        return specs[0].default_metric
    return "f1" if task_type == "classification" else "rmse"


def _serialize_scored(scored: ScoredModel) -> dict[str, Any]:
    """DB ``metric_summary_json`` 에 저장할 구조. success/failed 구분 정보를 자체 포함."""
    if scored.is_success:
        return {
            "status": "success",
            "metrics": dict(scored.metrics),
            "train_time_ms": scored.train_time_ms,
        }
    return {
        "status": "failed",
        "metrics": {},
        "error": (scored.error or "")[:_ERROR_TRUNCATE_LEN],
        "train_time_ms": scored.train_time_ms,
    }


def _row_from_orm(model_entity: Any) -> ModelComparisonRowDTO:
    """ORM Model → ModelComparisonRowDTO. ``metric_summary_json`` 포맷을 역직렬화."""
    raw = model_entity.metric_summary_json or {}
    status = raw.get("status", "success" if raw.get("metrics") else "failed")
    metrics = dict(raw.get("metrics", {})) if status == "success" else {}
    error = raw.get("error") if status == "failed" else None
    return ModelComparisonRowDTO(
        algo_name=model_entity.algorithm_name,
        status=status,
        metrics=metrics,
        train_time_ms=int(raw.get("train_time_ms", 0) or 0),
        is_best=bool(model_entity.is_best),
        error=error,
        model_id=int(model_entity.model_id),
    )


def _fail_job(job_id: int, error: AppError) -> None:
    """학습 실패 시 별도 트랜잭션으로 상태/로그/감사를 기록."""
    try:
        with session_scope() as session:
            training_repository.update_status(session, job_id, "failed")
            _append_log(session, job_id, f"FAILED: {error}")
            audit_repository.write(
                session,
                action_type=Event.TRAINING_FAILED,
                target_type="TrainingJob",
                target_id=job_id,
                detail={"error": str(error)},
            )
    except Exception:  # pragma: no cover - 실패 처리 자체의 실패는 로깅만
        logger.exception("training.fail_job_write_failed", extra={"job_id": job_id})


def _cleanup_model_dirs(dirs: list[Path]) -> None:
    for d in dirs:
        with suppress(Exception):
            shutil.rmtree(d, ignore_errors=True)


def _metrics_payload(
    algo_name: str,
    metric_key: str,
    scored: ScoredModel,
    *,
    task_type: str,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """디스크 ``metrics.json`` 에 저장할 메타/성능 스냅샷."""
    import sklearn

    return {
        "algo_name": algo_name,
        "task_type": task_type,
        "metric_key": metric_key,
        "metric_score": scored.metrics.get(metric_key),
        "metrics": dict(scored.metrics),
        "train_time_ms": scored.train_time_ms,
        "n_train": n_train,
        "n_test": n_test,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "random_seed": settings.RANDOM_SEED,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def _persist_and_save(
    *,
    job_id: int,
    scored: list[ScoredModel],
    best: ScoredModel | None,
    trained_by_name: dict[str, TrainedModel],
    feature_schema: FeatureSchema,
    metric_key: str,
    task_type: str,
    n_train: int,
    n_test: int,
    saved_dirs: list[Path],
    plot_data_by_algo: dict[str, dict[str, Any]] | None = None,
) -> None:
    """§4.3a 보상 트랜잭션: Model insert → 파일 저장 → set_paths → best/상태 확정.

    외부 ``saved_dirs`` 리스트를 채워줘서, 호출자가 예외 시 파일 정리를 할 수 있도록 한다.
    예외 발생 시 ``session_scope`` 가 자동 롤백, 호출자는 ``saved_dirs`` 를 ``shutil.rmtree``.
    """
    with session_scope() as session:
        insert_rows = [
            {
                "model_name": sm.algo_name,
                "algorithm_name": sm.algo_name,
                "metric_score": sm.metrics.get(metric_key) if sm.is_success else None,
                "metric_summary_json": _serialize_scored(sm),
                "feature_schema_json": feature_schema.to_dict() if sm.is_success else None,
            }
            for sm in scored
        ]
        entities = model_repository.bulk_insert(session, job_id, insert_rows)
        model_by_name = {e.algorithm_name: e for e in entities}

        for sm in scored:
            if not sm.is_success:
                continue
            trained_model = trained_by_name.get(sm.algo_name)
            if trained_model is None or trained_model.estimator is None:
                continue
            entity = model_by_name[sm.algo_name]
            model_dir = settings.models_dir / str(entity.model_id)
            fitted_pipeline = trained_model.estimator
            fitted_pre = fitted_pipeline.named_steps["preprocessor"]

            save_model_bundle(
                model_dir,
                estimator=fitted_pipeline,
                preprocessor=fitted_pre,
                schema=feature_schema,
                metrics=_metrics_payload(
                    sm.algo_name,
                    metric_key,
                    sm,
                    task_type=task_type,
                    n_train=n_train,
                    n_test=n_test,
                ),
            )
            saved_dirs.append(model_dir)

            # 결과 비교 페이지(§6.4) 용 플롯 데이터 — 보조 파일이라 실패 시 무시
            plot_payload = (plot_data_by_algo or {}).get(sm.algo_name)
            if plot_payload is not None:
                with suppress(OSError, TypeError, ValueError):
                    (model_dir / "plot_data.json").write_text(
                        json.dumps(plot_payload, ensure_ascii=False),
                        encoding="utf-8",
                    )

            model_repository.update_paths(
                session,
                entity.model_id,
                model_path=str(model_dir / "model.joblib"),
                preprocessing_path=str(model_dir / "preprocessor.joblib"),
            )
            audit_repository.write(
                session,
                action_type=Event.MODEL_SAVED,
                target_type="Model",
                target_id=entity.model_id,
                detail={
                    "training_job_id": job_id,
                    "algo": sm.algo_name,
                    "metric_key": metric_key,
                    "metric_score": sm.metrics.get(metric_key),
                },
            )

        if best is not None:
            best_entity = model_by_name.get(best.algo_name)
            if best_entity is not None:
                model_repository.mark_best(session, job_id, best_entity.model_id)

        any_success = any(sm.is_success for sm in scored)
        final_status = "completed" if any_success else "failed"
        training_repository.update_status(session, job_id, final_status)
        _append_log(
            session,
            job_id,
            f"{final_status}: best={best.algo_name if best else 'none'} metric={metric_key}",
        )

        audit_event = Event.TRAINING_COMPLETED if any_success else Event.TRAINING_FAILED
        audit_repository.write(
            session,
            action_type=audit_event,
            target_type="TrainingJob",
            target_id=job_id,
            detail={
                "best_algo": best.algo_name if best else None,
                "metric_key": metric_key,
                "metric_score": best.metrics.get(metric_key) if best else None,
                "n_models": len(scored),
                "n_success": sum(1 for s in scored if s.is_success),
            },
        )
        log_event(
            logger,
            audit_event,
            job_id=job_id,
            best_algo=best.algo_name if best else None,
            n_models=len(scored),
        )


def _load_dataset_for_training(config: TrainingConfig) -> tuple[int, Any]:
    """Dataset 레코드 조회 + 파일 로드 + 타깃 컬럼 존재 확인.

    반환: ``(project_id, dataframe)``.
    """
    with session_scope() as session:
        dataset = dataset_repository.get(session, config.dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", config.dataset_id))
        project_id = dataset.project_id
        file_path = Path(dataset.file_path)

    if not file_path.exists():
        raise StorageError(Msg.FILE_PARSE_FAILED)

    try:
        df = read_tabular(file_path)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001 - 파일 파싱 실패를 StorageError 로 통일
        raise StorageError(Msg.FILE_PARSE_FAILED, cause=exc) from exc

    if config.target_column not in df.columns:
        raise ValidationError(f"타깃 컬럼({config.target_column})이 데이터셋에 존재하지 않습니다.")
    return project_id, df


def _create_running_job(
    config: TrainingConfig,
    *,
    project_id: int,
    metric_key: str,
) -> int:
    """TrainingJob insert → running 전이 + 시작 감사 로그."""
    with session_scope() as session:
        job = training_repository.insert(
            session,
            project_id=project_id,
            dataset_id=config.dataset_id,
            task_type=config.task_type,
            target_column=config.target_column,
            metric_key=metric_key,
            excluded_columns=list(config.excluded_columns),
        )
        job_id = job.training_job_id
        training_repository.update_status(session, job_id, "running")
        _append_log(session, job_id, f"preprocessing (target={config.target_column})")
        audit_repository.write(
            session,
            action_type=Event.TRAINING_STARTED,
            target_type="TrainingJob",
            target_id=job_id,
            detail={
                "project_id": project_id,
                "dataset_id": config.dataset_id,
                "task_type": config.task_type,
                "metric_key": metric_key,
            },
        )
        log_event(
            logger,
            Event.TRAINING_STARTED,
            job_id=job_id,
            project_id=project_id,
            dataset_id=config.dataset_id,
            task_type=config.task_type,
        )
    return job_id


def _build_preprocessing(
    df: Any,
    config: TrainingConfig,
) -> tuple[Any, FeatureSchema, Any, Any]:
    """전처리기/스키마/(X, y) 빌드. 피처가 비면 ``ValidationError``."""
    num_cols, cat_cols = split_feature_types(
        df,
        target=config.target_column,
        excluded=config.excluded_columns,
    )
    try:
        preprocessor = build_preprocessor(num_cols, cat_cols)
    except ValueError as exc:
        raise ValidationError(
            "학습에 사용할 수치/범주 피처가 없습니다. 제외 컬럼 설정을 확인해 주세요.",
            cause=exc,
        ) from exc
    feature_schema = build_feature_schema(df, num_cols, cat_cols, config.target_column)
    X, y = prepare_xy(df, config)
    return preprocessor, feature_schema, X, y


# --------------------------------------------------------- Public use-cases


def run_training(
    config: TrainingConfig,
    *,
    on_progress: ProgressCallback | None = None,
) -> TrainingResultDTO:
    """FR-060~066: 학습 실행 유스케이스.

    반환: ``TrainingResultDTO`` (모든 모델의 비교행 + best).
    실패: ``NotFoundError`` / ``ValidationError`` / ``StorageError`` / ``MLTrainingError``.

    진행 단계 (``on_progress(stage, ratio)``)::

        "preprocessing"  0.05
        "split"          0.10
        "train:<algo>"   0.20 → 0.70 (선형)
        "score"          0.80
        "save"           0.90
        "completed"      1.00
    """
    project_id, df = _load_dataset_for_training(config)
    metric_key = _resolve_metric_key(config.task_type, config.metric_key)
    job_id = _create_running_job(config, project_id=project_id, metric_key=metric_key)

    _emit(on_progress, "preprocessing", _STAGE_RATIO["preprocessing"])

    try:
        preprocessor, feature_schema, X, y = _build_preprocessing(df, config)

        # 4) split
        _emit(on_progress, "split", _STAGE_RATIO["split"])
        with session_scope() as session:
            _append_log(
                session,
                job_id,
                f"split (test_size={config.test_size}, rows={len(df)})",
            )
        X_train, X_test, y_train, y_test = split_dataset(
            X, y, test_size=config.test_size, task_type=config.task_type
        )

        # 5) 다중 학습
        specs = get_specs(config.task_type)
        if not specs:
            raise ValidationError(f"등록된 알고리즘이 없습니다: task_type={config.task_type}")

        train_start = _STAGE_RATIO["train_start"]
        train_end = _STAGE_RATIO["train_end"]

        def _bridge_progress(idx: int, total: int, algo_name: str, status: str) -> None:
            ratio = train_start + (train_end - train_start) * (idx / max(total, 1))
            _emit(on_progress, f"train:{algo_name}", ratio)
            message = f"train:{algo_name} FAILED" if status == "failed" else f"train:{algo_name} ok"
            with suppress(Exception), session_scope() as session:
                _append_log(session, job_id, message)
            if status == "failed":
                log_event(logger, Event.MODEL_TRAIN_FAILED, job_id=job_id, algo=algo_name)

        trained: list[TrainedModel] = train_all(
            specs,
            preprocessor,
            X_train,
            y_train,
            on_progress=_bridge_progress,
        )

        # 6) 평가
        _emit(on_progress, "score", _STAGE_RATIO["score"])
        with session_scope() as session:
            _append_log(session, job_id, "score")
        scored = score_models(trained, X_test, y_test, task_type=config.task_type)
        best = select_best(scored, metric_key)
        # 결과 비교 페이지용 플롯 데이터 (실패 모델 / 예외는 내부에서 스킵)
        plot_data_by_algo = build_plot_data(trained, X_test, y_test, task_type=config.task_type)

        # 7) DB insert + 아티팩트 저장 (§4.3a 보상 트랜잭션)
        _emit(on_progress, "save", _STAGE_RATIO["save"])

        n_train = len(X_train)
        n_test = len(X_test)

        trained_by_name = {t.algo_name: t for t in trained}

        saved_dirs: list[Path] = []
        try:
            _persist_and_save(
                job_id=job_id,
                scored=scored,
                best=best,
                trained_by_name=trained_by_name,
                feature_schema=feature_schema,
                metric_key=metric_key,
                task_type=config.task_type,
                n_train=n_train,
                n_test=n_test,
                saved_dirs=saved_dirs,
                plot_data_by_algo=plot_data_by_algo,
            )
        except Exception as exc:
            # DB 트랜잭션은 session_scope 가 rollback. 파일만 정리.
            _cleanup_model_dirs(saved_dirs)
            wrapped = (
                exc
                if isinstance(exc, AppError)
                else MLTrainingError(Msg.TRAINING_FAILED, cause=exc)
            )
            _fail_job(job_id, wrapped)
            log_event(logger, Event.ARTIFACT_SAVE_FAILED, job_id=job_id, error=str(wrapped))
            if isinstance(exc, AppError):
                raise
            raise wrapped from exc

        _emit(on_progress, "completed", _STAGE_RATIO["completed"])
        return get_training_result(job_id)

    except AppError as err:
        _fail_job(job_id, err)
        raise
    except Exception as exc:  # 파이프라인 전반의 예기치 못한 실패
        wrapped = MLTrainingError(Msg.TRAINING_FAILED, cause=exc)
        _fail_job(job_id, wrapped)
        raise wrapped from exc


def get_training_result(job_id: int) -> TrainingResultDTO:
    """FR-064, FR-066: 저장된 학습 결과(비교표 + best) 복원."""
    with session_scope() as session:
        job = training_repository.get(session, job_id)
        if job is None:
            raise NotFoundError(entity_not_found("학습 잡", job_id))
        models = list(model_repository.list_by_training_job(session, job_id))
        rows = [_row_from_orm(m) for m in models]
        best_algo: str | None = next((r.algo_name for r in rows if r.is_best), None)
        return TrainingResultDTO(
            job_id=job.training_job_id,
            rows=rows,
            best_algo=best_algo,
            metric_key=job.metric_key,
            task_type=job.task_type,
        )


def list_training_jobs(project_id: int) -> list[TrainingJobDTO]:
    """FR-065: 프로젝트 소속 학습 잡 목록(최신순)."""
    with session_scope() as session:
        rows = training_repository.list_by_project(session, project_id)
        return [TrainingJobDTO.from_orm(r) for r in rows]


__all__ = [
    "run_training",
    "get_training_result",
    "list_training_jobs",
]
