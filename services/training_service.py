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
from ml.balancing import apply_imbalance_strategy
from ml.evaluators import (
    METRIC_DIRECTIONS,
    build_plot_data,
    score_models,
    select_best,
)
from ml.preprocess import (
    PreprocessingRouteReport,
    build_feature_schema,
    build_preprocessor,
    plan_categorical_routing,
    prepare_xy,
    split_feature_types_v2,
)
from ml.registry import AlgoSpec, get_specs, optional_backends_status
from ml.schemas import PreprocessingConfig
from ml.trainers import split_dataset, train_all
from repositories import (
    audit_repository,
    dataset_repository,
    model_repository,
    training_repository,
)
from repositories.base import session_scope
from services.dto import (
    AlgorithmInfoDTO,
    FeaturePreviewDTO,
    ModelComparisonRowDTO,
    OptionalBackendInfoDTO,
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

    import pandas as pd

    from ml.schemas import FeatureSchema, ScoredModel, TrainingConfig
    from ml.trainers import BalancerCallable, TrainedModel

    ProgressCallback = Callable[[str, float], None]

logger = get_logger(__name__)

_ERROR_TRUNCATE_LEN = 500
# 학습 수명주기 단계 가중치. ratio 계산에 쓰인다.
# §9.7: feature_engineering / balance 단계 삽입 (항상 emit — 기본 설정에서도 stage 출력 보장).
_STAGE_RATIO = {
    "preprocessing": 0.05,
    "feature_engineering": 0.07,
    "split": 0.10,
    "balance": 0.12,
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
    preprocessing_config: PreprocessingConfig | None = None,
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
                preprocessing_config=preprocessing_config,
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


def _validate_target_column(df: Any, config: TrainingConfig) -> None:
    """타깃 컬럼 dtype/고유값 사전 검증 (학습 진입 전).

    다음 경우 ``ValidationError`` 로 즉시 차단한다:
    - 타깃이 ``datetime64`` 계열 (분류/회귀 모두 의미 없음, xgboost 등은 fit 자체 실패)
    - ``task_type='classification'`` 인데 고유 클래스 수가 ``max(50, n_samples/2)`` 이상
      (거의 unique 에 가까운 컬럼을 분류 타깃으로 잘못 선택한 경우 방어)
    """
    import pandas as pd  # 지연 import: ml/ 레이어에 두지 않기 위함

    y = df[config.target_column]
    if pd.api.types.is_datetime64_any_dtype(y):
        raise ValidationError(Msg.TARGET_DATETIME_NOT_SUPPORTED)

    if config.task_type == "classification":
        n_samples = int(len(y))
        n_unique = int(y.nunique(dropna=True))
        threshold = max(50, n_samples // 2)
        if n_unique >= threshold and n_samples > 0:
            raise ValidationError(
                f"{Msg.TARGET_TOO_MANY_CLASSES} (고유값 {n_unique} / 샘플 {n_samples})"
            )


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


def _record_preprocessing_metadata(
    *,
    job_id: int,
    pp_cfg_effective: PreprocessingConfig,
    dropped_dt_cols: tuple[str, ...],
) -> None:
    """§9.7/§9.8: 전처리 요약 run_log + (커스텀이면) AuditLog + (datetime drop) 안내."""
    with session_scope() as session:
        _append_log(session, job_id, f"preprocessing: {pp_cfg_effective.summary()}")
        if dropped_dt_cols:
            _append_log(
                session,
                job_id,
                (
                    "datetime_dropped: "
                    + ",".join(dropped_dt_cols)
                    + " (고급 전처리에서 'datetime 분해'를 켜면 year/month 등으로 분해됩니다)"
                ),
            )
        if not pp_cfg_effective.is_default:
            audit_repository.write(
                session,
                action_type=Event.TRAINING_PREPROCESSING_CUSTOMIZED,
                target_type="TrainingJob",
                target_id=job_id,
                detail={"summary": pp_cfg_effective.summary()},
            )
            log_event(
                logger,
                Event.TRAINING_PREPROCESSING_CUSTOMIZED,
                job_id=job_id,
                summary=pp_cfg_effective.summary(),
            )


def _build_preprocessing(
    df: Any,
    config: TrainingConfig,
) -> tuple[Any, FeatureSchema, Any, Any, PreprocessingRouteReport | None, tuple[str, ...]]:
    """전처리기/스키마/(X, y)/드롭된 datetime 컬럼 빌드. 피처가 비면 ``ValidationError``.

    §9.7: ``config.preprocessing`` 이 주어지면 고급 설정을 그대로 ml 레이어로 forward.
    None 이면 기존 MVP 경로 유지. 단 **두 경로 모두** ``split_feature_types_v2`` 로
    datetime 컬럼을 선제 분리한다 — ``SimpleImputer`` 가 ``datetime64[ns]`` dtype 을
    거부하는 이슈(모든 알고리즘 동반 실패)를 차단하기 위함.

    기본 경로 (``pp_cfg=None``):
    - datetime 컬럼은 **자동 drop** (사용자가 ``excluded_columns`` 로 명시했을 때와 동일 효과).
    - bool 컬럼은 기존 v1 호환을 위해 cat 그룹으로 합류.
    - drop 된 datetime 컬럼 목록은 호출자가 ``run_log`` 에 안내 문구로 append.

    반환값: ``(preprocessor, feature_schema, X, y, route_report, dropped_datetime_cols)``.
    """
    pp_cfg = config.preprocessing
    num_cols_v2, cat_cols_v2, dt_cols, bool_cols = split_feature_types_v2(
        df,
        target=config.target_column,
        excluded=config.excluded_columns,
    )
    if pp_cfg is None:
        # 기본 경로: bool 은 cat 으로 합류(v1 호환), datetime 은 자동 drop.
        effective_cat = list(cat_cols_v2) + [c for c in bool_cols if c not in cat_cols_v2]
        try:
            preprocessor = build_preprocessor(num_cols_v2, effective_cat)
        except ValueError as exc:
            raise ValidationError(
                "학습에 사용할 수치/범주 피처가 없습니다. 제외 컬럼 설정을 확인해 주세요.",
                cause=exc,
            ) from exc
        feature_schema = build_feature_schema(
            df,
            num_cols_v2,
            effective_cat,
            config.target_column,
            datetime_cols=dt_cols,
        )
        X, y = prepare_xy(df, config)
        if dt_cols:
            X = X.drop(columns=[c for c in dt_cols if c in X.columns])
        return preprocessor, feature_schema, X, y, None, tuple(dt_cols)

    # §9.7 신규 경로: 타입 분류 v2 + config 기반 preprocessor + route report
    route_report = plan_categorical_routing(df, cat_cols_v2, pp_cfg)
    try:
        preprocessor = build_preprocessor(
            num_cols_v2,
            cat_cols_v2,
            config=pp_cfg,
            df_sample=df,
            datetime_cols=dt_cols,
            bool_cols=bool_cols,
        )
    except ValueError as exc:
        raise ValidationError(
            "학습에 사용할 수치/범주 피처가 없습니다. 제외 컬럼 설정을 확인해 주세요.",
            cause=exc,
        ) from exc
    feature_schema = build_feature_schema(
        df,
        num_cols_v2,
        cat_cols_v2,
        config.target_column,
        datetime_cols=dt_cols,
        bool_cols=bool_cols,
        config=pp_cfg,
        route_report=route_report,
    )
    X, y = prepare_xy(df, config)
    # datetime_decompose=False 면 datetime 컬럼은 ColumnTransformer 에서 drop 되지만
    # pandas DataFrame 이 여전히 해당 컬럼을 담고 있으면 하위 추정기 호출 직전
    # 검증 단계에서 dtype 체크가 통과하지 않는 경우가 있어 사전 제거해 둔다.
    drop_dt: tuple[str, ...] = ()
    if dt_cols and not pp_cfg.datetime_decompose:
        drop_dt = tuple(dt_cols)
        X = X.drop(columns=[c for c in dt_cols if c in X.columns])
    return preprocessor, feature_schema, X, y, route_report, drop_dt


def _make_balancer(
    pp_cfg: PreprocessingConfig,
    task_type: str,
) -> BalancerCallable | None:
    """§9.7: PreprocessingConfig 의 imbalance 전략에 맞는 per-spec balancer 생성.

    - ``none`` → None (train_all 패스스루)
    - ``class_weight`` / ``smote`` → 각 spec 의 fresh estimator 에 대해
      ``apply_imbalance_strategy`` 를 호출하는 클로저.

    **주의**: 이 callable 에 전달되는 ``X_train/y_train`` 은 반드시 ``split_dataset``
    이후의 train split 이어야 한다 (§9.5 docstring). 호출자(run_training)가 보장.
    """
    if pp_cfg.imbalance == "none":
        return None

    def _balancer(
        estimator: Any, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[Any, pd.DataFrame, pd.Series]:
        return apply_imbalance_strategy(estimator, X_train, y_train, pp_cfg, task_type=task_type)

    return _balancer


def _apply_algorithm_filter(
    specs: list[AlgoSpec], config: TrainingConfig, job_id: int
) -> list[AlgoSpec]:
    """§10.3 (FR-067): ``config.algorithms`` 로 후보 specs 를 필터링.

    - ``config.algorithms is None`` → 입력을 그대로 반환 (v0.2.0 byte/audit 동치).
    - 빈/중복 값은 ``TrainingConfig.__post_init__`` 에서 이미 차단됨.
    - 미등록 이름 포함 → ``ValidationError``.
    - 필터가 실제로 적용될 때만 ``Event.TRAINING_ALGORITHMS_FILTERED`` 를
      AuditLog + 구조화 로그 각 1회 기록.
    """
    if config.algorithms is None:
        return specs

    requested = set(config.algorithms)
    known = {s.name for s in specs}
    unknown = requested - known
    if unknown:
        raise ValidationError(
            f"등록되지 않은 알고리즘: {sorted(unknown)} (task={config.task_type})"
        )
    filtered = [s for s in specs if s.name in requested]
    if not filtered:
        raise ValidationError("선택한 알고리즘이 현재 task 에 등록되어 있지 않습니다.")

    sorted_names = sorted(requested)
    with session_scope() as session:
        _append_log(session, job_id, f"algorithms={sorted_names}")
        audit_repository.write(
            session,
            action_type=Event.TRAINING_ALGORITHMS_FILTERED,
            target_type="TrainingJob",
            target_id=job_id,
            detail={"algorithms": sorted_names},
        )
    log_event(
        logger,
        Event.TRAINING_ALGORITHMS_FILTERED,
        job_id=job_id,
        algorithms=sorted_names,
    )
    return filtered


def rebuild_xy_for_influence_analysis(df: Any, config: TrainingConfig) -> tuple[Any, Any]:
    """FR-094: 특성 영향도 계산용으로 학습과 동일한 전처리 경로에서 ``(X, y)`` 재구성.

    호출 전에 ``assert_valid_training_target(df, config)`` 로 타깃 검증을 권장한다.
    """
    _, _schema, X, y, _route, _dropped = _build_preprocessing(df, config)
    return X, y


def assert_valid_training_target(df: Any, config: TrainingConfig) -> None:
    """FR-094: 사후 분석·재현 경로에서 학습과 동일한 타깃 검증."""
    _validate_target_column(df, config)


def _emit_tuning_downgrade(config: TrainingConfig, job_id: int) -> None:
    """§10.3 (§11 선반영): 튜닝 요청 시 안전 downgrade.

    ``ml/tuners.py`` 가 없는 현 시점에서 ``method != "none"`` 이면 run_log 에
    경고를 남기고 구조화 로그로 ``Event.TRAINING_TUNING_DOWNGRADED`` 을 1회 emit
    한 뒤, 실제 학습은 기본(비튜닝) 경로로 진행한다.
    """
    if config.tuning is None or config.tuning.method == "none":
        return
    with session_scope() as session:
        _append_log(
            session,
            job_id,
            f"tuning=downgraded_v010 (requested method={config.tuning.method})",
        )
    log_event(
        logger,
        Event.TRAINING_TUNING_DOWNGRADED,
        job_id=job_id,
        requested_method=config.tuning.method,
    )


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

        "preprocessing"        0.05
        "feature_engineering"  0.07   (§9.7 추가)
        "split"                0.10
        "balance"              0.12   (§9.7 추가 — 실제 리샘플 여부와 무관하게 emit)
        "train:<algo>"         0.20 → 0.70 (선형)
        "score"                0.80
        "save"                 0.90
        "completed"            1.00
    """
    project_id, df = _load_dataset_for_training(config)
    _validate_target_column(df, config)
    metric_key = _resolve_metric_key(config.task_type, config.metric_key)
    job_id = _create_running_job(config, project_id=project_id, metric_key=metric_key)

    _emit(on_progress, "preprocessing", _STAGE_RATIO["preprocessing"])

    try:
        (
            preprocessor,
            feature_schema,
            X,
            y,
            _route_report,
            dropped_dt_cols,
        ) = _build_preprocessing(df, config)

        pp_cfg_effective = config.preprocessing or PreprocessingConfig()
        _record_preprocessing_metadata(
            job_id=job_id,
            pp_cfg_effective=pp_cfg_effective,
            dropped_dt_cols=dropped_dt_cols,
        )

        _emit(on_progress, "feature_engineering", _STAGE_RATIO["feature_engineering"])

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

        # §9.6 주의사항: balancer 는 반드시 train split 이후에 적용. 테스트 세트 리샘플 금지.
        _emit(on_progress, "balance", _STAGE_RATIO["balance"])
        balancer = _make_balancer(pp_cfg_effective, config.task_type)
        if balancer is not None:
            with session_scope() as session:
                _append_log(session, job_id, f"balance: strategy={pp_cfg_effective.imbalance}")

        # 5) 다중 학습
        specs = get_specs(config.task_type)
        if not specs:
            raise ValidationError(f"등록된 알고리즘이 없습니다: task_type={config.task_type}")

        specs = _apply_algorithm_filter(specs, config, job_id)
        _emit_tuning_downgrade(config, job_id)

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
            preprocess_cfg=pp_cfg_effective,
            balancer=balancer,
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
                preprocessing_config=(
                    config.preprocessing
                    if config.preprocessing is not None and not pp_cfg_effective.is_default
                    else None
                ),
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


def preview_preprocessing(
    dataset_id: int,
    config: TrainingConfig,
) -> FeaturePreviewDTO:
    """FR-058 / §9.7: 실제 fit 없이 PreprocessingConfig 기반 변환 요약을 산출.

    - 학습 잡을 **생성하지 않는다**. 읽기 전용 유스케이스.
    - 5만 행 기준 < 2초 목표 (NFR-003) — ``nunique`` 기반 메타데이터만 사용.
    - ``config.preprocessing`` 이 None 이면 ``PreprocessingConfig()`` 기본값으로 계산.
    """
    with session_scope() as session:
        dataset = dataset_repository.get(session, dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", dataset_id))
        file_path = Path(dataset.file_path)

    if not file_path.exists():
        raise StorageError(Msg.FILE_PARSE_FAILED)

    try:
        df = read_tabular(file_path)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001 - 파일 파싱 실패는 StorageError 로 통일
        raise StorageError(Msg.FILE_PARSE_FAILED, cause=exc) from exc

    if config.target_column not in df.columns:
        raise ValidationError(f"타깃 컬럼({config.target_column})이 데이터셋에 존재하지 않습니다.")

    pp_cfg = config.preprocessing or PreprocessingConfig()

    num_cols, cat_cols, dt_cols, bool_cols = split_feature_types_v2(
        df,
        target=config.target_column,
        excluded=config.excluded_columns,
    )
    route_report = plan_categorical_routing(df, cat_cols, pp_cfg)

    # bool_as_numeric=False 일 때 bool 은 범주로 합류 (UI 표시도 그 규칙을 반영).
    effective_cat_cols = list(cat_cols) if pp_cfg.bool_as_numeric else [*cat_cols, *bool_cols]
    effective_bool_cols = list(bool_cols) if pp_cfg.bool_as_numeric else []

    feature_schema = build_feature_schema(
        df,
        num_cols,
        effective_cat_cols,
        config.target_column,
        datetime_cols=dt_cols,
        bool_cols=effective_bool_cols,
        config=pp_cfg,
        route_report=route_report,
    )

    # 입력 열 수: 타깃/제외 제외. datetime 은 decompose=False 면 드롭되지만 "입력 컬럼" 기준이라 포함.
    n_cols_in = len(num_cols) + len(cat_cols) + len(dt_cols) + len(bool_cols)

    # 출력 열 수: num(passthrough 1) + derived 목록 크기.
    # - cat: onehot 분해 값 수, ordinal/frequency 는 1
    # - datetime_decompose=True 면 파트 수만큼, False 면 0 (드롭)
    # - bool_as_numeric=True 면 1, 아니면 위 effective_cat_cols 로 합류되어 cat 경로로 계산
    n_cols_out = len(num_cols) + len(feature_schema.derived)

    derived_tuples: tuple[tuple[str, str, str], ...] = tuple(
        (d.source, d.name, d.kind) for d in feature_schema.derived
    )

    return FeaturePreviewDTO(
        n_cols_in=n_cols_in,
        n_cols_out=n_cols_out,
        derived=derived_tuples,
        encoding_summary=dict(route_report.encoding_per_col),
        auto_downgraded=route_report.auto_downgraded,
    )


# --------------------------------------------------- §10.4 Algorithm discovery


def list_optional_backends() -> list[OptionalBackendInfoDTO]:
    """FR-069: Optional backend(xgboost/lightgbm/catboost) 가용 상태 목록.

    UI 가 `ml.registry` 를 직접 import 하지 않도록 하기 위한 얇은 래퍼.
    조회 순서는 등록 시도 순서 (xgboost → lightgbm → catboost).
    """
    return [
        OptionalBackendInfoDTO(name=s.name, available=s.available, reason=s.reason)
        for s in optional_backends_status()
    ]


def list_algorithms(task_type: str) -> list[AlgorithmInfoDTO]:
    """FR-067: 지정된 task 의 학습 후보 알고리즘 목록.

    - 현재 등록된 모든 스펙(= 가용 스펙)을 ``available=True`` 로 노출.
    - **미설치 optional backend** 는 같은 task 에 대한 "가상 스펙" 으로
      ``available=False`` + 사유(`unavailable_reason`) 와 함께 포함해 UI 가
      "왜 후보에 없는지" 를 표시할 수 있게 한다.
    - 반환 순서: 가용 스펙 → 미설치 optional 백엔드. 가용 스펙 내부 순서는
      registry 등록 순서를 그대로 보존(훈련 경로와 동일).
    """
    if task_type not in ("classification", "regression"):
        raise ValidationError(
            f"task_type 은 classification/regression 중 하나여야 합니다: {task_type}"
        )

    result: list[AlgorithmInfoDTO] = []
    registered_names = set()
    for spec in get_specs(task_type):  # type: ignore[arg-type]
        registered_names.add(spec.name)
        result.append(
            AlgorithmInfoDTO(
                name=spec.name,
                task_type=task_type,
                default_metric=spec.default_metric,
                is_optional_backend=spec.is_optional_backend,
                available=True,
                unavailable_reason="",
            )
        )

    # 미설치 optional backend 는 task 무관하게 unavailable 로 노출.
    for backend in optional_backends_status():
        if backend.available or backend.name in registered_names:
            continue
        result.append(
            AlgorithmInfoDTO(
                name=backend.name,
                task_type=task_type,
                default_metric="",
                is_optional_backend=True,
                available=False,
                unavailable_reason=backend.reason,
            )
        )

    return result


__all__ = [
    "assert_valid_training_target",
    "get_training_result",
    "list_algorithms",
    "list_optional_backends",
    "list_training_jobs",
    "preview_preprocessing",
    "rebuild_xy_for_influence_analysis",
    "run_training",
]
