"""Service 반환 DTO (IMPLEMENTATION_PLAN §3.1b).

원칙:
- **Repository → Service 경계에서 ORM → DTO 로 변환한다.**
- UI 는 ORM 객체를 절대 받지 않는다 (룰: streamlit-ui.mdc / service-layer.mdc).
- 각 DTO 는 가능한 ``@dataclass(frozen=True, slots=True)`` + ``from_orm`` 클래스 메서드를 제공한다.
- ``ml/schemas.py`` 는 ML 내부 연산용이며 여기와 분리한다.
  다만 예측 검증에 필요한 ``FeatureSchema`` 는 ``ModelDetailDTO`` 에 ``FeatureSchemaDTO`` 로 얇게 노출.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from ml.schemas import PreprocessingConfig
    from repositories.models import Dataset, Model, PredictionJob, Project, TrainingJob


# ------------------------------------------------------------------- Project


@dataclass(frozen=True, slots=True)
class ProjectDTO:
    id: int
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    dataset_count: int = 0
    model_count: int = 0

    @classmethod
    def from_orm(
        cls,
        entity: Project,
        *,
        dataset_count: int = 0,
        model_count: int = 0,
    ) -> ProjectDTO:
        return cls(
            id=entity.project_id,
            name=entity.project_name,
            description=entity.description,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            dataset_count=dataset_count,
            model_count=model_count,
        )


# ------------------------------------------------------------------ Datasets


@dataclass(frozen=True, slots=True)
class DatasetDTO:
    id: int
    project_id: int
    file_name: str
    row_count: int
    column_count: int
    created_at: datetime

    @classmethod
    def from_orm(cls, entity: Dataset) -> DatasetDTO:
        return cls(
            id=entity.dataset_id,
            project_id=entity.project_id,
            file_name=entity.file_name,
            row_count=entity.row_count,
            column_count=entity.column_count,
            created_at=entity.created_at,
        )


@dataclass(frozen=True, slots=True)
class ColumnProfileDTO:
    name: str
    dtype: str
    n_missing: int
    n_unique: int
    missing_ratio: float
    unique_ratio: float


@dataclass(frozen=True, slots=True)
class DatasetProfileDTO:
    rows: int
    cols: int
    columns: list[ColumnProfileDTO]


# ------------------------------------------------------------- Training Jobs


@dataclass(frozen=True, slots=True)
class TrainingJobDTO:
    id: int
    project_id: int
    dataset_id: int
    task_type: str
    target_column: str
    metric_key: str
    status: str
    started_at: datetime | None
    ended_at: datetime | None

    @classmethod
    def from_orm(cls, entity: TrainingJob) -> TrainingJobDTO:
        return cls(
            id=entity.training_job_id,
            project_id=entity.project_id,
            dataset_id=entity.dataset_id,
            task_type=entity.task_type,
            target_column=entity.target_column,
            metric_key=entity.metric_key,
            status=entity.status,
            started_at=entity.started_at,
            ended_at=entity.ended_at,
        )


# ---------------------------------------------------------- Training Results


@dataclass(frozen=True, slots=True)
class ModelComparisonRowDTO:
    algo_name: str
    status: str  # success | failed
    metrics: dict[str, float] = field(default_factory=dict)
    train_time_ms: int = 0
    is_best: bool = False
    error: str | None = None
    # 결과 페이지(§6.4) 에서 save_model/get_model_plot_data 에 필요. 실패/아직 미저장 시 None.
    model_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingResultDTO:
    job_id: int
    rows: list[ModelComparisonRowDTO]
    best_algo: str | None
    metric_key: str
    task_type: str = ""  # 페이지 분기용 (classification | regression). 빈 문자열 허용(구버전 호환)


# -------------------------------------------------------------------- Models


@dataclass(frozen=True, slots=True)
class ModelDTO:
    id: int
    training_job_id: int
    algo_name: str
    metric_score: float | None
    is_best: bool
    created_at: datetime

    @classmethod
    def from_orm(cls, entity: Model) -> ModelDTO:
        return cls(
            id=entity.model_id,
            training_job_id=entity.training_job_id,
            algo_name=entity.algorithm_name,
            metric_score=entity.metric_score,
            is_best=entity.is_best,
            created_at=entity.created_at,
        )


@dataclass(frozen=True, slots=True)
class FeatureSchemaDTO:
    """UI 가 예측 입력 폼을 그리기 위한 최소 피처 스키마."""

    numeric: list[str]
    categorical: list[str]
    target: str
    categories: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelDetailDTO:
    base: ModelDTO
    feature_schema: FeatureSchemaDTO
    metrics_summary: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------- Prediction Jobs


@dataclass(frozen=True, slots=True)
class PredictionResultDTO:
    job_id: int
    rows: list[dict[str, Any]]
    result_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_orm(
        cls,
        entity: PredictionJob,
        *,
        rows: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> PredictionResultDTO:
        return cls(
            job_id=entity.prediction_job_id,
            rows=rows or [],
            result_path=entity.result_path,
            warnings=warnings or [],
        )


# --------------------------------------------------------------------- Admin
# ``pages/07_admin.py`` + ``services/admin_service.py`` 가 소비한다. 순수 DTO 로 Repository
# 의 join 결과(프로젝트명 포함 등)를 한 번에 포장한다.


@dataclass(frozen=True, slots=True)
class AdminStatsDTO:
    """운영 대시보드 상단 카드용 집계 (FR-090)."""

    projects: int = 0
    datasets: int = 0
    training_jobs: int = 0
    models: int = 0
    predictions: int = 0
    training_failures: int = 0
    prediction_failures: int = 0


@dataclass(frozen=True, slots=True)
class TrainingHistoryRowDTO:
    """학습 이력 1행 (FR-091). 프로젝트명/성공·실패 모델 수/베스트 지표 포함."""

    id: int
    project_id: int
    project_name: str
    dataset_id: int
    task_type: str
    target_column: str
    metric_key: str
    status: str
    started_at: datetime | None
    ended_at: datetime | None
    duration_ms: int | None
    n_models_success: int
    n_models_failed: int
    best_algo: str | None
    best_metric: float | None


@dataclass(frozen=True, slots=True)
class PredictionHistoryRowDTO:
    """예측 이력 1행 (FR-091). 상위 학습 잡을 경유해 프로젝트명 해석."""

    id: int
    model_id: int
    algorithm_name: str
    project_id: int
    project_name: str
    input_type: str  # form | file
    status: str
    created_at: datetime
    input_file_path: str | None
    result_path: str | None


@dataclass(frozen=True, slots=True)
class AuditLogEntryDTO:
    """감사 로그 한 줄 (FR-093 실패 요약용)."""

    id: int
    action_type: str
    target_type: str | None
    target_id: int | None
    action_time: datetime
    detail: dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------- Preprocessing (§9.7)


@dataclass(frozen=True, slots=True)
class PreprocessingConfigDTO:
    """UI ↔ Service 경계에서 오가는 고급 전처리 설정 DTO (FR-055~058).

    ``PreprocessingConfig`` (ml 레이어) 와 1:1 대응하지만 JSON/폼 친화적으로
    ``tuple`` → ``list`` 로 평면화되어 있다. ``to_config`` / ``from_config`` 로
    경계에서 변환.
    """

    numeric_impute: str = "median"
    numeric_scale: str = "standard"
    outlier: str = "none"
    outlier_iqr_k: float = 1.5
    winsorize_p: float = 0.01
    categorical_impute: str = "most_frequent"
    categorical_encoding: str = "onehot"
    highcard_threshold: int = 50
    highcard_auto_downgrade: bool = True
    datetime_decompose: bool = False
    datetime_parts: list[str] = field(default_factory=list)
    bool_as_numeric: bool = True
    imbalance: str = "none"
    smote_k_neighbors: int = 5

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> PreprocessingConfigDTO:
        return cls(
            numeric_impute=config.numeric_impute,
            numeric_scale=config.numeric_scale,
            outlier=config.outlier,
            outlier_iqr_k=config.outlier_iqr_k,
            winsorize_p=config.winsorize_p,
            categorical_impute=config.categorical_impute,
            categorical_encoding=config.categorical_encoding,
            highcard_threshold=config.highcard_threshold,
            highcard_auto_downgrade=config.highcard_auto_downgrade,
            datetime_decompose=config.datetime_decompose,
            datetime_parts=list(config.datetime_parts),
            bool_as_numeric=config.bool_as_numeric,
            imbalance=config.imbalance,
            smote_k_neighbors=config.smote_k_neighbors,
        )

    def to_config(self) -> PreprocessingConfig:
        # Local import to keep ml/ 의존을 typing 범위로 격리 (streamlit-ui.mdc 경계 유지).
        from ml.schemas import PreprocessingConfig as _PC

        return _PC(
            numeric_impute=self.numeric_impute,  # type: ignore[arg-type]
            numeric_scale=self.numeric_scale,  # type: ignore[arg-type]
            outlier=self.outlier,  # type: ignore[arg-type]
            outlier_iqr_k=self.outlier_iqr_k,
            winsorize_p=self.winsorize_p,
            categorical_impute=self.categorical_impute,  # type: ignore[arg-type]
            categorical_encoding=self.categorical_encoding,  # type: ignore[arg-type]
            highcard_threshold=self.highcard_threshold,
            highcard_auto_downgrade=self.highcard_auto_downgrade,
            datetime_decompose=self.datetime_decompose,
            datetime_parts=tuple(self.datetime_parts),  # type: ignore[arg-type]
            bool_as_numeric=self.bool_as_numeric,
            imbalance=self.imbalance,  # type: ignore[arg-type]
            smote_k_neighbors=self.smote_k_neighbors,
        )


@dataclass(frozen=True, slots=True)
class FeaturePreviewDTO:
    """피처 변환 미리보기 결과 (§9.7 / FR-058 / UI 소비용).

    - ``n_cols_in``: 타깃/제외 컬럼을 뺀 입력 피처 개수.
    - ``n_cols_out``: 예상되는 변환 후 열 수 (onehot 분해·datetime 파트 포함).
      **실제 fit 없이** ``nunique`` / `datetime_parts` 등 메타데이터로 추정.
    - ``derived``: ``(source, name, kind)`` 튜플 목록. onehot 은 `col__val` 형태.
    - ``encoding_summary``: 범주형 컬럼별 실제 적용될 인코딩 (``onehot|ordinal|frequency``).
    - ``auto_downgraded``: ``onehot`` 요청이었지만 고카디널리티로 ``frequency``
      자동 강등된 컬럼명 목록.
    """

    n_cols_in: int
    n_cols_out: int
    derived: tuple[tuple[str, str, str], ...] = ()
    encoding_summary: dict[str, str] = field(default_factory=dict)
    auto_downgraded: tuple[str, ...] = ()


# --------------------------------------------------------- Algorithm registry


@dataclass(frozen=True, slots=True)
class AlgorithmInfoDTO:
    """학습 페이지가 소비하는 알고리즘 후보 1건 (§10.4, FR-067).

    - ``available=False`` 이면 ``unavailable_reason`` 에 사용자 복구 힌트가 담긴다
      (예: "패키지 미설치 (pip install 필요)", "libomp 미설치 ...").
      이런 항목은 UI 선택지에 **disabled** 로 노출되거나 목록에서 제외된다.
    - ``is_optional_backend`` 가 True 인 알고리즘은 특별한 caption/설치 안내를
      붙이는 분기점.
    """

    name: str
    task_type: str  # "classification" | "regression"
    default_metric: str
    is_optional_backend: bool = False
    available: bool = True
    unavailable_reason: str = ""


@dataclass(frozen=True, slots=True)
class OptionalBackendInfoDTO:
    """Optional backend(XGBoost/LightGBM/CatBoost) 가용 상태 1건 (§10.4, FR-069)."""

    name: str
    available: bool
    reason: str = ""
