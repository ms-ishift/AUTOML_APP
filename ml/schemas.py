"""ML 엔진 내부 도메인 스키마 (IMPLEMENTATION_PLAN §3.1a / §9.1, FR-055~058).

원칙:
- 이 모듈은 **Streamlit / SQLAlchemy / DB 의존 없이** 동작해야 한다. (룰: ml-engine.mdc)
- 모든 구조체는 ``@dataclass(frozen=True, slots=True)`` 로 불변.
- 리스트/딕셔너리 대신 ``tuple`` 을 우선 사용해 해시 가능성을 확보.
- ORM 엔터티(`repositories.models.*`) 와 1:1 매핑이 아닌 점 주의.
  ORM 은 저장 형식, 이 스키마는 **연산 입력/출력** 형식.
- 라이브러리 의존(``sklearn``, ``imblearn``)은 여기에 두지 않는다 —
  실제 transformer 주입은 ``ml/preprocess.py`` · ``ml/balancing.py`` 로 분리.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["classification", "regression"]
ModelStatus = Literal["success", "failed"]


# --------------------------------------------------- Preprocessing strategies
#
# §9.1 — 전처리 전략 축. 모든 Literal 은 사용자 노출 키와 1:1 매핑된다.
# 실제 transformer 변환 규칙은 ``ml/preprocess.py`` 에서 이 Literal 값을 받아 분기한다.

ImputeNumericStrategy = Literal["median", "mean", "most_frequent", "constant_zero", "drop_rows"]
ImputeCategoricalStrategy = Literal["most_frequent", "constant_missing"]
ScaleStrategy = Literal["standard", "minmax", "robust", "none"]
OutlierStrategy = Literal["none", "iqr_clip", "winsorize"]
CategoricalEncoding = Literal["onehot", "ordinal", "frequency"]
ImbalanceStrategy = Literal["none", "class_weight", "smote"]
DatetimePart = Literal["year", "month", "day", "weekday", "hour", "is_weekend"]


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """단일 컬럼 요약."""

    name: str
    dtype: str
    n_missing: int
    n_unique: int
    missing_ratio: float = 0.0
    unique_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """데이터셋 전체 요약."""

    n_rows: int
    n_cols: int
    columns: tuple[ColumnProfile, ...]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass(frozen=True, slots=True)
class DerivedFeature:
    """전처리에서 생성된 파생 피처 1개 (FR-055~057).

    ``name`` 은 최종 피처명(예측 입력에서 실제로 등장하는 이름),
    ``source`` 는 원본 입력 컬럼(1개), ``kind`` 는 변환 유형 태그이다.

    권장 ``kind`` 값 (free-form 허용, 고정 열거형은 아님):
    - ``datetime_year`` / ``datetime_month`` / ``datetime_day`` /
      ``datetime_weekday`` / ``datetime_hour`` / ``datetime_is_weekend``
    - ``bool_numeric``
    - ``onehot`` / ``ordinal`` / ``frequency``
    - ``passthrough``
    - ``iqr_clipped`` / ``winsorized``

    ``kind`` 를 Literal 로 고정하지 않은 이유: §9.3 이후 새 변환 유형
    (예: ``log1p``, ``yeojohnson``)을 도입할 때 스키마 변경 없이 확장하기 위함.
    """

    name: str
    source: str
    kind: str


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """사용자 제어 가능한 전처리 설정 (§9.1, FR-055~058).

    모든 필드는 기본값을 가지며, ``PreprocessingConfig()`` 는 현재 MVP 동작
    (median → standard / most_frequent → onehot)과 **논리적으로 동일**하다.

    이 스키마는 전략 키만 보관하고 실제 transformer 를 import 하지 않는다.
    SMOTE · sklearn 주입은 ``ml/preprocess.py`` / ``ml/balancing.py`` 책임.
    """

    numeric_impute: ImputeNumericStrategy = "median"
    numeric_scale: ScaleStrategy = "standard"
    outlier: OutlierStrategy = "none"
    outlier_iqr_k: float = 1.5
    winsorize_p: float = 0.01
    categorical_impute: ImputeCategoricalStrategy = "most_frequent"
    categorical_encoding: CategoricalEncoding = "onehot"
    highcard_threshold: int = 50
    highcard_auto_downgrade: bool = True
    datetime_decompose: bool = False
    datetime_parts: tuple[DatetimePart, ...] = ()
    bool_as_numeric: bool = True
    imbalance: ImbalanceStrategy = "none"
    smote_k_neighbors: int = 5

    def __post_init__(self) -> None:
        if self.outlier_iqr_k <= 0:
            raise ValueError("outlier_iqr_k 는 0보다 커야 합니다.")
        if not (0.0 < self.winsorize_p < 0.5):
            raise ValueError("winsorize_p 는 (0, 0.5) 범위여야 합니다.")
        if self.smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors 는 1 이상이어야 합니다.")
        if self.highcard_threshold < 2:
            raise ValueError("highcard_threshold 는 2 이상이어야 합니다.")
        if self.datetime_decompose and not self.datetime_parts:
            raise ValueError(
                "datetime_decompose=True 이면 datetime_parts 에 최소 1개 파트가 필요합니다."
            )
        # task 의존 검증(회귀+SMOTE)은 TrainingConfig.__post_init__ 에서 수행.

    @property
    def is_default(self) -> bool:
        """모든 필드가 기본값이면 True (run_log 기록 분기용)."""
        return self == PreprocessingConfig()

    def summary(self) -> str:
        """기본값과 다른 축만 ``key=value`` 로 나열한 로그 문자열.

        기본 인스턴스면 ``"default"`` 반환.
        """
        if self.is_default:
            return "default"
        default = PreprocessingConfig()
        parts: list[str] = []
        # __dataclass_fields__ 순서를 그대로 따른다 (선언 순서 = 표시 순서).
        for fname in self.__dataclass_fields__:
            cur = getattr(self, fname)
            if cur != getattr(default, fname):
                # tuple 은 list 처럼 간결 표기
                if isinstance(cur, tuple):
                    parts.append(f"{fname}={list(cur)}")
                else:
                    parts.append(f"{fname}={cur}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-호환 dict 로 직렬화 (아티팩트 저장용)."""
        return {
            "numeric_impute": self.numeric_impute,
            "numeric_scale": self.numeric_scale,
            "outlier": self.outlier,
            "outlier_iqr_k": self.outlier_iqr_k,
            "winsorize_p": self.winsorize_p,
            "categorical_impute": self.categorical_impute,
            "categorical_encoding": self.categorical_encoding,
            "highcard_threshold": self.highcard_threshold,
            "highcard_auto_downgrade": self.highcard_auto_downgrade,
            "datetime_decompose": self.datetime_decompose,
            "datetime_parts": list(self.datetime_parts),
            "bool_as_numeric": self.bool_as_numeric,
            "imbalance": self.imbalance,
            "smote_k_neighbors": self.smote_k_neighbors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingConfig:
        """dict → PreprocessingConfig. 누락된 키는 기본값 적용 (구 아티팩트 호환).

        알 수 없는 키는 **조용히 무시** 한다 (점진적 스키마 확장 대비).
        """
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for fname in cls.__dataclass_fields__:
            if fname in data:
                kwargs[fname] = data[fname]
            else:
                kwargs[fname] = getattr(defaults, fname)
        # tuple 캐스팅 보정 (JSON 은 list 로 들어옴)
        dt_parts = kwargs.get("datetime_parts")
        if isinstance(dt_parts, list):
            kwargs["datetime_parts"] = tuple(dt_parts)
        return cls(**kwargs)


TuningMethod = Literal["none", "grid", "halving"]


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """하이퍼파라미터 튜닝 설정 (§10.3 스키마만, 실제 실행은 §11).

    이번 스프린트(§10) 에서는 **슬롯만** 제공한다. ``method != "none"`` 이면
    ``services/training_service.run_training`` 이 안전하게 downgrade 후
    ``Event.TRAINING_TUNING_DOWNGRADED`` 을 1회 emit 한다.
    §11 에서 `ml/tuners.py` + 통합 로직이 추가되면 즉시 활성화된다.
    """

    method: TuningMethod = "none"
    cv_folds: int = 3
    max_iter: int | None = None
    timeout_sec: int | None = None

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds 는 2 이상이어야 합니다.")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter 는 양의 정수여야 합니다.")
        if self.timeout_sec is not None and self.timeout_sec <= 0:
            raise ValueError("timeout_sec 는 양의 정수여야 합니다.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """학습 실행 입력 파라미터 (UI → Service → ML)."""

    dataset_id: int
    task_type: TaskType
    target_column: str
    excluded_columns: tuple[str, ...] = ()
    test_size: float = 0.2
    metric_key: str = ""  # 비어 있으면 기본 metric (registry 기본값)
    job_name: str | None = None
    preprocessing: PreprocessingConfig | None = None
    # §10.3 (FR-067): 학습 후보 필터. None = 전체(기존 동작).
    algorithms: tuple[str, ...] | None = None
    # §10.3 (§11 스키마 선반영): 튜닝 설정. None 또는 method="none" = 튜닝 비활성.
    tuning: TuningConfig | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size 는 (0, 1) 범위여야 합니다.")
        if self.task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type 은 classification/regression 중 하나여야 합니다: {self.task_type}"
            )
        if not self.target_column:
            raise ValueError("target_column 은 비어 있을 수 없습니다.")
        # Cross-field: SMOTE 는 분류 전용.
        if (
            self.preprocessing is not None
            and self.preprocessing.imbalance == "smote"
            and self.task_type == "regression"
        ):
            raise ValueError("SMOTE 는 분류(classification) 작업 전용입니다.")
        # §10.3: algorithms 필드 검증 — 빈 튜플/중복 금지.
        if self.algorithms is not None:
            if len(self.algorithms) == 0:
                raise ValueError("최소 1개 알고리즘을 선택해야 합니다.")
            if len(set(self.algorithms)) != len(self.algorithms):
                raise ValueError(f"algorithms 에 중복 이름이 있습니다: {list(self.algorithms)}")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """학습에 실제로 쓰인 피처 구조. 예측 시 입력 검증 기준.

    §9.1 확장: ``datetime`` (원본 datetime 컬럼), ``derived`` (파생 피처).
    ``input_columns`` 는 예측 시 사용자가 제공해야 할 **원본** 컬럼만 반환한다 —
    datetime/derived 는 파이프라인 내부에서 전개되므로 원본만 노출한다.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    datetime: tuple[str, ...] = ()
    derived: tuple[DerivedFeature, ...] = ()

    @property
    def input_columns(self) -> tuple[str, ...]:
        """target 을 제외한 예측 입력 컬럼 순서 (원본 numeric + categorical)."""
        return self.numeric + self.categorical

    def to_dict(self) -> dict[str, Any]:
        """저장 가능한 JSON 호환 dict 로 직렬화."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "target": self.target,
            "categories": {k: list(v) for k, v in self.categories.items()},
            "datetime": list(self.datetime),
            "derived": [{"name": d.name, "source": d.source, "kind": d.kind} for d in self.derived],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSchema:
        """dict → FeatureSchema. 구 아티팩트(datetime/derived 키 부재) 호환."""
        derived_raw = data.get("derived", [])
        derived = tuple(
            DerivedFeature(
                name=str(d["name"]),
                source=str(d["source"]),
                kind=str(d["kind"]),
            )
            for d in derived_raw
        )
        return cls(
            numeric=tuple(data.get("numeric", [])),
            categorical=tuple(data.get("categorical", [])),
            target=str(data["target"]),
            categories={k: tuple(v) for k, v in data.get("categories", {}).items()},
            datetime=tuple(data.get("datetime", [])),
            derived=derived,
        )


@dataclass(frozen=True, slots=True)
class ScoredModel:
    """개별 알고리즘 학습/평가 결과. 실패 시 ``status='failed'`` + ``error``."""

    algo_name: str
    status: ModelStatus
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    train_time_ms: int = 0

    @property
    def is_success(self) -> bool:
        return self.status == "success"
