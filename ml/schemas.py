"""ML 엔진 내부 도메인 스키마 (IMPLEMENTATION_PLAN §3.1a).

원칙:
- 이 모듈은 **Streamlit / SQLAlchemy / DB 의존 없이** 동작해야 한다. (룰: ml-engine.mdc)
- 모든 구조체는 ``@dataclass(frozen=True, slots=True)`` 로 불변.
- 리스트/딕셔너리 대신 ``tuple`` 을 우선 사용해 해시 가능성을 확보.
- ORM 엔터티(`repositories.models.*`) 와 1:1 매핑이 아닌 점 주의.
  ORM 은 저장 형식, 이 스키마는 **연산 입력/출력** 형식.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["classification", "regression"]
ModelStatus = Literal["success", "failed"]


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
class TrainingConfig:
    """학습 실행 입력 파라미터 (UI → Service → ML)."""

    dataset_id: int
    task_type: TaskType
    target_column: str
    excluded_columns: tuple[str, ...] = ()
    test_size: float = 0.2
    metric_key: str = ""  # 비어 있으면 기본 metric (registry 기본값)
    job_name: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size 는 (0, 1) 범위여야 합니다.")
        if self.task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type 은 classification/regression 중 하나여야 합니다: {self.task_type}"
            )
        if not self.target_column:
            raise ValueError("target_column 은 비어 있을 수 없습니다.")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """학습에 실제로 쓰인 피처 구조. 예측 시 입력 검증 기준."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def input_columns(self) -> tuple[str, ...]:
        """target 을 제외한 예측 입력 컬럼 순서."""
        return self.numeric + self.categorical

    def to_dict(self) -> dict[str, Any]:
        """저장 가능한 JSON 호환 dict 로 직렬화."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "target": self.target,
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSchema:
        return cls(
            numeric=tuple(data.get("numeric", [])),
            categorical=tuple(data.get("categorical", [])),
            target=str(data["target"]),
            categories={k: tuple(v) for k, v in data.get("categories", {}).items()},
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
