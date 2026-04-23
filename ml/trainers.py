"""학습 실행 (IMPLEMENTATION_PLAN §3.5 / §9.6, FR-061, FR-063, FR-057, NFR-004).

설계:
- ``split_dataset`` 는 분류일 때 ``stratify=y`` 적용. random_state 는 settings.RANDOM_SEED.
- ``train_all`` 은 spec 목록을 순회하며 각 알고리즘을 **독립 Pipeline** 으로 학습한다.
  - 전처리기는 ``sklearn.base.clone`` 으로 복제 → fit 상태 공유 방지.
  - 개별 알고리즘 실패는 ``TrainedModel(status='failed')`` 로 기록하고 전체 중단 없음.
  - 학습 시간(ms)을 측정해 함께 저장.
- ``TrainedModel`` 은 메모리 전용 구조. 직렬화는 §3.7 artifacts 에서 Pipeline 전체를 joblib 으로.

§9.6 확장 (FR-057):
- ``train_all`` 에 `preprocess_cfg`, `balancer` 키워드 인자 추가.
  - 인자 생략 시 기존 동작과 완전히 동일 (테스트 회귀 0).
  - ``balancer(estimator, X_train, y_train) -> (estimator, X_train, y_train)`` 는 각 spec
    의 fresh estimator 에 대해 fit 직전에 호출된다 (class_weight 설정, SMOTE 리샘플 등).
    **호출자 책임**: balancer 에 전달되는 ``X_train/y_train`` 은 반드시 train split 이후
    데이터여야 한다 (테스트 세트 리샘플링 금지 — §9.5 docstring 참조).
  - ``preprocess_cfg`` 는 현재 구현에서 직접 분기에 쓰이지 않고, 메타데이터/미래 확장용
    으로 예약된다 (caller 가 이미 preprocessor 와 balancer 를 구성해서 넘기기 때문).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from sklearn.compose import ColumnTransformer

    from ml.registry import AlgoSpec
    from ml.schemas import PreprocessingConfig

    ProgressCallback = Callable[[int, int, str, Literal["success", "failed"]], None]
    # (estimator, X_train, y_train) -> (estimator, X_train, y_train)
    BalancerCallable = Callable[
        [Any, "pd.DataFrame", "pd.Series"],
        tuple[Any, "pd.DataFrame", "pd.Series"],
    ]


TrainedStatus = Literal["success", "failed"]


@dataclass
class TrainedModel:
    """학습된 단일 알고리즘 결과.

    성공 시 ``estimator`` 는 fit 된 ``sklearn.pipeline.Pipeline``(preprocessor+model).
    실패 시 ``estimator`` 는 ``None``, ``error`` 에 메시지.
    """

    algo_name: str
    estimator: Any | None
    status: TrainedStatus
    train_time_ms: int
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status == "success"


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """train/test 분할. 분류면 stratify=y 시도, 실패 시(클래스 분포 이슈) stratify 없이 재시도."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 는 (0, 1) 범위여야 합니다.")

    stratify = y if task_type == "classification" else None
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=settings.RANDOM_SEED,
            stratify=stratify,
        )
    except ValueError:
        # 클래스별 샘플 수가 너무 적어 stratify 불가한 경우 fallback
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=settings.RANDOM_SEED,
        )


def _build_pipeline(preprocessor: ColumnTransformer, estimator: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", estimator)])


def train_all(
    specs: list[AlgoSpec],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    on_progress: ProgressCallback | None = None,
    preprocess_cfg: PreprocessingConfig | None = None,  # noqa: ARG001 — §9.6 예약
    balancer: BalancerCallable | None = None,
) -> list[TrainedModel]:
    """주어진 스펙을 순회해 각각 학습. 개별 실패는 ``TrainedModel(status='failed')`` 로 기록.

    ``on_progress(index, total, algo_name, status)`` 콜백으로 진행률을 외부에 보고할 수 있다.
    (§4.3 - Streamlit rerun 모델 호환을 위해 동기 콜백)

    §9.6 추가:
    - ``balancer`` 가 주어지면 spec 별 fresh estimator 에 대해 fit 직전에 호출.
      반환값 ``(estimator, X_use, y_use)`` 로 해당 pipeline 학습.
      balancer 호출 실패는 해당 spec 의 학습 실패로 격리 (기존 실패 격리 정책 유지).
    - ``preprocess_cfg`` 는 메타데이터/미래 확장용 예약 파라미터 (현재 분기에는 사용 안 함,
      호출자가 이미 preprocessor/balancer 를 구성해서 전달하는 책임 분리 구조).
    """
    results: list[TrainedModel] = []
    total = len(specs)
    for idx, spec in enumerate(specs):
        start = time.perf_counter()
        try:
            estimator = spec.factory()
            X_use: pd.DataFrame = X_train
            y_use: pd.Series = y_train
            if balancer is not None:
                estimator, X_use, y_use = balancer(estimator, X_train, y_train)
            pipeline = _build_pipeline(preprocessor, estimator)
            pipeline.fit(X_use, y_use)
            elapsed = int((time.perf_counter() - start) * 1000)
            results.append(
                TrainedModel(
                    algo_name=spec.name,
                    estimator=pipeline,
                    status="success",
                    train_time_ms=elapsed,
                )
            )
            status: TrainedStatus = "success"
        except Exception as e:  # noqa: BLE001 - 개별 알고리즘 실패 격리가 목적
            elapsed = int((time.perf_counter() - start) * 1000)
            results.append(
                TrainedModel(
                    algo_name=spec.name,
                    estimator=None,
                    status="failed",
                    train_time_ms=elapsed,
                    error=str(e),
                )
            )
            status = "failed"

        if on_progress is not None:
            # 콜백 실패가 학습 전체를 중단시키지 않도록 방어
            with contextlib.suppress(Exception):
                on_progress(idx + 1, total, spec.name, status)

    return results
