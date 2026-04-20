"""평가 (IMPLEMENTATION_PLAN §3.6, FR-064, FR-071, FR-072).

계약:
- 분류 기본: accuracy / f1 / roc_auc
    - 다중 클래스: f1=macro, roc_auc=ovr(macro)
    - roc_auc 는 ``predict_proba`` 가 가능한 estimator 에서만 계산
- 회귀 기본: rmse / mae / r2
- ``score_models`` 는 trainers 의 ``TrainedModel`` 목록과 테스트셋을 받아 ``ScoredModel`` 목록을 반환.
  - 학습 단계에서 실패한 모델은 그대로 failed 로 전달(메트릭 빈 dict).
- ``select_best`` 는 metric 의 방향(max/min)을 내장 테이블에서 조회.
- 시각화는 이 모듈에서 matplotlib 을 호출하지 않고, UI 가 그리기 좋은 dict 만 반환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ml.schemas import ScoredModel

if TYPE_CHECKING:
    import pandas as pd

    from ml.trainers import TrainedModel

MetricDirection = Literal["max", "min"]

METRIC_DIRECTIONS: Final[dict[str, MetricDirection]] = {
    "accuracy": "max",
    "f1": "max",
    "roc_auc": "max",
    "rmse": "min",
    "mae": "min",
    "r2": "max",
}

CLASSIFICATION_METRICS: Final[tuple[str, ...]] = ("accuracy", "f1", "roc_auc")
REGRESSION_METRICS: Final[tuple[str, ...]] = ("rmse", "mae", "r2")


def _classification_metrics(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_pred = estimator.predict(X_test)
    classes = np.unique(y_test)
    is_binary = len(classes) == 2

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                average="binary" if is_binary else "macro",
                zero_division=0,
            )
        ),
    }

    if hasattr(estimator, "predict_proba"):
        try:
            y_proba = estimator.predict_proba(X_test)
            if is_binary:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception:  # noqa: BLE001 - roc_auc 는 보조 지표, 실패해도 accuracy/f1 유지
            pass

    return metrics


def _regression_metrics(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_pred = estimator.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def score_models(
    trained: list[TrainedModel],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> list[ScoredModel]:
    """학습 결과를 받아 각 모델의 메트릭을 산출. 학습 실패/채점 실패는 failed 로 귀속."""
    if task_type not in ("classification", "regression"):
        raise ValueError(f"지원하지 않는 task_type: {task_type}")

    results: list[ScoredModel] = []
    for t in trained:
        if not t.is_success or t.estimator is None:
            results.append(
                ScoredModel(
                    algo_name=t.algo_name,
                    status="failed",
                    metrics={},
                    error=t.error,
                    train_time_ms=t.train_time_ms,
                )
            )
            continue

        try:
            if task_type == "classification":
                metrics = _classification_metrics(t.estimator, X_test, y_test)
            else:
                metrics = _regression_metrics(t.estimator, X_test, y_test)
            results.append(
                ScoredModel(
                    algo_name=t.algo_name,
                    status="success",
                    metrics=metrics,
                    train_time_ms=t.train_time_ms,
                )
            )
        except Exception as e:  # noqa: BLE001 - 개별 채점 실패는 전체 중단 없음
            results.append(
                ScoredModel(
                    algo_name=t.algo_name,
                    status="failed",
                    metrics={},
                    error=f"scoring failed: {e}",
                    train_time_ms=t.train_time_ms,
                )
            )
    return results


def select_best(
    scored: list[ScoredModel],
    metric_key: str,
) -> ScoredModel | None:
    """metric_key 기준 최고 성능 모델. 성공 + 해당 메트릭 포함 건만 후보."""
    direction = METRIC_DIRECTIONS.get(metric_key)
    if direction is None:
        raise ValueError(f"알 수 없는 metric: {metric_key}")

    candidates = [s for s in scored if s.is_success and metric_key in s.metrics]
    if not candidates:
        return None

    sign = 1.0 if direction == "max" else -1.0
    return max(candidates, key=lambda s: sign * s.metrics[metric_key])


def confusion_matrix_data(
    y_true: pd.Series | np.ndarray | list[Any],
    y_pred: pd.Series | np.ndarray | list[Any],
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """UI 가 그릴 수 있도록 혼동행렬을 직렬화 가능한 dict 로 반환."""
    if labels is None:
        all_labels = sorted({*list(y_true), *list(y_pred)}, key=lambda x: str(x))
    else:
        all_labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    return {
        "labels": [str(label) for label in all_labels],
        "matrix": cm.tolist(),
    }


def regression_scatter_data(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> dict[str, list[float]]:
    """회귀 산점도용 y_true/y_pred 배열."""
    return {
        "y_true": [float(v) for v in y_true],
        "y_pred": [float(v) for v in y_pred],
    }


# ----------------------------------------------------------- plot_data builder


# 회귀 산점도에 저장할 최대 샘플 수. 플롯 렌더 경량화를 위해 상한을 둔다.
_SCATTER_MAX_POINTS: Final[int] = 2000


def build_plot_data(
    trained: list[TrainedModel],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> dict[str, dict[str, Any]]:
    """알고리즘 별로 UI 렌더용 플롯 데이터를 만든다.

    반환: ``{algo_name: plot_data}``
        - 분류: ``{"kind": "confusion_matrix", "labels": [...], "matrix": [[...]]}``
        - 회귀: ``{"kind": "regression_scatter", "y_true": [...], "y_pred": [...]}``

    실패한 모델(``is_success=False``)은 결과 dict 에 포함하지 않는다 (키 누락).
    개별 ``predict`` 실패도 조용히 스킵 — 이 헬퍼는 보조 시각화이므로 본 메트릭 산출을 방해하지 않는다.
    """
    if task_type not in ("classification", "regression"):
        raise ValueError(f"지원하지 않는 task_type: {task_type}")

    result: dict[str, dict[str, Any]] = {}
    for t in trained:
        if not t.is_success or t.estimator is None:
            continue
        try:
            y_pred = t.estimator.predict(X_test)
        except Exception:  # noqa: BLE001 - 보조 데이터 수집 실패는 무시
            continue
        if task_type == "classification":
            data = confusion_matrix_data(y_test, y_pred)
            result[t.algo_name] = {"kind": "confusion_matrix", **data}
        else:
            # 대량 데이터셋 방어
            y_true_list = list(y_test)
            y_pred_list = list(y_pred)
            if len(y_true_list) > _SCATTER_MAX_POINTS:
                # 균등 간격 샘플링 (순서 보존)
                step = max(1, len(y_true_list) // _SCATTER_MAX_POINTS)
                y_true_list = y_true_list[::step][:_SCATTER_MAX_POINTS]
                y_pred_list = y_pred_list[::step][:_SCATTER_MAX_POINTS]
            data = regression_scatter_data(y_true_list, y_pred_list)
            result[t.algo_name] = {"kind": "regression_scatter", **data}
    return result
