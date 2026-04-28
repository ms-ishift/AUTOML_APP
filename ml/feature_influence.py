"""전역 특성 영향도 (IMPLEMENTATION_PLAN §11, FR-094, FR-095).

순열 중요도(Phase A) 및 트리 내장 ``feature_importances_``(Phase B) 계산.
**Streamlit / SQLAlchemy 비의존** — 예외는 ``ValueError`` 로만 발생시키고 Service 가 도메인 예외로 변환한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# 상한: 순열 비용 폭주 방지 (Service 가 metrics 기반 subsample 과 조합)
DEFAULT_MAX_ROWS: int = 5000
DEFAULT_N_REPEATS: int = 8


def subsample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """행이 ``max_rows`` 초과면 무작위 부분 표본을 반환.

    반환: ``(X_out, y_out, n_total_before)`` — 부분 표본이면 ``len(X_out) < n_total_before``.
    """
    n = int(len(X))
    if n <= max_rows:
        return X, y, n
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=max_rows, replace=False)
    return X.iloc[idx], y.iloc[idx], n


def scoring_for_permutation(task_type: str, metric_key: str) -> str:
    """학습 잡의 ``metric_key`` 에 맞춘 sklearn ``scoring`` 문자열."""
    if task_type == "regression":
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }.get(metric_key, "neg_root_mean_squared_error")
    return {
        "accuracy": "accuracy",
        "f1": "f1_macro",
        "roc_auc": "roc_auc_ovr_weighted",
    }.get(metric_key, "accuracy")


def compute_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: str,
    metric_key: str,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int,
    n_jobs: int = -1,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[list[tuple[str, float, float]], int, int]:
    """fit 된 ``estimator``(보통 Pipeline)에 대해 순열 중요도를 계산한다.

    ``X`` 는 학습 시 ``prepare_xy`` 이후 **원시 피처 컬럼** DataFrame (파이프라인 1단계 입력).

    반환:
    - ``rows``: ``(feature_name, mean_importance, std)`` 를 mean 기준 내림차순 정렬.
    - ``n_eval``: 평가에 사용한 행 수 (subsample 후).
    - ``n_total``: subsample 전 원본 행 수.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X 는 pandas.DataFrame 이어야 합니다.")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X 또는 y 가 비어 있습니다.")

    X_eval, y_eval, n_total = subsample_xy(X, y, max_rows=max_rows, random_state=random_state)
    n_eval = len(X_eval)
    scoring = scoring_for_permutation(task_type, metric_key)

    try:
        result = permutation_importance(
            estimator,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    except (ValueError, TypeError) as exc:
        # roc_auc 등 일부 조합에서 실패 시 단순 스코어로 1회 폴백
        fallback = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"
        if scoring == fallback:
            raise ValueError(f"순열 중요도 계산 실패: {exc}") from exc
        result = permutation_importance(
            estimator,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=fallback,
            n_jobs=n_jobs,
        )

    names = list(X_eval.columns)
    if result.importances_mean.shape[0] != len(names):
        raise ValueError(
            "순열 중요도 결과 차원이 피처 수와 일치하지 않습니다. "
            f"expected={len(names)} got={result.importances_mean.shape[0]}"
        )

    rows = [
        (names[i], float(result.importances_mean[i]), float(result.importances_std[i]))
        for i in range(len(names))
    ]
    rows.sort(key=lambda t: -t[1])
    return rows, n_eval, n_total


def extract_builtin_transformed_importances(estimator: Any) -> list[tuple[str, float]] | None:
    """Pipeline 최종 추정기의 ``feature_importances_`` 를 **전처리 후** 피처 이름과 함께 반환.

    - ``named_steps['model']`` 에 ``feature_importances_`` 가 없으면 ``None``.
    - ``named_steps['preprocessor'].get_feature_names_out()`` 와 길이 불일치 시 ``None``.
    """
    if not hasattr(estimator, "named_steps"):
        return None
    steps = getattr(estimator, "named_steps", None)
    if not isinstance(steps, dict) or "model" not in steps:
        return None
    final = steps["model"]
    if not hasattr(final, "feature_importances_"):
        return None
    importances = np.asarray(final.feature_importances_, dtype=float)
    pre = steps.get("preprocessor")
    if pre is None:
        return None
    try:
        names = list(pre.get_feature_names_out())
    except (AttributeError, TypeError, ValueError):
        return None
    if len(names) != len(importances):
        return None
    pairs = [(str(names[i]), float(importances[i])) for i in range(len(names))]
    pairs.sort(key=lambda t: -t[1])
    return pairs


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_N_REPEATS",
    "compute_permutation_importance",
    "extract_builtin_transformed_importances",
    "scoring_for_permutation",
    "subsample_xy",
]
