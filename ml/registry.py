"""알고리즘 레지스트리 (IMPLEMENTATION_PLAN §3.2, FR-062).

설계:
- ``AlgoSpec`` 는 이름/태스크/factory 를 묶어 Service 계층이 순회하기 쉽게 한다.
- factory 는 **매번 새 estimator 인스턴스** 를 반환해야 한다(교차검증/재실행 안전성).
- XGBoost / LightGBM 은 환경에 따라 미설치일 수 있으므로 import 가드로 누락 시 skip.
- ``random_state`` 는 ``settings.RANDOM_SEED`` 일괄 적용.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.tree import DecisionTreeClassifier

from config.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

TaskType = Literal["classification", "regression"]
Estimator = Any  # sklearn-compatible


@dataclass(frozen=True, slots=True)
class AlgoSpec:
    """알고리즘 등록 정보."""

    name: str
    task: TaskType
    factory: Callable[[], Estimator]
    default_metric: str = ""


_RANDOM_STATE = settings.RANDOM_SEED


# ---------------------------------------------------------- Classification


def _logistic_regression() -> Estimator:
    return LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE, n_jobs=None)


def _decision_tree_clf() -> Estimator:
    return DecisionTreeClassifier(random_state=_RANDOM_STATE)


def _random_forest_clf() -> Estimator:
    return RandomForestClassifier(n_estimators=200, random_state=_RANDOM_STATE, n_jobs=-1)


_CLASSIFICATION_SPECS: list[AlgoSpec] = [
    AlgoSpec("logistic_regression", "classification", _logistic_regression, "f1"),
    AlgoSpec("decision_tree", "classification", _decision_tree_clf, "f1"),
    AlgoSpec("random_forest", "classification", _random_forest_clf, "f1"),
]


# -------------------------------------------------------------- Regression


def _linear_regression() -> Estimator:
    return LinearRegression()


def _ridge() -> Estimator:
    return Ridge(random_state=_RANDOM_STATE)


def _lasso() -> Estimator:
    return Lasso(random_state=_RANDOM_STATE)


def _random_forest_reg() -> Estimator:
    return RandomForestRegressor(n_estimators=200, random_state=_RANDOM_STATE, n_jobs=-1)


_REGRESSION_SPECS: list[AlgoSpec] = [
    AlgoSpec("linear", "regression", _linear_regression, "rmse"),
    AlgoSpec("ridge", "regression", _ridge, "rmse"),
    AlgoSpec("lasso", "regression", _lasso, "rmse"),
    AlgoSpec("random_forest", "regression", _random_forest_reg, "rmse"),
]


# -------------------------------------------- Optional: XGBoost / LightGBM


def _try_register_xgboost() -> None:
    # macOS 는 libomp 미설치 시 import 자체가 `XGBoostError` 로 실패할 수 있다.
    # 이 경우도 선택 알고리즘 skip 으로 취급해 전체 파이프라인을 차단하지 않는다.
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:  # pragma: no cover
        return

    def _xgb_clf() -> Estimator:
        return XGBClassifier(
            random_state=_RANDOM_STATE,
            n_estimators=300,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )

    def _xgb_reg() -> Estimator:
        return XGBRegressor(
            random_state=_RANDOM_STATE,
            n_estimators=300,
            n_jobs=-1,
        )

    _CLASSIFICATION_SPECS.append(AlgoSpec("xgboost", "classification", _xgb_clf, "f1"))
    _REGRESSION_SPECS.append(AlgoSpec("xgboost", "regression", _xgb_reg, "rmse"))


def _try_register_lightgbm() -> None:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:  # pragma: no cover
        return

    def _lgbm_clf() -> Estimator:
        return LGBMClassifier(random_state=_RANDOM_STATE, n_estimators=300, n_jobs=-1)

    def _lgbm_reg() -> Estimator:
        return LGBMRegressor(random_state=_RANDOM_STATE, n_estimators=300, n_jobs=-1)

    _CLASSIFICATION_SPECS.append(AlgoSpec("lightgbm", "classification", _lgbm_clf, "f1"))
    _REGRESSION_SPECS.append(AlgoSpec("lightgbm", "regression", _lgbm_reg, "rmse"))


_try_register_xgboost()
_try_register_lightgbm()


# ------------------------------------------------------------------ Public


def get_specs(task_type: TaskType) -> list[AlgoSpec]:
    """지정된 태스크의 전체 알고리즘 스펙 목록을 반환 (호출 시점 기준 복사본)."""
    if task_type == "classification":
        return list(_CLASSIFICATION_SPECS)
    if task_type == "regression":
        return list(_REGRESSION_SPECS)
    raise ValueError(f"지원하지 않는 task_type: {task_type}")


def get_spec(task_type: TaskType, algo_name: str) -> AlgoSpec:
    """이름으로 스펙 1건 조회. 미등록이면 ``KeyError``."""
    for spec in get_specs(task_type):
        if spec.name == algo_name:
            return spec
    raise KeyError(f"알고리즘 미등록: task={task_type}, name={algo_name}")


def available_algorithms(task_type: TaskType) -> list[str]:
    return [s.name for s in get_specs(task_type)]
