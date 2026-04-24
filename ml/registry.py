"""알고리즘 레지스트리 (IMPLEMENTATION_PLAN §3.2 / §10.1~§10.2, FR-062, FR-067~069).

설계:
- ``AlgoSpec`` 는 이름/태스크/factory + 부가 메타(optional backend 여부, param_grid)
  를 묶어 Service 계층이 순회하기 쉽게 한다.
- factory 는 **매번 새 estimator 인스턴스** 를 반환해야 한다(교차검증/재실행 안전성).
- XGBoost / LightGBM / CatBoost 는 환경에 따라 미설치이거나 네이티브 런타임(libomp 등)
  누락으로 import 자체가 실패할 수 있으므로 import 가드로 누락 시 skip 한다. 이 때:
    * 조용히 없어지지 않도록 ``optional_backends_status()`` 로 사유를 조회 가능하게 하고
    * 첫 import 시점에 한 번 구조화 로그(``log_event``)로 남긴다.
- ``random_state`` 는 ``settings.RANDOM_SEED`` 일괄 적용.
- ``param_grid`` 는 §11 하이퍼파라미터 튜너가 소비할 메타데이터. §10 에서는 저장만
  하고 실행하지 않는다(튜닝 비활성 = 기본 factory 호출 = 현재 동작 유지).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config.settings import settings
from utils.log_utils import get_logger, log_event

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = get_logger(__name__)

TaskType = Literal["classification", "regression"]
Estimator = Any  # sklearn-compatible


@dataclass(frozen=True, slots=True)
class AlgoSpec:
    """알고리즘 등록 정보.

    Attributes:
        name: task 내에서 unique 해야 한다 (§10.2 불변식).
        task: ``"classification"`` | ``"regression"``.
        factory: 매 호출 시 fresh estimator 인스턴스를 반환.
        default_metric: UI 에서 기준 지표 미지정 시 사용.
        is_optional_backend: xgboost/lightgbm/catboost 처럼 런타임 의존에 따라
            skip 될 수 있는 backend 면 True. UI 가 "미설치" 안내를 분기하는 데 사용.
        param_grid: §11 튜너가 소비할 (param_name -> 후보 tuple) 매핑. 없으면 None.
    """

    name: str
    task: TaskType
    factory: Callable[[], Estimator]
    default_metric: str = ""
    is_optional_backend: bool = False
    param_grid: Mapping[str, tuple[Any, ...]] | None = field(default=None)


_RANDOM_STATE = settings.RANDOM_SEED


# ---------------------------------------------------------- Classification


def _logistic_regression() -> Estimator:
    return LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE, n_jobs=None)


def _decision_tree_clf() -> Estimator:
    return DecisionTreeClassifier(random_state=_RANDOM_STATE)


def _random_forest_clf() -> Estimator:
    return RandomForestClassifier(n_estimators=200, random_state=_RANDOM_STATE, n_jobs=-1)


def _hist_gradient_boosting_clf() -> Estimator:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        early_stopping="auto",
        random_state=_RANDOM_STATE,
    )


def _extra_trees_clf() -> Estimator:
    return ExtraTreesClassifier(n_estimators=300, random_state=_RANDOM_STATE, n_jobs=-1)


def _gradient_boosting_clf() -> Estimator:
    return GradientBoostingClassifier(n_estimators=200, random_state=_RANDOM_STATE)


def _kneighbors_clf() -> Estimator:
    return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)


_CLASSIFICATION_SPECS: list[AlgoSpec] = [
    AlgoSpec("logistic_regression", "classification", _logistic_regression, "f1"),
    AlgoSpec("decision_tree", "classification", _decision_tree_clf, "f1"),
    AlgoSpec("random_forest", "classification", _random_forest_clf, "f1"),
    AlgoSpec(
        "hist_gradient_boosting",
        "classification",
        _hist_gradient_boosting_clf,
        "f1",
        param_grid={
            "learning_rate": (0.05, 0.1, 0.2),
            "max_iter": (200, 400),
        },
    ),
    AlgoSpec(
        "extra_trees",
        "classification",
        _extra_trees_clf,
        "f1",
        param_grid={
            "n_estimators": (200, 500),
            "max_depth": (None, 10, 20),
        },
    ),
    AlgoSpec(
        "gradient_boosting",
        "classification",
        _gradient_boosting_clf,
        "f1",
        param_grid={
            "n_estimators": (100, 200),
            "learning_rate": (0.05, 0.1),
        },
    ),
    AlgoSpec(
        "kneighbors",
        "classification",
        _kneighbors_clf,
        "f1",
        param_grid={
            "n_neighbors": (3, 5, 10),
            "weights": ("uniform", "distance"),
        },
    ),
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


def _hist_gradient_boosting_reg() -> Estimator:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        early_stopping="auto",
        random_state=_RANDOM_STATE,
    )


def _extra_trees_reg() -> Estimator:
    return ExtraTreesRegressor(n_estimators=300, random_state=_RANDOM_STATE, n_jobs=-1)


def _gradient_boosting_reg() -> Estimator:
    return GradientBoostingRegressor(n_estimators=200, random_state=_RANDOM_STATE)


def _kneighbors_reg() -> Estimator:
    return KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)


def _elastic_net() -> Estimator:
    # alpha/l1_ratio 는 §10.1 설계값. max_iter=5000 으로 수렴 안정성 확보.
    return ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=_RANDOM_STATE)


def _decision_tree_reg() -> Estimator:
    return DecisionTreeRegressor(random_state=_RANDOM_STATE)


_REGRESSION_SPECS: list[AlgoSpec] = [
    AlgoSpec("linear", "regression", _linear_regression, "rmse"),
    AlgoSpec("ridge", "regression", _ridge, "rmse"),
    AlgoSpec("lasso", "regression", _lasso, "rmse"),
    AlgoSpec("random_forest", "regression", _random_forest_reg, "rmse"),
    AlgoSpec(
        "hist_gradient_boosting",
        "regression",
        _hist_gradient_boosting_reg,
        "rmse",
        param_grid={
            "learning_rate": (0.05, 0.1, 0.2),
            "max_iter": (200, 400),
        },
    ),
    AlgoSpec(
        "extra_trees",
        "regression",
        _extra_trees_reg,
        "rmse",
        param_grid={
            "n_estimators": (200, 500),
            "max_depth": (None, 10, 20),
        },
    ),
    AlgoSpec(
        "gradient_boosting",
        "regression",
        _gradient_boosting_reg,
        "rmse",
        param_grid={
            "n_estimators": (100, 200),
            "learning_rate": (0.05, 0.1),
        },
    ),
    AlgoSpec(
        "kneighbors",
        "regression",
        _kneighbors_reg,
        "rmse",
        param_grid={
            "n_neighbors": (3, 5, 10),
            "weights": ("uniform", "distance"),
        },
    ),
    AlgoSpec(
        "elastic_net",
        "regression",
        _elastic_net,
        "rmse",
        param_grid={
            "alpha": (0.1, 0.5, 1.0),
            "l1_ratio": (0.2, 0.5, 0.8),
        },
    ),
    AlgoSpec(
        "decision_tree",
        "regression",
        _decision_tree_reg,
        "rmse",
        param_grid={
            "max_depth": (None, 5, 10),
            "min_samples_split": (2, 5),
        },
    ),
]


# -------------------------------------- Optional: XGBoost / LightGBM / CatBoost


@dataclass(frozen=True, slots=True)
class OptionalBackendStatus:
    """선택 백엔드(xgboost/lightgbm/catboost) 로딩 상태.

    ``available=False`` 일 때 ``reason`` 에 축약된 원인이 담긴다. 원문 예외는
    ``error`` 에 보관돼 진단에 활용 가능하다. macOS 에서 흔한 `libomp` 누락은
    `reason` 에 "libomp" 토큰이 포함된다.
    """

    name: str
    available: bool
    reason: str = ""
    error: str = ""


_OPTIONAL_STATUS: dict[str, OptionalBackendStatus] = {}


def _summarize_reason(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    if "libomp" in low or "openmp" in low:
        return "libomp 미설치 (macOS: `brew install libomp` 필요)"
    if isinstance(exc, ModuleNotFoundError):
        return "패키지 미설치 (pip install 필요)"
    return f"{type(exc).__name__}: {msg.splitlines()[0][:120]}"


def _record_backend_status(name: str, exc: BaseException | None) -> None:
    if exc is None:
        _OPTIONAL_STATUS[name] = OptionalBackendStatus(name=name, available=True)
        return
    reason = _summarize_reason(exc)
    _OPTIONAL_STATUS[name] = OptionalBackendStatus(
        name=name, available=False, reason=reason, error=f"{type(exc).__name__}: {exc}"
    )
    log_event(
        logger,
        "registry.optional_backend_skipped",
        backend=name,
        reason=reason,
        error_type=type(exc).__name__,
    )


def _try_register_xgboost() -> None:
    # macOS 는 libomp 미설치 시 import 자체가 `XGBoostError` 로 실패할 수 있다.
    # 이 경우도 선택 알고리즘 skip 으로 취급해 전체 파이프라인을 차단하지 않지만,
    # `_record_backend_status` 로 사유는 반드시 남긴다.
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:  # pragma: no cover - 플랫폼/런타임 의존
        _record_backend_status("xgboost", exc)
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

    _CLASSIFICATION_SPECS.append(
        AlgoSpec("xgboost", "classification", _xgb_clf, "f1", is_optional_backend=True)
    )
    _REGRESSION_SPECS.append(
        AlgoSpec("xgboost", "regression", _xgb_reg, "rmse", is_optional_backend=True)
    )
    _record_backend_status("xgboost", None)


def _try_register_lightgbm() -> None:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as exc:  # pragma: no cover - 플랫폼/런타임 의존
        _record_backend_status("lightgbm", exc)
        return

    def _lgbm_clf() -> Estimator:
        return LGBMClassifier(random_state=_RANDOM_STATE, n_estimators=300, n_jobs=-1)

    def _lgbm_reg() -> Estimator:
        return LGBMRegressor(random_state=_RANDOM_STATE, n_estimators=300, n_jobs=-1)

    _CLASSIFICATION_SPECS.append(
        AlgoSpec("lightgbm", "classification", _lgbm_clf, "f1", is_optional_backend=True)
    )
    _REGRESSION_SPECS.append(
        AlgoSpec("lightgbm", "regression", _lgbm_reg, "rmse", is_optional_backend=True)
    )
    _record_backend_status("lightgbm", None)


def _try_register_catboost() -> None:
    # CatBoost 는 `requirements-optional.txt` 에만 명시돼 기본 설치에는 포함되지
    # 않는다. 미설치 시 다른 optional backend 와 동일하게 `skip + status` 처리.
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception as exc:  # pragma: no cover - 선택 의존
        _record_backend_status("catboost", exc)
        return

    def _catboost_clf() -> Estimator:
        # allow_writing_files=False: 학습 중 catboost_info/ 디렉토리 부산물 방지.
        # verbose=0: 진행 로그가 stdout 을 도배하지 않도록.
        return CatBoostClassifier(
            iterations=300,
            verbose=0,
            random_seed=_RANDOM_STATE,
            allow_writing_files=False,
        )

    def _catboost_reg() -> Estimator:
        return CatBoostRegressor(
            iterations=300,
            verbose=0,
            random_seed=_RANDOM_STATE,
            allow_writing_files=False,
        )

    _CLASSIFICATION_SPECS.append(
        AlgoSpec("catboost", "classification", _catboost_clf, "f1", is_optional_backend=True)
    )
    _REGRESSION_SPECS.append(
        AlgoSpec("catboost", "regression", _catboost_reg, "rmse", is_optional_backend=True)
    )
    _record_backend_status("catboost", None)


_try_register_xgboost()
_try_register_lightgbm()
_try_register_catboost()


def optional_backends_status() -> list[OptionalBackendStatus]:
    """선택 백엔드(xgboost/lightgbm/catboost) 의 현재 로딩 상태 스냅샷.

    ``make doctor`` / 학습 페이지에서 "이 알고리즘이 왜 후보에 없는지" 를 사용자에게
    노출하기 위한 진단 API. 조회 순서는 등록 시도 순서(xgboost → lightgbm → catboost).
    """
    return [
        _OPTIONAL_STATUS[n] for n in ("xgboost", "lightgbm", "catboost") if n in _OPTIONAL_STATUS
    ]


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


__all__ = [
    "AlgoSpec",
    "OptionalBackendStatus",
    "available_algorithms",
    "get_spec",
    "get_specs",
    "optional_backends_status",
]
