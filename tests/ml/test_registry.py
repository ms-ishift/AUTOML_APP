from __future__ import annotations

from typing import get_args

import pytest

from ml.registry import (
    AlgoSpec,
    OptionalBackendStatus,
    TaskType,
    _summarize_reason,
    available_algorithms,
    get_spec,
    get_specs,
    optional_backends_status,
)

CLASSIFICATION_CORE = {
    "logistic_regression",
    "decision_tree",
    "random_forest",
    # §10.1 Tier 1 추가 (sklearn 내장, 의존성 증가 0)
    "hist_gradient_boosting",
    "extra_trees",
    "gradient_boosting",
    "kneighbors",
}
REGRESSION_CORE = {
    "linear",
    "ridge",
    "lasso",
    "random_forest",
    # §10.1 Tier 1 추가
    "hist_gradient_boosting",
    "extra_trees",
    "gradient_boosting",
    "kneighbors",
    "elastic_net",
    "decision_tree",
}

# §10.1 Tier 1 신규 이름 집합 — factory smoke 에 재사용.
TIER1_NEW_CLASSIFICATION = {
    "hist_gradient_boosting",
    "extra_trees",
    "gradient_boosting",
    "kneighbors",
}
TIER1_NEW_REGRESSION = {
    "hist_gradient_boosting",
    "extra_trees",
    "gradient_boosting",
    "kneighbors",
    "elastic_net",
    "decision_tree",
}

OPTIONAL_BACKEND_NAMES = ("xgboost", "lightgbm", "catboost")


def test_classification_core_specs_exist() -> None:
    names = set(available_algorithms("classification"))
    missing = CLASSIFICATION_CORE - names
    assert not missing, f"분류 기본 알고리즘 누락: {missing}"


def test_regression_core_specs_exist() -> None:
    names = set(available_algorithms("regression"))
    missing = REGRESSION_CORE - names
    assert not missing, f"회귀 기본 알고리즘 누락: {missing}"


def test_get_specs_invalid_task() -> None:
    with pytest.raises(ValueError):
        get_specs("clustering")  # type: ignore[arg-type]


def test_get_spec_unknown() -> None:
    with pytest.raises(KeyError):
        get_spec("classification", "mystery_forest")


def test_factory_returns_fresh_instances() -> None:
    spec = get_spec("classification", "random_forest")
    a = spec.factory()
    b = spec.factory()
    assert a is not b


def test_all_specs_are_algospec_instances() -> None:
    for task in ("classification", "regression"):
        for s in get_specs(task):  # type: ignore[arg-type]
            assert isinstance(s, AlgoSpec)
            assert s.task == task
            assert callable(s.factory)
            assert s.name


def test_classification_defaults_metric_is_set() -> None:
    for s in get_specs("classification"):
        assert s.default_metric in {"f1", "accuracy", "roc_auc"}


def test_regression_defaults_metric_is_set() -> None:
    for s in get_specs("regression"):
        assert s.default_metric in {"rmse", "mae", "r2"}


def test_registry_does_not_import_streamlit_or_sqlalchemy() -> None:
    """ml 레이어는 UI/DB 에 의존하면 안 된다 (룰: ml-engine.mdc)."""
    import ml.registry as reg_mod

    src = reg_mod.__file__
    assert src is not None
    with open(src, encoding="utf-8") as f:
        text = f.read()
    assert "streamlit" not in text
    assert "sqlalchemy" not in text


# --------------------------------------- Optional backend visibility (§10.1)


def test_optional_backends_status_covers_all_three() -> None:
    """§10.1: xgboost / lightgbm / catboost 3건이 설치 여부와 무관하게 등장한다."""
    names = tuple(s.name for s in optional_backends_status())
    assert names == OPTIONAL_BACKEND_NAMES, names


def test_optional_backend_status_structure() -> None:
    for s in optional_backends_status():
        assert isinstance(s, OptionalBackendStatus)
        if s.available:
            assert s.reason == ""
        else:
            # skip 사유는 반드시 비어있지 않아 사용자가 읽고 복구할 수 있어야 한다.
            assert s.reason
            # libomp 누락 같은 흔한 케이스는 reason 안에 복구 힌트가 들어간다.
            assert len(s.reason) < 200


def test_summarize_reason_detects_libomp() -> None:
    """macOS 의 libomp 누락 메시지가 포함되면 사용자 친화적 사유로 축약한다."""
    err = OSError("dlopen(libxgboost.dylib): Library not loaded: @rpath/libomp.dylib")
    reason = _summarize_reason(err)
    assert "libomp" in reason
    assert "brew install libomp" in reason


def test_summarize_reason_detects_missing_module() -> None:
    reason = _summarize_reason(ModuleNotFoundError("No module named 'xgboost'"))
    assert "pip install" in reason


def test_skipped_backend_is_absent_from_specs() -> None:
    """status 가 skip 된 백엔드는 get_specs 결과에도 나타나지 않아야 한다."""
    skipped = {s.name for s in optional_backends_status() if not s.available}
    clf_names = set(available_algorithms("classification"))
    reg_names = set(available_algorithms("regression"))
    assert not (skipped & clf_names), f"skip 되었는데 분류에 등록됨: {skipped & clf_names}"
    assert not (skipped & reg_names), f"skip 되었는데 회귀에 등록됨: {skipped & reg_names}"


# -------------------------------- §10.2 AlgoSpec 메타데이터 확장


def test_algo_names_unique_per_task() -> None:
    """§10.2 불변식: 같은 task 내에서 name 은 중복될 수 없다."""
    for task in get_args(TaskType):
        names = [s.name for s in get_specs(task)]
        assert len(names) == len(set(names)), f"{task} 에 중복 이름: {names}"


def test_is_optional_backend_flag_is_correct() -> None:
    """§10.2: xgboost/lightgbm/catboost 는 is_optional_backend=True,
    그 외 sklearn 내장은 False 여야 한다."""
    for task in get_args(TaskType):
        for spec in get_specs(task):
            if spec.name in OPTIONAL_BACKEND_NAMES:
                assert spec.is_optional_backend is True, spec.name
            else:
                assert spec.is_optional_backend is False, spec.name


def test_tier1_models_have_param_grid() -> None:
    """§10.2 사용자 결정 B: Tier 1 6종 전부에 param_grid 가 채워져 있어야 한다."""
    clf = {s.name: s for s in get_specs("classification")}
    reg = {s.name: s for s in get_specs("regression")}
    for name in TIER1_NEW_CLASSIFICATION:
        grid = clf[name].param_grid
        assert grid and len(grid) >= 2, f"{name}(classification) param_grid 부족"
    for name in TIER1_NEW_REGRESSION:
        grid = reg[name].param_grid
        assert grid and len(grid) >= 2, f"{name}(regression) param_grid 부족"


def test_tier1_factories_return_sklearn_compatible_estimator() -> None:
    """§10.1: 신규 factory 가 get_params() 를 지원하는 sklearn-compatible 객체를 반환.

    실제 fit 은 비용이 커서 호출하지 않고 구조만 검증한다.
    """
    tier1_by_task = {
        "classification": TIER1_NEW_CLASSIFICATION,
        "regression": TIER1_NEW_REGRESSION,
    }
    for task, names in tier1_by_task.items():
        for name in names:
            spec = get_spec(task, name)  # type: ignore[arg-type]
            est = spec.factory()
            # sklearn BaseEstimator 인터페이스
            assert hasattr(est, "get_params")
            params = est.get_params()
            assert isinstance(params, dict)


def test_core_specs_have_no_param_grid_by_default() -> None:
    """§10.2: MVP 코어(logistic/linear/ridge/lasso/random_forest/decision_tree-clf)
    는 이번 스프린트에서 param_grid 를 채우지 않는다 (튜닝 본체는 §11)."""
    mvp_core = {
        "classification": {"logistic_regression", "random_forest", "decision_tree"},
        "regression": {"linear", "ridge", "lasso", "random_forest"},
    }
    for task, names in mvp_core.items():
        for spec in get_specs(task):  # type: ignore[arg-type]
            if spec.name in names:
                assert spec.param_grid is None, spec.name
