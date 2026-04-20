from __future__ import annotations

import pytest

from ml.registry import (
    AlgoSpec,
    available_algorithms,
    get_spec,
    get_specs,
)

CLASSIFICATION_CORE = {"logistic_regression", "decision_tree", "random_forest"}
REGRESSION_CORE = {"linear", "ridge", "lasso", "random_forest"}


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
