from __future__ import annotations

import pytest

from ml.registry import (
    AlgoSpec,
    OptionalBackendStatus,
    _summarize_reason,
    available_algorithms,
    get_spec,
    get_specs,
    optional_backends_status,
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


# ---------------------------------------------- Optional backend visibility


def test_optional_backends_status_covers_both_backends() -> None:
    """xgboost / lightgbm 는 설치 여부와 무관하게 status 리스트에 2건 등장해야 한다."""
    names = [s.name for s in optional_backends_status()]
    assert names == ["xgboost", "lightgbm"], names


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
    err = OSError(
        "dlopen(libxgboost.dylib): Library not loaded: @rpath/libomp.dylib"
    )
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
