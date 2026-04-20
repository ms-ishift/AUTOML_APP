"""ml.trainers 단위 테스트 (IMPLEMENTATION_PLAN §3.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocess import build_preprocessor
from ml.registry import AlgoSpec, get_specs
from ml.trainers import TrainedModel, split_dataset, train_all


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "income": rng.normal(50, 15, size=n),
            "city": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y = pd.Series(
        (df["income"] + df["age"] * 0.2 > 55).astype(int),
        name="target",
    )
    return df, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=n),
            "x2": rng.normal(0, 1, size=n),
            "cat": rng.choice(["p", "q"], size=n),
        }
    )
    y = pd.Series(df["x1"] * 2 + df["x2"] + rng.normal(0, 0.1, size=n), name="y")
    return df, y


class TestSplitDataset:
    def test_classification_stratify(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        X_tr, X_te, y_tr, y_te = split_dataset(X, y, test_size=0.25, task_type="classification")
        assert len(X_tr) + len(X_te) == len(X)
        # stratify 가 적용되면 train/test 의 클래스 비율이 유사해야 한다
        ratio_tr = y_tr.mean()
        ratio_te = y_te.mean()
        assert abs(ratio_tr - ratio_te) < 0.15

    def test_regression_no_stratify(self, regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = regression_data
        X_tr, X_te, y_tr, y_te = split_dataset(X, y, test_size=0.3, task_type="regression")
        assert len(X_te) == pytest.approx(len(X) * 0.3, abs=1)

    def test_invalid_test_size(self, regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = regression_data
        with pytest.raises(ValueError):
            split_dataset(X, y, test_size=0.0, task_type="regression")
        with pytest.raises(ValueError):
            split_dataset(X, y, test_size=1.0, task_type="regression")


class TestTrainAll:
    def test_classification_all_success(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        specs = get_specs("classification")
        pre = build_preprocessor(["age", "income"], ["city"])
        results = train_all(specs, pre, X, y)

        assert len(results) == len(specs)
        for r in results:
            assert isinstance(r, TrainedModel)
            assert r.is_success, f"{r.algo_name} failed: {r.error}"
            assert r.estimator is not None
            assert r.train_time_ms >= 0

    def test_regression_all_success(self, regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = regression_data
        specs = get_specs("regression")
        pre = build_preprocessor(["x1", "x2"], ["cat"])
        results = train_all(specs, pre, X, y)
        for r in results:
            assert r.is_success, f"{r.algo_name} failed: {r.error}"

    def test_individual_failure_is_isolated(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])

        def _broken_factory():  # type: ignore[no-untyped-def]
            raise RuntimeError("synthetic boom")

        specs = [
            *get_specs("classification")[:1],
            AlgoSpec("broken", "classification", _broken_factory, "f1"),
        ]
        results = train_all(specs, pre, X, y)
        assert len(results) == 2
        assert results[0].is_success
        assert not results[1].is_success
        assert results[1].error is not None
        assert "synthetic boom" in results[1].error

    def test_progress_callback_receives_updates(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:2]

        calls: list[tuple[int, int, str, str]] = []

        def _on_progress(idx: int, total: int, name: str, status: str) -> None:
            calls.append((idx, total, name, status))

        train_all(specs, pre, X, y, on_progress=_on_progress)
        assert [c[0] for c in calls] == [1, 2]
        assert all(c[1] == 2 for c in calls)
        assert {c[2] for c in calls} == {s.name for s in specs}

    def test_preprocessor_is_not_shared_across_specs(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """같은 preprocessor 인스턴스를 재사용해도 (clone 덕분에) 각 Pipeline 은 독립이어야 한다."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")
        results = train_all(specs, pre, X, y)
        pipelines = [r.estimator for r in results]
        preprocessors = [p.named_steps["preprocessor"] for p in pipelines]
        # 동일 객체가 아니어야 한다
        assert len({id(p) for p in preprocessors}) == len(preprocessors)
