"""ml.evaluators 단위 테스트 (IMPLEMENTATION_PLAN §3.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluators import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    build_plot_data,
    confusion_matrix_data,
    regression_scatter_data,
    score_models,
    select_best,
)
from ml.preprocess import build_preprocessor
from ml.registry import get_specs
from ml.schemas import ScoredModel
from ml.trainers import TrainedModel, split_dataset, train_all


@pytest.fixture
def classification_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "income": rng.normal(50, 15, size=n),
            "city": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y = pd.Series((df["income"] + df["age"] * 0.2 > 55).astype(int), name="target")
    return split_dataset(df, y, test_size=0.3, task_type="classification")


@pytest.fixture
def regression_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=n),
            "x2": rng.normal(0, 1, size=n),
            "cat": rng.choice(["p", "q"], size=n),
        }
    )
    y = pd.Series(df["x1"] * 2 + df["x2"] + rng.normal(0, 0.1, size=n), name="y")
    return split_dataset(df, y, test_size=0.3, task_type="regression")


class TestScoreModelsClassification:
    def test_metrics_populated(
        self,
        classification_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_tr, X_te, y_tr, y_te = classification_split
        pre = build_preprocessor(["age", "income"], ["city"])
        trained = train_all(get_specs("classification"), pre, X_tr, y_tr)
        scored = score_models(trained, X_te, y_te, task_type="classification")

        assert len(scored) == len(trained)
        for s in scored:
            assert s.is_success
            assert "accuracy" in s.metrics
            assert "f1" in s.metrics
            assert 0.0 <= s.metrics["accuracy"] <= 1.0
            # roc_auc 는 predict_proba 가능한 알고리즘에서만 존재
            if "roc_auc" in s.metrics:
                assert 0.0 <= s.metrics["roc_auc"] <= 1.0

    def test_failed_trained_passthrough(self) -> None:
        trained = [
            TrainedModel(
                algo_name="boom",
                estimator=None,
                status="failed",
                train_time_ms=5,
                error="prev fail",
            )
        ]
        X_te = pd.DataFrame({"age": [1], "city": ["A"]})
        y_te = pd.Series([0])
        out = score_models(trained, X_te, y_te, task_type="classification")
        assert len(out) == 1
        assert not out[0].is_success
        assert out[0].error == "prev fail"
        assert out[0].metrics == {}

    def test_invalid_task_type_raises(self) -> None:
        with pytest.raises(ValueError):
            score_models([], pd.DataFrame(), pd.Series(dtype=float), task_type="bogus")


class TestScoreModelsRegression:
    def test_metrics_populated(
        self,
        regression_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_tr, X_te, y_tr, y_te = regression_split
        pre = build_preprocessor(["x1", "x2"], ["cat"])
        trained = train_all(get_specs("regression"), pre, X_tr, y_tr)
        scored = score_models(trained, X_te, y_te, task_type="regression")

        for s in scored:
            assert s.is_success
            for key in REGRESSION_METRICS:
                assert key in s.metrics
            assert s.metrics["rmse"] >= 0
            assert s.metrics["mae"] >= 0


class TestSelectBest:
    def test_pick_max_for_f1(self) -> None:
        scored = [
            ScoredModel("a", "success", {"f1": 0.70}),
            ScoredModel("b", "success", {"f1": 0.85}),
            ScoredModel("c", "success", {"f1": 0.80}),
        ]
        best = select_best(scored, "f1")
        assert best is not None
        assert best.algo_name == "b"

    def test_pick_min_for_rmse(self) -> None:
        scored = [
            ScoredModel("a", "success", {"rmse": 1.2}),
            ScoredModel("b", "success", {"rmse": 0.8}),
            ScoredModel("c", "success", {"rmse": 2.0}),
        ]
        best = select_best(scored, "rmse")
        assert best is not None
        assert best.algo_name == "b"

    def test_skip_failed_and_missing_metric(self) -> None:
        scored = [
            ScoredModel("a", "failed", {}, error="x"),
            ScoredModel("b", "success", {"accuracy": 0.6}),  # no f1
            ScoredModel("c", "success", {"f1": 0.7}),
        ]
        best = select_best(scored, "f1")
        assert best is not None
        assert best.algo_name == "c"

    def test_all_failed_returns_none(self) -> None:
        scored = [ScoredModel("a", "failed", {}, error="x")]
        assert select_best(scored, "f1") is None

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            select_best([ScoredModel("a", "success", {"foo": 1.0})], "foo")


class TestMetricCatalog:
    def test_classification_metric_names(self) -> None:
        assert set(CLASSIFICATION_METRICS) == {"accuracy", "f1", "roc_auc"}

    def test_regression_metric_names(self) -> None:
        assert set(REGRESSION_METRICS) == {"rmse", "mae", "r2"}


class TestVisualizationData:
    def test_confusion_matrix_data_default_labels(self) -> None:
        out = confusion_matrix_data([0, 1, 1, 0], [0, 1, 0, 0])
        assert out["labels"] == ["0", "1"]
        assert out["matrix"] == [[2, 0], [1, 1]]

    def test_confusion_matrix_custom_labels(self) -> None:
        out = confusion_matrix_data(["a", "b"], ["a", "a"], labels=["a", "b", "c"])
        assert out["labels"] == ["a", "b", "c"]
        assert len(out["matrix"]) == 3

    def test_regression_scatter_data(self) -> None:
        out = regression_scatter_data([1.0, 2.0, 3.0], np.array([1.1, 1.9, 3.05]))
        assert out["y_true"] == [1.0, 2.0, 3.0]
        assert out["y_pred"] == pytest.approx([1.1, 1.9, 3.05])


class TestBuildPlotData:
    def test_classification_returns_confusion_matrix_per_algo(
        self,
        classification_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_tr, X_te, y_tr, y_te = classification_split
        pre = build_preprocessor(["age", "income"], ["city"])
        trained = train_all(get_specs("classification"), pre, X_tr, y_tr)
        plots = build_plot_data(trained, X_te, y_te, task_type="classification")

        success_names = {t.algo_name for t in trained if t.is_success}
        assert set(plots.keys()) == success_names
        for data in plots.values():
            assert data["kind"] == "confusion_matrix"
            assert isinstance(data["labels"], list) and len(data["labels"]) >= 2
            assert isinstance(data["matrix"], list)
            # 정사각 행렬
            assert all(len(row) == len(data["labels"]) for row in data["matrix"])

    def test_regression_returns_scatter_per_algo(
        self,
        regression_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ) -> None:
        X_tr, X_te, y_tr, y_te = regression_split
        pre = build_preprocessor(["x1", "x2"], ["cat"])
        trained = train_all(get_specs("regression"), pre, X_tr, y_tr)
        plots = build_plot_data(trained, X_te, y_te, task_type="regression")

        assert plots  # 최소 1개 이상
        for data in plots.values():
            assert data["kind"] == "regression_scatter"
            assert len(data["y_true"]) == len(y_te)
            assert len(data["y_pred"]) == len(y_te)

    def test_skips_failed_trained_models(self) -> None:
        trained = [
            TrainedModel(
                algo_name="bad",
                estimator=None,
                status="failed",
                train_time_ms=0,
                error="x",
            )
        ]
        plots = build_plot_data(
            trained, pd.DataFrame(), pd.Series(dtype=int), task_type="classification"
        )
        assert plots == {}

    def test_rejects_unknown_task_type(self) -> None:
        with pytest.raises(ValueError):
            build_plot_data([], pd.DataFrame(), pd.Series(dtype=int), task_type="bogus")
