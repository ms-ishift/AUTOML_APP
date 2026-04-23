"""ml.trainers 단위 테스트 (IMPLEMENTATION_PLAN §3.5 / §9.6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.preprocess import build_preprocessor
from ml.registry import AlgoSpec, get_specs
from ml.schemas import PreprocessingConfig
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


class TestTrainAllBalancer:
    """§9.6: train_all 의 balancer/preprocess_cfg 키워드 인자 확장 검증."""

    def test_preprocess_cfg_passthrough_does_not_alter_behavior(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """preprocess_cfg 만 전달해도 balancer 없으면 기존 경로와 동일하게 학습 성공."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:1]
        cfg = PreprocessingConfig()

        results = train_all(specs, pre, X, y, preprocess_cfg=cfg)

        assert len(results) == 1
        assert results[0].is_success, results[0].error

    def test_balancer_invoked_per_spec(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """balancer 는 각 spec 의 fresh estimator 에 대해 정확히 한 번씩 호출되어야 한다."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:2]

        calls: list[tuple[str, int]] = []

        def _balancer(
            estimator: Any, X_in: pd.DataFrame, y_in: pd.Series
        ) -> tuple[Any, pd.DataFrame, pd.Series]:
            calls.append((type(estimator).__name__, len(X_in)))
            return estimator, X_in, y_in

        results = train_all(specs, pre, X, y, balancer=_balancer)

        assert len(calls) == len(specs)
        assert all(n == len(X) for _, n in calls)
        for r in results:
            assert r.is_success, r.error

    def test_balancer_may_resample_training_data(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """balancer 가 X_train/y_train 을 치환(리샘플)한 경우 치환된 데이터로 fit 되어야 한다."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:1]

        # 소수 클래스 업샘플링을 모사: 단순히 첫 N 행을 두 배로 복제
        X_resampled = pd.concat([X, X.head(10)], ignore_index=True)
        y_resampled = pd.concat([y, y.head(10)], ignore_index=True)

        def _balancer(
            estimator: Any, _X: pd.DataFrame, _y: pd.Series
        ) -> tuple[Any, pd.DataFrame, pd.Series]:
            return estimator, X_resampled, y_resampled

        results = train_all(specs, pre, X, y, balancer=_balancer)

        assert results[0].is_success
        # 원본 X/y 는 불변
        assert len(X) == len(y) == len(classification_data[0])

    def test_balancer_failure_is_isolated_per_spec(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """balancer 호출 자체가 실패해도 해당 spec 만 failed 로 기록되고 다음 spec 은 진행."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:2]

        seen = {"count": 0}

        def _balancer(
            estimator: Any, X_in: pd.DataFrame, y_in: pd.Series
        ) -> tuple[Any, pd.DataFrame, pd.Series]:
            seen["count"] += 1
            if seen["count"] == 1:
                raise RuntimeError("balancer synthetic boom")
            return estimator, X_in, y_in

        results = train_all(specs, pre, X, y, balancer=_balancer)

        assert not results[0].is_success
        assert results[0].error is not None
        assert "balancer synthetic boom" in results[0].error
        assert results[1].is_success

    def test_default_kwargs_preserve_legacy_behavior(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """preprocess_cfg/balancer 둘 다 생략하면 기존 호출 시그니처 결과와 동일해야 한다."""
        X, y = classification_data
        pre = build_preprocessor(["age", "income"], ["city"])
        specs = get_specs("classification")[:1]

        legacy = train_all(specs, pre, X, y)
        extended = train_all(specs, pre, X, y, preprocess_cfg=None, balancer=None)

        assert [r.is_success for r in legacy] == [r.is_success for r in extended]
        assert [r.algo_name for r in legacy] == [r.algo_name for r in extended]
