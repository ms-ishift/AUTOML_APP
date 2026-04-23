"""ml.balancing 단위 테스트 (IMPLEMENTATION_PLAN §9.5, FR-057)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ml import balancing
from ml.balancing import apply_imbalance_strategy
from ml.schemas import PreprocessingConfig
from utils.errors import MLTrainingError

# ----------------------------------------------------------------------- helpers


def _imbalanced_classification_data(
    n_majority: int = 200, n_minority: int = 20, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_major = rng.normal(0.0, 1.0, size=(n_majority, 3))
    X_minor = rng.normal(3.0, 1.0, size=(n_minority, 3))
    X = np.vstack([X_major, X_minor])
    y = np.concatenate([np.zeros(n_majority, dtype=int), np.ones(n_minority, dtype=int)])
    return pd.DataFrame(X, columns=["a", "b", "c"]), pd.Series(y, name="y")


# ----------------------------------------------------------------------- none


class TestNoneStrategy:
    def test_returns_inputs_unchanged(self) -> None:
        X, y = _imbalanced_classification_data()
        est = LogisticRegression()
        cfg = PreprocessingConfig(imbalance="none")
        out_est, out_X, out_y = apply_imbalance_strategy(est, X, y, cfg)
        assert out_est is est
        assert out_X is X
        assert out_y is y


# ----------------------------------------------------------------------- class_weight


class TestClassWeightStrategy:
    def test_sets_balanced_on_supported_estimator(self) -> None:
        X, y = _imbalanced_classification_data()
        est = LogisticRegression()
        cfg = PreprocessingConfig(imbalance="class_weight")
        out_est, out_X, out_y = apply_imbalance_strategy(est, X, y, cfg)
        assert out_est.get_params()["class_weight"] == "balanced"
        assert out_X is X
        assert out_y is y

    def test_unsupported_estimator_logs_warning_and_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # automl 로거는 propagate=False 이므로 logger.warning 을 직접 가로채 검증.
        messages: list[str] = []

        def _capture(msg: str, *_a: object, **_kw: object) -> None:
            messages.append(msg)

        monkeypatch.setattr(balancing.logger, "warning", _capture)

        X, y = _imbalanced_classification_data()
        est = KNeighborsClassifier()  # class_weight 미지원
        cfg = PreprocessingConfig(imbalance="class_weight")
        out_est, _, _ = apply_imbalance_strategy(est, X, y, cfg)
        assert out_est is est
        # class_weight 파라미터 자체가 존재하지 않아야 passthrough 경로가 검증됨
        assert "class_weight" not in est.get_params()
        assert any("class_weight 를 지원하지 않는" in m for m in messages)


# ----------------------------------------------------------------------- smote


class TestSmoteStrategy:
    def test_rebalances_minority_class(self) -> None:
        X, y = _imbalanced_classification_data(n_majority=200, n_minority=20)
        est = LogisticRegression()
        cfg = PreprocessingConfig(imbalance="smote", smote_k_neighbors=3)
        out_est, out_X, out_y = apply_imbalance_strategy(est, X, y, cfg, task_type="classification")
        # SMOTE 기본 sampling_strategy='auto' → 소수 클래스가 다수와 같은 수로 증강
        counts = pd.Series(out_y).value_counts()
        assert counts[0] == counts[1]
        assert len(out_X) == len(out_y)
        assert len(out_X) > len(X)
        # estimator 는 같은 인스턴스여야 (fit 은 호출자 책임)
        assert out_est is est

    def test_k_neighbors_forwarded_to_smote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        X, y = _imbalanced_classification_data()
        captured: dict[str, object] = {}

        class _FakeSMOTE:
            def __init__(self, *, k_neighbors: int, random_state: int) -> None:
                captured["k_neighbors"] = k_neighbors
                captured["random_state"] = random_state

            def fit_resample(self, X_: object, y_: object) -> tuple[object, object]:
                return X_, y_

        monkeypatch.setattr(balancing, "_SMOTE", _FakeSMOTE)
        monkeypatch.setattr(balancing, "SMOTE_AVAILABLE", True)
        cfg = PreprocessingConfig(imbalance="smote", smote_k_neighbors=7)
        apply_imbalance_strategy(LogisticRegression(), X, y, cfg)
        assert captured["k_neighbors"] == 7
        # RANDOM_SEED 는 settings 에 존재 (기본 42)
        from config.settings import settings

        assert captured["random_state"] == settings.RANDOM_SEED

    def test_regression_raises_ml_training_error(self) -> None:
        X, y_cls = _imbalanced_classification_data()
        # 회귀인 것처럼 task_type 만 바꿔 호출 (방어 경로)
        cfg = PreprocessingConfig(imbalance="smote")
        with pytest.raises(MLTrainingError, match="회귀"):
            apply_imbalance_strategy(LogisticRegression(), X, y_cls, cfg, task_type="regression")

    def test_smote_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(balancing, "SMOTE_AVAILABLE", False)
        monkeypatch.setattr(balancing, "_SMOTE", None)
        X, y = _imbalanced_classification_data()
        cfg = PreprocessingConfig(imbalance="smote")
        with pytest.raises(MLTrainingError, match="imbalanced-learn"):
            apply_imbalance_strategy(LogisticRegression(), X, y, cfg)


# ----------------------------------------------------------------------- unknown


class TestUnknownStrategy:
    def test_invalid_strategy_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # PreprocessingConfig 는 Literal 로 검증되므로 우회를 위해 frozen 해제 후 set
        cfg = PreprocessingConfig(imbalance="none")
        object.__setattr__(cfg, "imbalance", "bogus")
        with pytest.raises(ValueError, match="알 수 없는 imbalance"):
            apply_imbalance_strategy(
                LogisticRegression(),
                *_imbalanced_classification_data(),
                cfg,
            )
