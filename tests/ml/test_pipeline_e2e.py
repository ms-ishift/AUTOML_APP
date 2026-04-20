"""ML 파이프라인 end-to-end smoke (IMPLEMENTATION_PLAN §3.4~§3.6 통합).

샘플 CSV 를 읽어 전처리 → 학습 → 평가 → best 선정 까지 한 흐름을 검증한다.
DB/Streamlit 계층을 타지 않고 순수 ml/ 만 사용.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.evaluators import score_models, select_best
from ml.preprocess import (
    build_feature_schema,
    build_preprocessor,
    prepare_xy,
    split_feature_types,
)
from ml.registry import get_specs
from ml.schemas import TrainingConfig
from ml.trainers import split_dataset, train_all


def _run_pipeline(df: pd.DataFrame, cfg: TrainingConfig, metric_key: str) -> None:
    X, y = prepare_xy(df, cfg)
    num_cols, cat_cols = split_feature_types(
        df, target=cfg.target_column, excluded=cfg.excluded_columns
    )

    schema = build_feature_schema(df, num_cols, cat_cols, cfg.target_column)
    assert schema.target == cfg.target_column
    assert set(schema.input_columns).issubset(set(X.columns))

    pre = build_preprocessor(num_cols, cat_cols)

    X_tr, X_te, y_tr, y_te = split_dataset(X, y, test_size=cfg.test_size, task_type=cfg.task_type)

    specs = get_specs(cfg.task_type)
    trained = train_all(specs, pre, X_tr, y_tr)
    assert any(t.is_success for t in trained)

    scored = score_models(trained, X_te, y_te, task_type=cfg.task_type)
    assert any(s.is_success for s in scored)

    best = select_best(scored, metric_key)
    assert best is not None
    assert best.is_success
    assert metric_key in best.metrics


def test_classification_e2e(classification_csv: Path) -> None:
    df = pd.read_csv(classification_csv)
    cfg = TrainingConfig(
        dataset_id=1,
        task_type="classification",
        target_column="species",
        test_size=0.25,
    )
    _run_pipeline(df, cfg, metric_key="f1")


def test_regression_e2e(regression_csv: Path) -> None:
    df = pd.read_csv(regression_csv)
    cfg = TrainingConfig(
        dataset_id=2,
        task_type="regression",
        target_column="progression",
        test_size=0.25,
    )
    _run_pipeline(df, cfg, metric_key="rmse")
