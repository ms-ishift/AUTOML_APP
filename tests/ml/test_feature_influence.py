"""``ml/feature_influence`` 단위 테스트 (IMPLEMENTATION_PLAN §11.1~11.2, FR-094~095)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.feature_influence import (
    compute_permutation_importance,
    extract_builtin_transformed_importances,
    subsample_xy,
)


def _tiny_classification_xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 80
    x_num = rng.normal(size=n)
    x_cat = rng.choice(["a", "b"], size=n)
    y = (x_num + (x_cat == "b") * 0.5 + rng.normal(0, 0.1, size=n) > 0).astype(int)
    X = pd.DataFrame({"num": x_num, "cat": x_cat})
    return X, pd.Series(y, name="target")


def test_subsample_xy_truncates() -> None:
    X, y = _tiny_classification_xy()
    X2, y2, n0 = subsample_xy(X, y, max_rows=10, random_state=1)
    assert n0 == len(X)
    assert len(X2) == 10
    assert len(y2) == 10


def test_compute_permutation_importance_smoke() -> None:
    X, y = _tiny_classification_xy()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer()), ("sc", StandardScaler())]), ["num"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cat"]),
        ],
    )
    pipe = Pipeline(
        [("preprocessor", pre), ("model", DecisionTreeClassifier(max_depth=4, random_state=0))]
    )
    pipe.fit(X, y)
    rows, n_used, n_tot = compute_permutation_importance(
        pipe,
        X,
        y,
        task_type="classification",
        metric_key="accuracy",
        n_repeats=5,
        random_state=0,
        n_jobs=1,
        max_rows=5000,
    )
    assert n_used == len(X) == n_tot
    names = {r[0] for r in rows}
    assert names == {"num", "cat"}
    assert all(r[1] >= 0 for r in rows)


def test_extract_builtin_returns_none_for_logistic_pipeline() -> None:
    X, y = _tiny_classification_xy()
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(), ["num"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cat"]),
        ],
    )
    pipe = Pipeline(
        [("preprocessor", pre), ("model", LogisticRegression(max_iter=200, random_state=0))]
    )
    pipe.fit(X, y)
    assert extract_builtin_transformed_importances(pipe) is None


def test_extract_builtin_tree_pipeline() -> None:
    X, y = _tiny_classification_xy()
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(), ["num"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cat"]),
        ],
    )
    pipe = Pipeline(
        [("preprocessor", pre), ("model", DecisionTreeClassifier(max_depth=4, random_state=0))]
    )
    pipe.fit(X, y)
    built = extract_builtin_transformed_importances(pipe)
    assert built is not None
    assert len(built) >= 2
    assert all(v >= 0.0 for _, v in built)
