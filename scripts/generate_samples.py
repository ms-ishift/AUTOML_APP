"""samples/*.csv 재현성 있게 생성.

- classification.csv : sklearn iris (타깃은 문자열 species)
- regression.csv     : sklearn diabetes (타깃은 progression)

단계 3/4 테스트와 단계 6 E2E 수용 검증의 공통 입력이다.
`make samples` 로 실행한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"


def generate_classification() -> Path:
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame.copy()
    target_map = {i: name for i, name in enumerate(iris.target_names)}
    df["species"] = df["target"].map(target_map)
    df = df.drop(columns=["target"])
    path = OUT / "classification.csv"
    df.to_csv(path, index=False)
    return path


def generate_regression() -> Path:
    data = load_diabetes(as_frame=True)
    df: pd.DataFrame = data.frame.copy()
    df = df.rename(columns={"target": "progression"})
    path = OUT / "regression.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cls_path = generate_classification()
    reg_path = generate_regression()
    print(f"[samples] generated: {cls_path.relative_to(ROOT)}")
    print(f"[samples] generated: {reg_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
