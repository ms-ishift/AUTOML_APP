"""모델 아티팩트 직렬화 (IMPLEMENTATION_PLAN §3.7, FR-073, 요구사항 §10.4).

레이아웃 (``storage/models/<model_id>/``)::

    model.joblib          # 전체 파이프라인 (preprocessor + estimator)
    preprocessor.joblib   # fit 된 ColumnTransformer (파이프라인에서 추출)
    feature_schema.json   # 예측 입력 폼 + 입력 검증의 단일 출처 (FR-082, FR-083)
    metrics.json          # 성능/메타 정보 스냅샷

원칙 (``.cursor/rules/ml-engine.mdc``):
- 이 모듈은 **Streamlit/DB 비의존**. 업로드/조회는 상위 Service 가 오케스트레이션.
- ``FeatureSchema`` 는 예측 시 입력 검증의 기준. ``validate_prediction_input`` 은
  누락 컬럼을 차단하고 추가 컬럼은 조용히 제거한다 (§10.4).
- 예외는 pure ``ValueError`` / ``FileNotFoundError`` 로 발생. ``utils.errors`` 로의
  변환은 Service 레이어 책임 (ml → utils 결합을 피해 레이어 경계를 지킨다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from ml.schemas import FeatureSchema

if TYPE_CHECKING:
    import pandas as pd


MODEL_FILENAME = "model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
SCHEMA_FILENAME = "feature_schema.json"
METRICS_FILENAME = "metrics.json"

_REQUIRED_FILES: tuple[str, ...] = (
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    SCHEMA_FILENAME,
    METRICS_FILENAME,
)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """예측에 필요한 4요소를 묶은 번들.

    ``estimator`` 는 일반적으로 ``sklearn.pipeline.Pipeline`` (전처리+추정기)이며,
    ``preprocessor`` 는 같은 파이프라인에서 추출된 fit 된 transformer 의 독립 복제본.
    """

    estimator: Any
    preprocessor: Any
    schema: FeatureSchema
    metrics: dict[str, Any]


# ---------------------------------------------------------------- save/load


def save_model_bundle(
    target_dir: Path | str,
    *,
    estimator: Any,
    preprocessor: Any,
    schema: FeatureSchema,
    metrics: dict[str, Any],
) -> dict[str, Path]:
    """4개 파일을 생성하고 각 경로를 dict 로 반환.

    - 디렉터리는 idempotent 하게 생성된다.
    - 파일 쓰기 중간 실패가 발생하면 호출자가 디렉터리 단위로 정리(``shutil.rmtree``)해야 한다.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": target_dir / MODEL_FILENAME,
        "preprocessor": target_dir / PREPROCESSOR_FILENAME,
        "schema": target_dir / SCHEMA_FILENAME,
        "metrics": target_dir / METRICS_FILENAME,
    }

    joblib.dump(estimator, paths["model"])
    joblib.dump(preprocessor, paths["preprocessor"])
    paths["schema"].write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return paths


def load_model_bundle(source_dir: Path | str) -> ModelBundle:
    """4개 파일을 읽어 ``ModelBundle`` 로 복원. 어느 하나라도 누락이면 ``FileNotFoundError``."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"모델 디렉터리가 없습니다: {source_dir}")

    for fn in _REQUIRED_FILES:
        if not (source_dir / fn).exists():
            raise FileNotFoundError(f"필수 아티팩트 누락: {fn} @ {source_dir}")

    estimator = joblib.load(source_dir / MODEL_FILENAME)
    preprocessor = joblib.load(source_dir / PREPROCESSOR_FILENAME)
    schema_raw = json.loads((source_dir / SCHEMA_FILENAME).read_text(encoding="utf-8"))
    schema = FeatureSchema.from_dict(schema_raw)
    metrics = json.loads((source_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    return ModelBundle(
        estimator=estimator,
        preprocessor=preprocessor,
        schema=schema,
        metrics=metrics,
    )


# ------------------------------------------------------------- input check


def validate_prediction_input(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """예측 입력 검증 (FR-083, §10.4).

    규칙:
    - 필수 컬럼(``schema.input_columns``) 누락 → ``ValueError`` (Service 가
      ``PredictionInputError`` 로 변환).
    - 추가 컬럼 → 조용히 제거.
    - 수치형 컬럼은 ``pd.to_numeric(..., errors='coerce')`` 로 강제 변환
      (비숫자 값은 NaN 이 되어 전처리기의 imputer 가 흡수한다).
    - 범주형 컬럼은 ``astype(str)`` 로 정규화 (``OneHotEncoder(handle_unknown='ignore')`` 가
      unseen 카테고리를 안전 처리).
    """
    import pandas as pd

    if df is None:
        raise ValueError("예측 입력 데이터가 비어 있습니다.")
    if df.empty:
        raise ValueError("예측 입력 데이터가 비어 있습니다.")

    expected = list(schema.input_columns)
    if not expected:
        raise ValueError("학습된 피처 스키마가 비어 있습니다. 모델이 손상되었을 수 있습니다.")

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"필수 입력 컬럼 누락: {', '.join(missing)}")

    cleaned = df.loc[:, expected].copy()
    for col in schema.numeric:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in schema.categorical:
        cleaned[col] = cleaned[col].astype(str)
    return cleaned


__all__ = [
    "MODEL_FILENAME",
    "PREPROCESSOR_FILENAME",
    "SCHEMA_FILENAME",
    "METRICS_FILENAME",
    "ModelBundle",
    "save_model_bundle",
    "load_model_bundle",
    "validate_prediction_input",
]
