"""클래스 불균형 대응 (IMPLEMENTATION_PLAN §9.5, FR-057).

원칙:
- 이 모듈은 Streamlit/DB 에 의존하지 않는다. ``imblearn`` 은 선택적 의존(없으면
  ``SMOTE_AVAILABLE=False``) — ``smote`` 요청 시에만 예외로 승격한다.
- ``apply_imbalance_strategy`` 는 학습 데이터에만 호출되어야 한다 (호출자 측
  train/test split 이후에만 통과시킨다 — 테스트 세트는 절대 리샘플링 금지).
- 전략별 의미:
    - ``"none"`` → passthrough (변경 없이 반환)
    - ``"class_weight"`` → ``estimator.set_params(class_weight="balanced")``.
      estimator 가 ``class_weight`` 를 지원하지 않으면 warning 로그 후 passthrough.
    - ``"smote"`` → ``imblearn.over_sampling.SMOTE(k_neighbors=..., random_state=...)``
      로 X_train/y_train 을 리샘플링. 회귀 작업 또는 imblearn 미설치 시
      ``MLTrainingError`` 로 즉시 실패 (상위에서 사용자 메시지로 변환).

후속(§9.6) 에서 ``ml/trainers.py`` 가 이 함수를 호출하도록 통합된다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from config.settings import settings
from utils.errors import MLTrainingError
from utils.log_utils import get_logger

if TYPE_CHECKING:
    from ml.schemas import PreprocessingConfig

logger = get_logger(__name__)


try:
    from imblearn.over_sampling import SMOTE as _SMOTE

    SMOTE_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - 의존 부재 경로, 테스트는 monkeypatch 로 커버
    _SMOTE = None
    SMOTE_AVAILABLE = False


__all__ = ["SMOTE_AVAILABLE", "apply_imbalance_strategy"]


def apply_imbalance_strategy(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    config: PreprocessingConfig,
    *,
    task_type: str = "classification",
) -> tuple[Any, Any, Any]:
    """불균형 전략을 estimator / 훈련 데이터에 적용한다.

    Returns:
        ``(estimator, X_train, y_train)`` 3-tuple — 전략에 따라 estimator 의 파라미터
        또는 (X_train, y_train) 이 교체된다. 비변경 경로는 입력 그대로.

    Raises:
        MLTrainingError:
            - ``imbalance="smote"`` 인데 ``task_type == "regression"``
            - ``imbalance="smote"`` 인데 ``imblearn`` 미설치 (``SMOTE_AVAILABLE=False``)
        ValueError: 알 수 없는 전략.
    """
    strategy = config.imbalance

    if strategy == "none":
        return estimator, X_train, y_train

    if strategy == "class_weight":
        return _apply_class_weight(estimator), X_train, y_train

    if strategy == "smote":
        return _apply_smote(estimator, X_train, y_train, config, task_type=task_type)

    raise ValueError(f"알 수 없는 imbalance 전략: {strategy}")


def _apply_class_weight(estimator: Any) -> Any:
    """``class_weight="balanced"`` 를 설정. 미지원 estimator 는 warning 후 passthrough."""
    try:
        estimator.set_params(class_weight="balanced")
    except (ValueError, TypeError):
        # sklearn 의 set_params 는 미지원 파라미터에 ValueError 를 던진다.
        logger.warning(
            "class_weight 를 지원하지 않는 estimator 입니다. 적용을 건너뜁니다.",
            extra={"estimator": type(estimator).__name__},
        )
    return estimator


def _apply_smote(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    config: PreprocessingConfig,
    *,
    task_type: str,
) -> tuple[Any, Any, Any]:
    """SMOTE 리샘플링. 회귀/미설치 가드 포함."""
    if task_type == "regression":
        # TrainingConfig.__post_init__ 에서 이미 차단되지만 defense-in-depth.
        raise MLTrainingError("회귀(regression) 작업에는 SMOTE 를 적용할 수 없습니다.")

    if not SMOTE_AVAILABLE or _SMOTE is None:
        raise MLTrainingError(
            "SMOTE 를 사용하려면 imbalanced-learn 패키지가 필요합니다. "
            "requirements.txt 를 설치하거나 imbalance 전략을 변경하세요."
        )

    sampler = _SMOTE(
        k_neighbors=config.smote_k_neighbors,
        random_state=settings.RANDOM_SEED,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE 리샘플링 완료",
        extra={
            "n_before": _n_rows(X_train),
            "n_after": _n_rows(X_resampled),
            "k_neighbors": config.smote_k_neighbors,
        },
    )
    return estimator, X_resampled, y_resampled


def _n_rows(X: Any) -> int:
    """로깅용 안전 행 수 추정."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return int(X.shape[0])
    arr = np.asarray(X)
    return int(arr.shape[0]) if arr.ndim >= 1 else 0
