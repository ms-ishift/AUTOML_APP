"""로깅 초기화 및 헬퍼 (IMPLEMENTATION_PLAN §1.3, NFR-007).

설계:
- 루트 로거 이름은 ``automl`` 이며 모든 하위 로거는 이를 상속.
- 핸들러: 콘솔 + RotatingFileHandler(``<storage>/logs/app.log``).
- 포매터: ``<timestamp> | <level> | <name> | <msg> | k=v ...``
  `logger.info(event, extra={"k": "v"})` 형태의 구조화 로그 지원.
- 초기화는 idempotent, 멀티 페이지 재진입에도 핸들러가 중복 추가되지 않는다.
"""

from __future__ import annotations

import logging
import logging.handlers
import threading
import warnings
from typing import Any, Final

from config.settings import settings


def _install_noisy_warning_filters() -> None:
    """예측 결과에 영향이 없는 반복 경고를 전역 필터로 억제한다.

    억제 대상:
    - ``X does not have valid feature names, but ... was fitted with feature names``
      (sklearn ``_check_feature_names``): ``ColumnTransformer`` 가 numpy 배열을 반환해
      추정기 predict 시 이름이 사라지는 구조적 경고로, 학습/예측 결과에는 영향 없음.

    필터는 전역 ``warnings`` 모듈에 등록되므로 멀티페이지/재진입 환경에서도 idempotent
    하다 (같은 메시지 패턴이 중복 등록돼도 동작은 동일).
    """
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but .* was fitted with feature names",
        category=UserWarning,
    )


_LOCK: Final[threading.Lock] = threading.Lock()
_INITIALIZED: bool = False

_ROOT_NAME: Final[str] = "automl"
_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class _KVFormatter(logging.Formatter):
    """``extra=`` 로 전달된 사용자 필드를 ``| k=v`` 로 덧붙인다."""

    _RESERVED: Final[set[str]] = set(
        logging.LogRecord("x", logging.INFO, "", 0, "", None, None).__dict__.keys()
    ) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in self._RESERVED}
        if not extras:
            return base
        extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
        return f"{base} | {extra_str}"


def _initialize() -> None:
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return

        _install_noisy_warning_filters()
        settings.ensure_dirs()
        log_path = settings.logs_dir / "app.log"

        root = logging.getLogger(_ROOT_NAME)
        root.setLevel(settings.LOG_LEVEL)
        root.propagate = False

        if not root.handlers:
            fmt = _KVFormatter(_FORMAT)

            file_h = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_h.setFormatter(fmt)
            root.addHandler(file_h)

            console_h = logging.StreamHandler()
            console_h.setFormatter(fmt)
            root.addHandler(console_h)

        _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """``automl`` 루트 아래 자식 로거를 반환한다."""
    _initialize()
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **extra: Any,
) -> None:
    """구조화 로그: 이벤트명(``utils.events.Event``) + extra 키=값.

    사용 예::

        log_event(logger, Event.DATASET_UPLOADED, project_id=1, bytes=1024)
    """
    logger.log(level, event, extra=extra)
