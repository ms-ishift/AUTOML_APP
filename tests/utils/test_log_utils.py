"""`utils.log_utils` 로거 초기화·롤링·포매터 회귀 테스트.

검증 항목:
- ``get_logger`` 가 ``automl`` 하위 로거를 반환하고 핸들러 중복 추가 없음
- ``_KVFormatter`` 가 ``extra=`` 필드를 ``| k=v`` 로 덧붙임
- ``RotatingFileHandler`` 가 5MB/3 rotation 설정을 따르며 실제로 백업 파일을 만든다
"""

from __future__ import annotations

import importlib
import logging
import logging.handlers
from pathlib import Path

import pytest

import utils.log_utils as log_utils


@pytest.fixture
def reset_log_utils(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """초기화 플래그/핸들러를 리셋해 격리된 로그 디렉터리로 재초기화."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    import config.settings as cfg

    importlib.reload(cfg)
    importlib.reload(log_utils)

    root = logging.getLogger("automl")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    log_utils._INITIALIZED = False

    yield tmp_path

    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    log_utils._INITIALIZED = False


def test_get_logger_idempotent_and_namespaced(reset_log_utils: Path) -> None:
    a = log_utils.get_logger("service.foo")
    b = log_utils.get_logger("service.foo")
    root = logging.getLogger("automl")

    assert a is b
    assert a.name == "automl.service.foo"
    assert len(root.handlers) == 2  # file + console

    log_utils.get_logger("service.bar")
    assert len(root.handlers) == 2


def test_log_event_format_includes_extra_kv(reset_log_utils: Path) -> None:
    """``log_event`` 는 ``| k=v`` 접미사 포맷으로 파일에 기록돼야 한다.

    ``automl`` 루트는 ``propagate=False`` 라 ``caplog`` 로는 잡히지 않는다 —
    실제 RotatingFileHandler 출력물을 읽어 검증한다.
    """
    logger = log_utils.get_logger("service.audit")
    log_utils.log_event(logger, "dataset.uploaded", project_id=7, size_bytes=1024)

    root = logging.getLogger("automl")
    for h in root.handlers:
        h.flush()

    log_path = reset_log_utils / "logs" / "app.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "dataset.uploaded" in content
    assert "project_id=7" in content
    assert "size_bytes=1024" in content
    assert "automl.service.audit" in content


def test_rotating_file_handler_is_configured(reset_log_utils: Path) -> None:
    log_utils.get_logger("init")
    root = logging.getLogger("automl")

    rotating = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(rotating) == 1
    rh = rotating[0]

    assert rh.maxBytes == 5 * 1024 * 1024
    assert rh.backupCount == 3
    assert Path(rh.baseFilename).name == "app.log"
    assert Path(rh.baseFilename).parent == reset_log_utils / "logs"


def test_rotating_file_handler_actually_rotates(reset_log_utils: Path) -> None:
    """작은 maxBytes 로 강제 롤링을 유도해 백업 파일이 생성되는지 확인."""
    log_utils.get_logger("bootstrap")
    root = logging.getLogger("automl")
    file_handlers = [
        h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    root.removeHandler(file_handlers[0])
    file_handlers[0].close()

    log_dir = reset_log_utils / "logs"
    log_path = log_dir / "app.log"
    small_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=256, backupCount=2, encoding="utf-8"
    )
    small_handler.setFormatter(log_utils._KVFormatter(log_utils._FORMAT))
    root.addHandler(small_handler)

    logger = log_utils.get_logger("rotate.bench")
    payload = "x" * 120
    for i in range(20):
        log_utils.log_event(logger, f"event.{i}", detail=payload)

    small_handler.flush()
    small_handler.close()

    assert log_path.exists()
    rotated = sorted(log_dir.glob("app.log.*"))
    assert rotated, "backup 파일이 하나도 생성되지 않음 — 롤링 동작 실패"
    assert len(rotated) <= 2, f"backupCount=2 초과 로테이션: {rotated}"
