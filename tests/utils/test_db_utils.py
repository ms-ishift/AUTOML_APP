"""``utils/db_utils`` 단위 테스트 (IMPLEMENTATION_PLAN §5.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine

from repositories import base as repo_base
from repositories.models import Base
from utils.db_utils import REQUIRED_TABLES, is_db_initialized

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine


def _install_engine(monkeypatch: pytest.MonkeyPatch, engine: Engine) -> None:
    monkeypatch.setattr(repo_base, "engine", engine)


def test_is_db_initialized_true_when_all_tables_present(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_engine(monkeypatch, sqlite_engine)
    assert is_db_initialized() is True


def test_is_db_initialized_false_on_empty_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = create_engine(f"sqlite:///{tmp_path}/empty.db", future=True)
    try:
        _install_engine(monkeypatch, empty)
        assert is_db_initialized() is False
    finally:
        empty.dispose()


def test_is_db_initialized_false_on_partial_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = create_engine(f"sqlite:///{tmp_path}/partial.db", future=True)
    try:
        # 전체가 아니라 User 테이블만 만들어 부분 초기화 상태 시뮬레이션
        Base.metadata.tables["users"].create(partial)
        _install_engine(monkeypatch, partial)
        assert is_db_initialized() is False
    finally:
        partial.dispose()


def test_required_tables_is_nonempty() -> None:
    # 호환성 회귀 방지: 7개 핵심 테이블이 유지되는지
    assert len(REQUIRED_TABLES) == 7
    assert "projects" in REQUIRED_TABLES
    assert "prediction_jobs" in REQUIRED_TABLES


def test_is_db_initialized_false_on_broken_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoomEngine:
        def connect(self) -> None:  # pragma: no cover - inspect 가 먼저 터짐
            raise RuntimeError("boom")

        url = "sqlite:///broken"

    monkeypatch.setattr(repo_base, "engine", BoomEngine())
    # sqlalchemy.inspect(BoomEngine()) 가 예외 → False 로 수렴
    assert is_db_initialized() is False
