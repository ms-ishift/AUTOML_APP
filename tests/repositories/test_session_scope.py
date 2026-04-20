"""session_scope 와 스키마 메타데이터 기본 동작 검증."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.models import (
    SYSTEM_USER_ID,
    SYSTEM_USER_LOGIN_ID,
    SYSTEM_USER_NAME,
    SYSTEM_USER_ROLE,
    Project,
    User,
)

EXPECTED_TABLES = {
    "users",
    "projects",
    "datasets",
    "training_jobs",
    "models",
    "prediction_jobs",
    "audit_logs",
}


def test_metadata_creates_all_expected_tables(sqlite_engine: Engine) -> None:
    inspector = inspect(sqlite_engine)
    actual = set(inspector.get_table_names())
    assert EXPECTED_TABLES.issubset(actual), f"누락 테이블: {EXPECTED_TABLES - actual}"


def test_project_owner_user_id_is_nullable(sqlite_engine: Engine) -> None:
    inspector = inspect(sqlite_engine)
    cols = {c["name"]: c for c in inspector.get_columns("projects")}
    assert cols["owner_user_id"]["nullable"] is True


def _make_session_scope(session_factory: sessionmaker):
    """테스트용 session_scope (운영 세션 팩토리를 건드리지 않기 위해 로컬 버전 구성)."""

    @contextmanager
    def _scope():
        s: Session = session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _scope


def test_session_scope_commits_on_success(session_factory) -> None:
    scope = _make_session_scope(session_factory)

    with scope() as s:
        s.add(Project(project_name="p1"))

    with session_factory() as verify:
        rows = verify.query(Project).all()
    assert len(rows) == 1
    assert rows[0].project_name == "p1"


def test_session_scope_rolls_back_on_exception(session_factory) -> None:
    scope = _make_session_scope(session_factory)

    with pytest.raises(RuntimeError), scope() as s:
        s.add(Project(project_name="p_rollback"))
        s.flush()
        raise RuntimeError("boom")

    with session_factory() as verify:
        rows = verify.query(Project).all()
    assert rows == []


def test_system_user_can_be_seeded(db_session: Session) -> None:
    db_session.add(
        User(
            user_id=SYSTEM_USER_ID,
            login_id=SYSTEM_USER_LOGIN_ID,
            user_name=SYSTEM_USER_NAME,
            role=SYSTEM_USER_ROLE,
        )
    )
    db_session.commit()

    loaded = db_session.get(User, SYSTEM_USER_ID)
    assert loaded is not None
    assert loaded.user_id == 0
    assert loaded.login_id == "system"
    assert loaded.role == "system"


def test_project_can_have_null_owner(db_session: Session) -> None:
    p = Project(project_name="no-owner")
    db_session.add(p)
    db_session.commit()

    loaded = db_session.get(Project, p.project_id)
    assert loaded is not None
    assert loaded.owner_user_id is None
