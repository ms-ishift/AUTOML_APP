"""Service 테스트 공통 fixtures (IMPLEMENTATION_PLAN §4.6).

- 최상위 ``tests/conftest.py`` 의 ``sqlite_engine`` 을 재사용한다.
- ``session_scope()`` 가 내부적으로 ``repositories.base.SessionLocal`` 을 쓰므로,
  테스트 격리를 위해 전역 ``engine`` / ``SessionLocal`` 을 임시 sqlite 로 오버라이드한다.
- Service 는 Streamlit 없이 import/실행 가능해야 한다는 규약 (service-layer.mdc)을 지킴.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.orm import sessionmaker

from repositories import base as repo_base
from repositories.models import (
    SYSTEM_USER_ID,
    SYSTEM_USER_LOGIN_ID,
    SYSTEM_USER_NAME,
    SYSTEM_USER_ROLE,
    User,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session


@pytest.fixture(autouse=True)
def _override_session_local(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_engine: Engine,
) -> Iterator[None]:
    """``session_scope()`` 가 테스트 전용 sqlite 엔진을 사용하도록 주입."""
    test_factory: sessionmaker[Session] = sessionmaker(
        bind=sqlite_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    monkeypatch.setattr(repo_base, "engine", sqlite_engine)
    monkeypatch.setattr(repo_base, "SessionLocal", test_factory)
    yield


@pytest.fixture()
def seeded_system_user(sqlite_engine: Engine) -> User:
    """시스템 사용자 시드 (AuditLog FK 안정성 확보)."""
    factory: sessionmaker[Session] = sessionmaker(
        bind=sqlite_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    with factory() as session:
        existing = session.get(User, SYSTEM_USER_ID)
        if existing is not None:
            return existing
        user = User(
            user_id=SYSTEM_USER_ID,
            login_id=SYSTEM_USER_LOGIN_ID,
            user_name=SYSTEM_USER_NAME,
            role=SYSTEM_USER_ROLE,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


@pytest.fixture()
def tmp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Service 가 파일을 생성할 경우를 대비한 임시 STORAGE_DIR.

    현재 ProjectService 는 파일을 만들지 않지만, 후속 Dataset/Model/Prediction
    Service 테스트에서 재사용될 수 있도록 공통 fixture 로 마련한다.
    """
    from config.settings import settings

    storage = tmp_path / "storage"
    (storage / "datasets").mkdir(parents=True, exist_ok=True)
    (storage / "models").mkdir(parents=True, exist_ok=True)
    (storage / "predictions").mkdir(parents=True, exist_ok=True)
    (storage / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "STORAGE_DIR", storage)
    return storage
