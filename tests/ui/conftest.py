"""UI 테스트 공통 fixture (Streamlit ``AppTest`` 헤드리스 기반).

- ``repositories.base.engine`` / ``SessionLocal`` 을 임시 sqlite 로 오버라이드 (Service 테스트와 동일 패턴).
- ``STORAGE_DIR`` 도 tmp_path 로 격리.
- Streamlit 의 스크립트 러너는 매 세션마다 모듈을 재로딩하므로 ``monkeypatch`` 는
  러너가 돌기 전에 적용되어야 한다 → autouse fixture 로 묶어 스크립트 내부 import 시점에 반영.
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
def _install_test_engine(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_engine: Engine,
) -> Iterator[None]:
    factory: sessionmaker[Session] = sessionmaker(
        bind=sqlite_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    monkeypatch.setattr(repo_base, "engine", sqlite_engine)
    monkeypatch.setattr(repo_base, "SessionLocal", factory)
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
    from config.settings import settings

    storage = tmp_path / "storage"
    (storage / "datasets").mkdir(parents=True, exist_ok=True)
    (storage / "models").mkdir(parents=True, exist_ok=True)
    (storage / "predictions").mkdir(parents=True, exist_ok=True)
    (storage / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "STORAGE_DIR", storage)
    return storage
