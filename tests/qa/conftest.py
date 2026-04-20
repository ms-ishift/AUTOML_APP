"""QA 회귀 수트 공통 fixture (IMPLEMENTATION_PLAN §7.3, NFR-004).

- Service 직접 호출 + ``streamlit.testing.v1.AppTest`` 를 한 테스트 안에서 섞기 때문에
  Service/UI conftest 의 autouse 엔진 override 패턴을 그대로 복제한다.
- 이 디렉터리는 "실패 경로가 앱을 죽이지 않는다" 를 회귀로 묶는 용도이므로,
  느린 학습 플로우는 `@pytest.mark.slow` 로만 격리한다.
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
    """``session_scope()`` 가 테스트 전용 sqlite 엔진을 사용하도록 주입."""
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
