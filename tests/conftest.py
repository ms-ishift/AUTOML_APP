"""공통 pytest fixtures.

- 샘플 CSV 경로 (session scope)
- 임시 SQLite 엔진 / SessionFactory / Session (function scope)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.base import Base
from repositories.models import (  # noqa: F401  # 엔터티 등록을 위해 import 필요
    AuditLog,
    Dataset,
    Model,
    PredictionJob,
    Project,
    TrainingJob,
    User,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def classification_csv(samples_dir: Path) -> Path:
    path = samples_dir / "classification.csv"
    if not path.exists():
        pytest.skip(
            f"샘플 파일 없음: {path} (해결: `make samples` 또는 "
            f"`python scripts/generate_samples.py`)",
        )
    return path


@pytest.fixture(scope="session")
def regression_csv(samples_dir: Path) -> Path:
    path = samples_dir / "regression.csv"
    if not path.exists():
        pytest.skip(
            f"샘플 파일 없음: {path} (해결: `make samples` 또는 "
            f"`python scripts/generate_samples.py`)",
        )
    return path


# ------------------------------------------------------------- DB fixtures


@pytest.fixture()
def sqlite_engine(tmp_path: Path) -> Iterator[Engine]:
    """테스트용 파일 기반 SQLite 엔진. 각 테스트마다 새 DB."""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=sqlite_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@pytest.fixture()
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
