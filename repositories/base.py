"""ORM 베이스 (IMPLEMENTATION_PLAN §2.1).

구성:
- ``engine``: ``settings.DATABASE_URL`` 기반 단일 엔진
- ``SessionLocal``: ``sessionmaker`` 팩토리
- ``Base``: ``DeclarativeBase`` (SQLAlchemy 2.x 스타일)
- ``session_scope``: 커밋/롤백/클로즈를 보장하는 컨텍스트 매니저 (Service 레이어에서만 사용)

트랜잭션 경계는 Service 가 소유한다. Repository 는 세션을 인자로 받는 함수 모듈이다.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.orm import Session


def _resolve_sqlite_path(url: str) -> str:
    """sqlite 경로의 상대 경로를 프로젝트 루트 기준으로 정규화하고 부모 디렉터리를 보장한다."""
    if not url.startswith("sqlite"):
        return url

    # "sqlite:///db/app.db" 형태만 처리. 절대 경로(sqlite:////...)는 그대로 둔다.
    prefix, _, rest = url.partition(":///")
    if not rest or rest.startswith("/"):
        return url

    p = Path(rest)
    if not p.is_absolute():
        p = (Path(settings.STORAGE_DIR).parent / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}:///{p}"


_DB_URL = _resolve_sqlite_path(settings.DATABASE_URL)

_connect_args: dict[str, object] = (
    {"check_same_thread": False} if _DB_URL.startswith("sqlite") else {}
)

engine = create_engine(
    _DB_URL,
    future=True,
    echo=False,
    connect_args=_connect_args,
)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(
    dbapi_connection: Any, connection_record: Any
) -> None:  # noqa: ARG001
    """SQLite 는 연결마다 PRAGMA foreign_keys=ON 이 필요하다 (cascade 동작용)."""
    try:
        import sqlite3

        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    except Exception:  # pragma: no cover - 최악의 경우도 로깅 생략하고 진행
        pass


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    """모든 ORM 엔터티의 공통 베이스."""


class TimestampMixin:
    """``created_at`` / ``updated_at`` 자동 관리 믹스인."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Service 전용 트랜잭션 컨텍스트.

    사용 예::

        with session_scope() as session:
            project_repository.insert(session, name="demo")
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
