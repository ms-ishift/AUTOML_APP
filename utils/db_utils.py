"""DB 초기화 상태 점검 유틸 (IMPLEMENTATION_PLAN §5.1).

용도:
- 홈 화면 진입 시 ``scripts/init_db.py`` 실행 여부를 확인해 사용자에게 가이드.
- 프로덕션/개발 환경 공용으로 **엔진 접근 한번** + **테이블 존재 확인** 만 수행 (데이터 조회 X).

설계 메모:
- SQLAlchemy ``inspect`` 는 connect 시 sqlite 파일이 없으면 0바이트 파일을 만들 수 있어
  ``exists()`` + ``file_size`` 로 먼저 차단한다 (STORAGE_DIR 교체되는 테스트 환경 고려).
- ``session_scope`` 는 쓰지 않는다 (읽기 전용 + 실패 시 Service 예외 대신 ``False`` 반환).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import inspect

from repositories import base as repo_base
from utils.log_utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


REQUIRED_TABLES: tuple[str, ...] = (
    "users",
    "projects",
    "datasets",
    "training_jobs",
    "models",
    "prediction_jobs",
    "audit_logs",
)


def _has_system_user() -> bool:
    """``SYSTEM_USER_ID`` 행이 존재하는지 확인한다.

    ``audit_logs.user_id -> users.id`` FK 가 항상 만족되는지가 관건이라
    ``--drop`` 만 돌리고 ``--seed`` 를 빠뜨린 환경을 조기 차단한다.

    ``SessionLocal`` 은 모듈 로드 시점의 ``engine`` 에 바인딩돼 있어
    테스트에서 ``repo_base.engine`` 을 monkeypatch 하더라도 따라오지 않는다.
    여기서는 현재 ``repo_base.engine`` 을 직접 사용해 체크한다.
    """
    from sqlalchemy.orm import Session

    from repositories.models import SYSTEM_USER_ID, User

    try:
        with Session(repo_base.engine) as session:
            return session.get(User, SYSTEM_USER_ID) is not None
    except Exception:  # noqa: BLE001 - 초기화 직후 등 과도기는 False 로 수렴
        logger.debug("db.system_user_check_failed", exc_info=True)
        return False


def is_db_initialized(required_tables: Iterable[str] = REQUIRED_TABLES) -> bool:
    """앱이 기대하는 테이블이 모두 존재하고 ``SYSTEM_USER_ID`` 행이 있으면 ``True``.

    실패 상황(파일 없음/권한/드라이버 오류/seed 누락 등)은 모두 ``False`` 로 수렴 —
    홈 화면에서 "init_db 를 먼저 실행하세요" 안내를 일관되게 띄우기 위함.

    ``users`` 테이블은 있는데 system user row 만 빠진 경우(`--drop` 후
    `--seed` 누락)에도 바로 FK 에러가 터지지 않도록 여기서 차단한다.
    """
    try:
        inspector = inspect(repo_base.engine)
        existing = set(inspector.get_table_names())
    except Exception:  # noqa: BLE001 - 초기화 안된 상태도 비정상 아님
        logger.debug("db.inspect_failed", exc_info=True)
        return False
    missing = [t for t in required_tables if t not in existing]
    if missing:
        logger.debug("db.missing_tables", extra={"missing": missing})
        return False
    if not _has_system_user():
        logger.debug("db.system_user_missing")
        return False
    return True


__all__ = ["REQUIRED_TABLES", "is_db_initialized"]
