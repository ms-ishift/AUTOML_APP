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


def is_db_initialized(required_tables: Iterable[str] = REQUIRED_TABLES) -> bool:
    """앱이 기대하는 테이블이 모두 존재하면 ``True``.

    실패 상황(파일 없음/권한/드라이버 오류 등)은 ``False`` 로 수렴 — 홈 화면에서
    "init_db 를 먼저 실행하세요" 안내를 일관되게 띄우기 위함.
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
    return True


__all__ = ["REQUIRED_TABLES", "is_db_initialized"]
