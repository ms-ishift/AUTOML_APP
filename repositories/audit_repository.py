"""AuditLog Repository (IMPLEMENTATION_PLAN §2.3, §2.2a).

MVP(AUTH_MODE=none) 정책:
- ``user_id`` 가 ``None`` 이면 자동으로 ``SYSTEM_USER_ID(=0)`` 로 기록한다.
- 시스템 유저 레코드가 아직 시드되지 않았을 수 있으므로 ``user_id`` FK 는 nullable,
  FK 제약은 SET NULL 로 처리되어 시드 누락 상황에서도 로깅이 끊기지 않는다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from repositories.models import SYSTEM_USER_ID, AuditLog

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.orm import Session


def write(
    session: Session,
    *,
    action_type: str,
    user_id: int | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    detail: dict[str, Any] | None = None,
) -> AuditLog:
    """감사 로그 한 건을 기록한다. ``user_id`` 미지정 시 시스템 유저로 귀속."""
    log = AuditLog(
        user_id=user_id if user_id is not None else SYSTEM_USER_ID,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        detail_json=detail,
    )
    session.add(log)
    session.flush()
    return log


def list_logs(
    session: Session,
    *,
    user_id: int | None = None,
    action_type: str | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[AuditLog]:
    """다차원 필터로 감사 로그를 조회 (시간 역순)."""
    stmt = select(AuditLog)
    if user_id is not None:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if action_type is not None:
        stmt = stmt.where(AuditLog.action_type == action_type)
    if target_type is not None:
        stmt = stmt.where(AuditLog.target_type == target_type)
    if target_id is not None:
        stmt = stmt.where(AuditLog.target_id == target_id)
    if since is not None:
        stmt = stmt.where(AuditLog.action_time >= since)
    if until is not None:
        stmt = stmt.where(AuditLog.action_time < until)

    stmt = stmt.order_by(AuditLog.action_time.desc(), AuditLog.audit_log_id.desc())
    stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()
