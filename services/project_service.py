"""Project Service (IMPLEMENTATION_PLAN §4.1, FR-020~024).

책임:
- 프로젝트 CRUD 유스케이스의 오케스트레이션 (입력 검증 → Repository → 감사 로그)
- 트랜잭션 경계 소유 (``session_scope`` 사용)
- ORM → DTO 변환 (UI 는 ORM 객체를 직접 보지 않는다)

레이어 규약 (``.cursor/rules/service-layer.mdc``):
- Streamlit 타입/호출 금지, SQLAlchemy 세션을 외부로 노출하지 않는다.
- 모든 변경 액션은 ``audit_repository.write`` 를 통해 감사 로그를 남긴다.
- 사용자 노출 메시지는 ``utils.messages.Msg`` 에서만 가져온다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from repositories import audit_repository, project_repository
from repositories.base import session_scope
from services.dto import ProjectDTO
from utils.errors import NotFoundError, ValidationError
from utils.events import Event
from utils.log_utils import get_logger, log_event
from utils.messages import Msg, entity_not_found

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from repositories.models import Project

logger = get_logger(__name__)

NAME_MAX_LEN: Final[int] = 100
DESC_MAX_LEN: Final[int] = 500


# ---------------------------------------------------------------- Validation


def _clean_name(name: str | None) -> str:
    """프로젝트명 정규화. 빈 값/초과 길이를 ``ValidationError`` 로 변환."""
    candidate = (name or "").strip()
    if not candidate:
        raise ValidationError(Msg.PROJECT_NAME_REQUIRED)
    if len(candidate) > NAME_MAX_LEN:
        raise ValidationError(Msg.PROJECT_NAME_TOO_LONG)
    return candidate


def _clean_description(description: str | None) -> str | None:
    """설명 정규화. ``None``/빈 문자열은 ``None`` 으로, 초과 길이는 예외."""
    if description is None:
        return None
    trimmed = description.strip()
    if not trimmed:
        return None
    if len(trimmed) > DESC_MAX_LEN:
        raise ValidationError(Msg.PROJECT_DESC_TOO_LONG)
    return trimmed


def _to_dto(session: Session, project: Project) -> ProjectDTO:
    """ORM → DTO (dataset_count/model_count 집계 포함).

    집계는 프로젝트 단위 SQL COUNT 2회. 목록 렌더링에서 N+1 이 되지만 MVP 규모에서 허용.
    """
    dataset_count = project_repository.count_datasets(session, project.project_id)
    model_count = project_repository.count_models(session, project.project_id)
    return ProjectDTO.from_orm(
        project,
        dataset_count=dataset_count,
        model_count=model_count,
    )


# ---------------------------------------------------------- Public use-cases


def create_project(name: str, description: str | None = None) -> ProjectDTO:
    """FR-020: 새 프로젝트 생성.

    - 이름 중복(동일 소유자 범위) 시 ``ValidationError``.
    - 성공 시 ``AuditLog`` 에 ``project.created`` 기록.
    """
    clean_name = _clean_name(name)
    clean_desc = _clean_description(description)

    with session_scope() as session:
        if project_repository.exists_by_name(session, clean_name):
            raise ValidationError(f"이미 같은 이름의 프로젝트가 있습니다: {clean_name}")
        project = project_repository.insert(
            session,
            project_name=clean_name,
            description=clean_desc,
        )
        audit_repository.write(
            session,
            action_type=Event.PROJECT_CREATED,
            target_type="Project",
            target_id=project.project_id,
            detail={"name": clean_name},
        )
        log_event(
            logger,
            Event.PROJECT_CREATED,
            project_id=project.project_id,
            project_name=clean_name,
        )
        return _to_dto(session, project)


def list_projects() -> list[ProjectDTO]:
    """FR-021: 프로젝트 목록. 최신순. MVP(AUTH_MODE=none) 는 전체 목록."""
    with session_scope() as session:
        rows = project_repository.list_all(session)
        return [_to_dto(session, project) for project in rows]


def get_project(project_id: int) -> ProjectDTO:
    """FR-022: 단건 조회. 없으면 ``NotFoundError``."""
    with session_scope() as session:
        project = project_repository.get(session, project_id)
        if project is None:
            raise NotFoundError(entity_not_found("프로젝트", project_id))
        return _to_dto(session, project)


def update_project(
    project_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ProjectDTO:
    """FR-023: 프로젝트 수정.

    - ``name`` / ``description`` 중 ``None`` 인 항목은 **수정하지 않음**.
    - 이름 중복(자기 자신 제외) 시 ``ValidationError``.
    - 최소 1개 이상의 항목이 실제로 변경될 것을 권장하지만, 동일 값 저장도 허용.
    """
    clean_name: str | None = None
    if name is not None:
        clean_name = _clean_name(name)

    clean_desc: str | None = None
    desc_provided = description is not None
    if desc_provided:
        clean_desc = _clean_description(description)

    with session_scope() as session:
        project = project_repository.get(session, project_id)
        if project is None:
            raise NotFoundError(entity_not_found("프로젝트", project_id))

        if clean_name is not None and project_repository.exists_by_name(
            session,
            clean_name,
            exclude_project_id=project_id,
        ):
            raise ValidationError(f"이미 같은 이름의 프로젝트가 있습니다: {clean_name}")

        project.project_name = clean_name if clean_name is not None else project.project_name
        if desc_provided:
            project.description = clean_desc
        session.flush()

        audit_repository.write(
            session,
            action_type=Event.PROJECT_UPDATED,
            target_type="Project",
            target_id=project.project_id,
            detail={
                "name": clean_name,
                "description_changed": desc_provided,
            },
        )
        log_event(
            logger,
            Event.PROJECT_UPDATED,
            project_id=project.project_id,
        )
        return _to_dto(session, project)


def delete_project(project_id: int, *, cascade: bool = False) -> None:
    """FR-024: 프로젝트 삭제.

    - ``cascade=False`` 인데 연결된 Dataset/TrainingJob/Model 이 있으면 ``ValidationError``.
    - ``cascade=True`` 는 ORM 관계(``cascade="all, delete-orphan"``)로 하위 레코드까지 일괄 삭제.
      파일 정리는 별도 Service 책임(단계 4.2/4.4).
    """
    with session_scope() as session:
        project = project_repository.get(session, project_id)
        if project is None:
            raise NotFoundError(entity_not_found("프로젝트", project_id))

        if not cascade:
            ds_count = project_repository.count_datasets(session, project_id)
            job_count = project_repository.count_training_jobs(session, project_id)
            model_count = project_repository.count_models(session, project_id)
            if ds_count or job_count or model_count:
                raise ValidationError(
                    "연결된 리소스가 있어 삭제할 수 없습니다 "
                    f"(데이터셋 {ds_count}, 학습 {job_count}, 모델 {model_count}). "
                    "cascade=True 로 다시 시도하거나 먼저 정리해 주세요.",
                )

        project_repository.delete(session, project_id)
        audit_repository.write(
            session,
            action_type=Event.PROJECT_DELETED,
            target_type="Project",
            target_id=project_id,
            detail={"cascade": cascade},
        )
        log_event(
            logger,
            Event.PROJECT_DELETED,
            project_id=project_id,
            cascade=cascade,
        )


__all__ = [
    "NAME_MAX_LEN",
    "DESC_MAX_LEN",
    "create_project",
    "list_projects",
    "get_project",
    "update_project",
    "delete_project",
]
