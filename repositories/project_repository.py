"""Project Repository (IMPLEMENTATION_PLAN §2.3).

- 함수형: ``session: Session`` 을 첫 인자로 받는다. 트랜잭션 경계는 Service 책임.
- 반환은 ORM 엔터티 그대로 (상위 레이어에서 DTO 로 매핑).
- 발견 실패는 ``None`` 반환. "필수 조회" 는 Service 가 ``NotFoundError`` 로 변환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

from repositories.models import Dataset, Model, Project, TrainingJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def insert(
    session: Session,
    *,
    project_name: str,
    description: str | None = None,
    owner_user_id: int | None = None,
) -> Project:
    project = Project(
        project_name=project_name,
        description=description,
        owner_user_id=owner_user_id,
    )
    session.add(project)
    session.flush()
    return project


def update(
    session: Session,
    project_id: int,
    *,
    project_name: str | None = None,
    description: str | None = None,
) -> Project | None:
    project = session.get(Project, project_id)
    if project is None:
        return None
    if project_name is not None:
        project.project_name = project_name
    if description is not None:
        project.description = description
    session.flush()
    return project


def delete(session: Session, project_id: int) -> bool:
    project = session.get(Project, project_id)
    if project is None:
        return False
    session.delete(project)
    session.flush()
    return True


def get(session: Session, project_id: int) -> Project | None:
    return session.get(Project, project_id)


def list_by_owner(
    session: Session,
    owner_user_id: int | None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[Project]:
    """소유자 기준 목록 조회. ``owner_user_id=None`` 이면 소유자 미지정 프로젝트만."""
    stmt = select(Project)
    if owner_user_id is None:
        stmt = stmt.where(Project.owner_user_id.is_(None))
    else:
        stmt = stmt.where(Project.owner_user_id == owner_user_id)

    stmt = stmt.order_by(Project.created_at.desc(), Project.project_id.desc())
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()


def list_all(
    session: Session,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[Project]:
    """소유자 관계없이 전체 목록 (MVP AUTH_MODE=none 에서 주 사용)."""
    stmt = select(Project).order_by(Project.created_at.desc(), Project.project_id.desc())
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().all()


def count(session: Session, owner_user_id: int | None = None) -> int:
    stmt = select(func.count(Project.project_id))
    if owner_user_id is not None:
        stmt = stmt.where(Project.owner_user_id == owner_user_id)
    return int(session.execute(stmt).scalar_one())


def exists_by_name(
    session: Session,
    project_name: str,
    *,
    owner_user_id: int | None = None,
    exclude_project_id: int | None = None,
) -> bool:
    stmt = select(func.count(Project.project_id)).where(Project.project_name == project_name)
    if owner_user_id is None:
        stmt = stmt.where(Project.owner_user_id.is_(None))
    else:
        stmt = stmt.where(Project.owner_user_id == owner_user_id)
    if exclude_project_id is not None:
        stmt = stmt.where(Project.project_id != exclude_project_id)
    return session.execute(stmt).scalar_one() > 0


def count_datasets(session: Session, project_id: int) -> int:
    """해당 프로젝트에 속한 Dataset 수."""
    stmt = select(func.count(Dataset.dataset_id)).where(Dataset.project_id == project_id)
    return int(session.execute(stmt).scalar_one())


def count_models(session: Session, project_id: int) -> int:
    """해당 프로젝트의 TrainingJob 에 속한 Model 레코드 수."""
    stmt = (
        select(func.count(Model.model_id))
        .join(TrainingJob, TrainingJob.training_job_id == Model.training_job_id)
        .where(TrainingJob.project_id == project_id)
    )
    return int(session.execute(stmt).scalar_one())


def count_training_jobs(session: Session, project_id: int) -> int:
    """해당 프로젝트에 속한 TrainingJob 수 (상태 무관)."""
    stmt = select(func.count(TrainingJob.training_job_id)).where(
        TrainingJob.project_id == project_id
    )
    return int(session.execute(stmt).scalar_one())
