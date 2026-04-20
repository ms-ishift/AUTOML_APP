"""Project Service 통합 테스트 (IMPLEMENTATION_PLAN §4.1, §4.6).

- 실제 sqlite 엔진을 통해 Service → Repository → ORM 경로를 모두 통과시킨다.
- ``session_scope`` 는 autouse fixture ``_override_session_local`` 로 테스트 DB 를 바라본다.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import sessionmaker

from repositories import dataset_repository, project_repository, training_repository
from repositories.models import AuditLog, User
from services import project_service
from services.dto import ProjectDTO
from utils.errors import NotFoundError, ValidationError
from utils.events import Event

# ----------------------------------------------------------- helpers


def _insert_dataset_and_job(sqlite_engine, project_id: int) -> None:
    """연결 리소스(cascade 가드 확인용) 생성."""
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        ds = dataset_repository.insert(
            session,
            project_id=project_id,
            file_name="x.csv",
            file_path="/tmp/x.csv",
            row_count=1,
            column_count=1,
        )
        training_repository.insert(
            session,
            project_id=project_id,
            dataset_id=ds.dataset_id,
            task_type="classification",
            target_column="y",
            metric_key="f1",
        )
        session.commit()


def _audit_actions(sqlite_engine) -> list[str]:
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        return [row.action_type for row in session.query(AuditLog).all()]


# ----------------------------------------------------------- create


def test_create_project_happy_path(sqlite_engine, seeded_system_user: User) -> None:
    dto = project_service.create_project("새 프로젝트", "설명")
    assert isinstance(dto, ProjectDTO)
    assert dto.name == "새 프로젝트"
    assert dto.description == "설명"
    assert dto.dataset_count == 0
    assert dto.model_count == 0
    assert Event.PROJECT_CREATED in _audit_actions(sqlite_engine)


def test_create_project_trims_whitespace(sqlite_engine, seeded_system_user: User) -> None:
    dto = project_service.create_project("  공백 이름  ", "  설명  ")
    assert dto.name == "공백 이름"
    assert dto.description == "설명"


def test_create_project_empty_description_becomes_none(
    sqlite_engine, seeded_system_user: User
) -> None:
    dto = project_service.create_project("이름만", "   ")
    assert dto.description is None


def test_create_project_rejects_empty_name(seeded_system_user: User) -> None:
    with pytest.raises(ValidationError):
        project_service.create_project("", "desc")
    with pytest.raises(ValidationError):
        project_service.create_project("   ", None)


def test_create_project_rejects_name_too_long(seeded_system_user: User) -> None:
    with pytest.raises(ValidationError):
        project_service.create_project("a" * (project_service.NAME_MAX_LEN + 1))


def test_create_project_rejects_description_too_long(seeded_system_user: User) -> None:
    with pytest.raises(ValidationError):
        project_service.create_project("ok", "d" * (project_service.DESC_MAX_LEN + 1))


def test_create_project_rejects_duplicate_name(seeded_system_user: User) -> None:
    project_service.create_project("dup", None)
    with pytest.raises(ValidationError):
        project_service.create_project("dup", None)


# ----------------------------------------------------------- list / get


def test_list_projects_orders_newest_first(seeded_system_user: User) -> None:
    project_service.create_project("a")
    project_service.create_project("b")
    rows = project_service.list_projects()
    assert [p.name for p in rows][0] == "b"
    assert len(rows) == 2


def test_list_projects_includes_counts(sqlite_engine, seeded_system_user: User) -> None:
    dto = project_service.create_project("with-dataset")
    _insert_dataset_and_job(sqlite_engine, dto.id)
    rows = project_service.list_projects()
    target = next(p for p in rows if p.id == dto.id)
    assert target.dataset_count == 1
    assert target.model_count == 0


def test_get_project_returns_dto(seeded_system_user: User) -> None:
    created = project_service.create_project("one")
    got = project_service.get_project(created.id)
    assert got.id == created.id
    assert got.name == "one"


def test_get_project_missing_raises_not_found(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        project_service.get_project(999)


# ----------------------------------------------------------- update


def test_update_project_partial_description(sqlite_engine, seeded_system_user: User) -> None:
    created = project_service.create_project("orig", "old")
    updated = project_service.update_project(created.id, description="new")
    assert updated.name == "orig"
    assert updated.description == "new"
    assert Event.PROJECT_UPDATED in _audit_actions(sqlite_engine)


def test_update_project_clears_description(seeded_system_user: User) -> None:
    created = project_service.create_project("orig", "old")
    updated = project_service.update_project(created.id, description="")
    assert updated.description is None


def test_update_project_rename(seeded_system_user: User) -> None:
    created = project_service.create_project("a")
    updated = project_service.update_project(created.id, name="a2")
    assert updated.name == "a2"


def test_update_project_rejects_duplicate_name(seeded_system_user: User) -> None:
    project_service.create_project("a")
    second = project_service.create_project("b")
    with pytest.raises(ValidationError):
        project_service.update_project(second.id, name="a")


def test_update_project_same_name_allowed(seeded_system_user: User) -> None:
    created = project_service.create_project("solo")
    updated = project_service.update_project(created.id, name="solo", description="d")
    assert updated.name == "solo"
    assert updated.description == "d"


def test_update_project_missing_raises_not_found(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        project_service.update_project(999, name="x")


def test_update_project_validates_inputs(seeded_system_user: User) -> None:
    created = project_service.create_project("valid")
    with pytest.raises(ValidationError):
        project_service.update_project(created.id, name="   ")
    with pytest.raises(ValidationError):
        project_service.update_project(
            created.id, description="d" * (project_service.DESC_MAX_LEN + 1)
        )


# ----------------------------------------------------------- delete


def test_delete_project_empty(sqlite_engine, seeded_system_user: User) -> None:
    created = project_service.create_project("to-delete")
    project_service.delete_project(created.id, cascade=False)
    with pytest.raises(NotFoundError):
        project_service.get_project(created.id)
    assert Event.PROJECT_DELETED in _audit_actions(sqlite_engine)


def test_delete_project_without_cascade_blocked_when_linked(
    sqlite_engine, seeded_system_user: User
) -> None:
    created = project_service.create_project("has-children")
    _insert_dataset_and_job(sqlite_engine, created.id)
    with pytest.raises(ValidationError):
        project_service.delete_project(created.id, cascade=False)
    got = project_service.get_project(created.id)
    assert got.id == created.id


def test_delete_project_cascade_removes_children(sqlite_engine, seeded_system_user: User) -> None:
    created = project_service.create_project("cascade-target")
    _insert_dataset_and_job(sqlite_engine, created.id)

    project_service.delete_project(created.id, cascade=True)

    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        assert project_repository.get(session, created.id) is None
        assert project_repository.count_datasets(session, created.id) == 0
        assert project_repository.count_training_jobs(session, created.id) == 0


def test_delete_project_missing_raises_not_found(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        project_service.delete_project(999, cascade=True)


# ----------------------------------------------------------- layer-boundary


def test_service_is_streamlit_free() -> None:
    """service-layer.mdc: Service 는 Streamlit import 를 가지지 않는다."""
    import inspect

    source = inspect.getsource(project_service)
    assert "streamlit" not in source
    assert "import st" not in source
