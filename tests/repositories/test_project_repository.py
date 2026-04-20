from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from repositories import audit_repository, project_repository
from repositories.models import SYSTEM_USER_ID, AuditLog, User


@pytest.fixture()
def system_user(db_session: Session) -> User:
    u = User(
        user_id=SYSTEM_USER_ID,
        login_id="system",
        user_name="시스템",
        role="system",
    )
    db_session.add(u)
    db_session.commit()
    return u


def test_insert_and_get(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p1", description="d")
    db_session.commit()

    loaded = project_repository.get(db_session, p.project_id)
    assert loaded is not None
    assert loaded.project_name == "p1"
    assert loaded.description == "d"
    assert loaded.owner_user_id is None


def test_insert_with_owner(db_session: Session, system_user: User) -> None:
    p = project_repository.insert(
        db_session, project_name="p-owner", owner_user_id=system_user.user_id
    )
    db_session.commit()
    assert p.owner_user_id == 0


def test_update_partial(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p1", description="old")
    db_session.commit()

    updated = project_repository.update(db_session, p.project_id, description="new")
    db_session.commit()
    assert updated is not None
    assert updated.project_name == "p1"
    assert updated.description == "new"


def test_update_nonexistent_returns_none(db_session: Session) -> None:
    assert project_repository.update(db_session, 9999, project_name="x") is None


def test_delete(db_session: Session) -> None:
    p = project_repository.insert(db_session, project_name="p-del")
    db_session.commit()

    assert project_repository.delete(db_session, p.project_id) is True
    db_session.commit()
    assert project_repository.get(db_session, p.project_id) is None

    # 두 번째 삭제는 False
    assert project_repository.delete(db_session, p.project_id) is False


def test_list_by_owner_none_vs_some(db_session: Session, system_user: User) -> None:
    project_repository.insert(db_session, project_name="unowned-1")
    project_repository.insert(db_session, project_name="unowned-2")
    project_repository.insert(db_session, project_name="owned", owner_user_id=system_user.user_id)
    db_session.commit()

    unowned = project_repository.list_by_owner(db_session, None)
    owned = project_repository.list_by_owner(db_session, system_user.user_id)

    assert {p.project_name for p in unowned} == {"unowned-1", "unowned-2"}
    assert [p.project_name for p in owned] == ["owned"]


def test_list_all_ordered_desc(db_session: Session) -> None:
    project_repository.insert(db_session, project_name="old")
    project_repository.insert(db_session, project_name="new")
    db_session.commit()

    rows = project_repository.list_all(db_session)
    # 최신이 먼저
    assert [p.project_name for p in rows][0] == "new"


def test_count(db_session: Session, system_user: User) -> None:
    project_repository.insert(db_session, project_name="a")
    project_repository.insert(db_session, project_name="b", owner_user_id=system_user.user_id)
    db_session.commit()

    assert project_repository.count(db_session) == 2
    assert project_repository.count(db_session, owner_user_id=system_user.user_id) == 1


def test_exists_by_name(db_session: Session) -> None:
    project_repository.insert(db_session, project_name="dup")
    db_session.commit()

    assert project_repository.exists_by_name(db_session, "dup") is True
    assert project_repository.exists_by_name(db_session, "none") is False


def test_cascade_deletes_datasets_and_training_jobs(db_session: Session) -> None:
    from repositories import dataset_repository, training_repository

    p = project_repository.insert(db_session, project_name="p")
    ds = dataset_repository.insert(
        db_session,
        project_id=p.project_id,
        file_name="x.csv",
        file_path="/tmp/x.csv",
        row_count=1,
        column_count=2,
    )
    training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()

    assert project_repository.delete(db_session, p.project_id) is True
    db_session.commit()

    from repositories.models import Dataset, TrainingJob

    assert db_session.query(Dataset).count() == 0
    assert db_session.query(TrainingJob).count() == 0


def test_audit_repository_defaults_to_system_user(db_session: Session, system_user: User) -> None:
    audit_repository.write(
        db_session,
        action_type="project.created",
        target_type="Project",
        target_id=1,
    )
    db_session.commit()

    row = db_session.query(AuditLog).one()
    assert row.user_id == SYSTEM_USER_ID
    assert row.action_type == "project.created"
