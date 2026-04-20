from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from repositories import (
    dataset_repository,
    model_repository,
    prediction_repository,
    project_repository,
    training_repository,
)
from repositories.models import Dataset, Project


@pytest.fixture()
def project_and_dataset(db_session: Session) -> tuple[Project, Dataset]:
    p = project_repository.insert(db_session, project_name="P")
    ds = dataset_repository.insert(
        db_session,
        project_id=p.project_id,
        file_name="iris.csv",
        file_path="/tmp/iris.csv",
        row_count=150,
        column_count=5,
    )
    db_session.commit()
    return p, ds


def test_insert_sets_defaults(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="species",
        metric_key="f1",
    )
    db_session.commit()
    assert job.status == "pending"
    assert job.excluded_columns_json == []
    assert job.started_at is None
    assert job.ended_at is None


def test_update_status_running_sets_started_at(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="accuracy",
    )
    db_session.commit()

    running = training_repository.update_status(db_session, job.training_job_id, "running")
    db_session.commit()
    assert running is not None
    assert running.started_at is not None

    done = training_repository.update_status(db_session, job.training_job_id, "completed")
    db_session.commit()
    assert done is not None
    assert done.ended_at is not None


def test_update_status_invalid(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="accuracy",
    )
    db_session.commit()
    with pytest.raises(ValueError):
        training_repository.update_status(db_session, job.training_job_id, "weird")


def test_append_run_log_accumulates(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="regression",
        target_column="target",
        metric_key="rmse",
    )
    db_session.commit()

    training_repository.append_run_log(db_session, job.training_job_id, "line 1")
    training_repository.append_run_log(db_session, job.training_job_id, "line 2\n")
    db_session.commit()

    loaded = training_repository.get(db_session, job.training_job_id)
    assert loaded is not None
    assert loaded.run_log == "line 1\nline 2\n"


def test_list_by_project_orders_desc(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    j1 = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    j2 = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()

    rows = training_repository.list_by_project(db_session, p.project_id)
    assert [r.training_job_id for r in rows][0] == j2.training_job_id
    assert [r.training_job_id for r in rows][1] == j1.training_job_id


def test_model_bulk_insert_and_mark_best(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()

    rows = [
        {
            "model_name": "rf_v1",
            "algorithm_name": "RandomForest",
            "metric_score": 0.91,
            "metric_summary_json": {"f1": 0.91},
            "feature_schema_json": {"numeric": ["a"], "categorical": [], "target": "y"},
        },
        {
            "model_name": "lr_v1",
            "algorithm_name": "LogisticRegression",
            "metric_score": 0.85,
            "metric_summary_json": {"f1": 0.85},
            "feature_schema_json": {"numeric": ["a"], "categorical": [], "target": "y"},
        },
    ]
    entities = model_repository.bulk_insert(db_session, job.training_job_id, rows)
    db_session.commit()
    assert all(e.model_id is not None for e in entities)
    assert all(e.is_best is False for e in entities)

    best = model_repository.mark_best(db_session, job.training_job_id, entities[0].model_id)
    db_session.commit()

    assert best is not None
    assert best.is_best is True
    by_job = model_repository.list_by_training_job(db_session, job.training_job_id)
    best_count = sum(1 for m in by_job if m.is_best)
    assert best_count == 1


def test_model_update_paths_and_cross_job_mark_best_rejected(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job1 = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    job2 = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()

    m1 = model_repository.bulk_insert(
        db_session,
        job1.training_job_id,
        [{"model_name": "m1", "algorithm_name": "A"}],
    )[0]
    db_session.commit()

    model_repository.update_paths(
        db_session,
        m1.model_id,
        model_path="/tmp/m1.joblib",
        preprocessing_path="/tmp/m1.prep.joblib",
    )
    db_session.commit()
    reloaded = model_repository.get(db_session, m1.model_id)
    assert reloaded is not None
    assert reloaded.model_path == "/tmp/m1.joblib"
    assert reloaded.preprocessing_path == "/tmp/m1.prep.joblib"

    # 다른 job 의 best 지정 시도 → None
    assert model_repository.mark_best(db_session, job2.training_job_id, m1.model_id) is None


def test_prediction_repository_insert_and_status(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    p, ds = project_and_dataset
    job = training_repository.insert(
        db_session,
        project_id=p.project_id,
        dataset_id=ds.dataset_id,
        task_type="classification",
        target_column="y",
        metric_key="f1",
    )
    db_session.commit()
    [model] = model_repository.bulk_insert(
        db_session,
        job.training_job_id,
        [{"model_name": "m", "algorithm_name": "A"}],
    )
    db_session.commit()

    pj = prediction_repository.insert(
        db_session,
        model_id=model.model_id,
        input_type="form",
    )
    db_session.commit()
    assert pj.status == "pending"

    updated = prediction_repository.update_status(
        db_session,
        pj.prediction_job_id,
        "completed",
        result_path="/tmp/result.csv",
    )
    db_session.commit()
    assert updated is not None
    assert updated.status == "completed"
    assert updated.result_path == "/tmp/result.csv"

    with pytest.raises(ValueError):
        prediction_repository.insert(db_session, model_id=model.model_id, input_type="weird")


def test_audit_repository_list_filters(
    db_session: Session, project_and_dataset: tuple[Project, Dataset]
) -> None:
    from repositories import audit_repository
    from repositories.models import SYSTEM_USER_ID, User

    db_session.add(
        User(user_id=SYSTEM_USER_ID, login_id="system", user_name="시스템", role="system")
    )
    db_session.commit()

    audit_repository.write(
        db_session, action_type="project.created", target_type="Project", target_id=1
    )
    audit_repository.write(
        db_session, action_type="dataset.uploaded", target_type="Dataset", target_id=1
    )
    audit_repository.write(
        db_session, action_type="project.created", target_type="Project", target_id=2
    )
    db_session.commit()

    only_project = audit_repository.list_logs(db_session, action_type="project.created")
    assert len(only_project) == 2

    by_target = audit_repository.list_logs(db_session, target_type="Dataset", target_id=1)
    assert len(by_target) == 1
    assert by_target[0].action_type == "dataset.uploaded"
