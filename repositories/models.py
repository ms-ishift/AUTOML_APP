"""ORM 엔터티 (IMPLEMENTATION_PLAN §2.2, 요구사항 §9.2).

스키마 매핑:
- 컬럼명은 요구사항(§9.2) 표기를 따른다.
- JSON 필드는 ``JSON`` 타입 (SQLite → TEXT, PostgreSQL → jsonb 호환).

AUTH_MODE 정책 (§2.2a):
- MVP 는 ``AUTH_MODE=none`` 이므로 ``Project.owner_user_id`` 는 **nullable**.
- 시스템 사용자 ``User(user_id=0, login_id='system', role='system')`` 을 init-db 시 시드한다.
- ``AuditLog.user_id`` 도 nullable 로 두어 시스템 컨텍스트(user_id=0)/미로그인 상태를 모두 수용.
- ``AUTH_MODE=basic`` 전환 체크리스트는 PLAN §8.1 로 이동 예정.

관계:
- Project 1 : N Dataset, TrainingJob
- Dataset 1 : N TrainingJob
- TrainingJob 1 : N Model
- Model 1 : N PredictionJob
- 삭제 정책은 보수적으로 Model/PredictionJob 까지는 ``cascade='all, delete-orphan'``
  (상위 삭제 시 하위 레코드까지 정리). 실 파일 삭제는 Service 책임.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repositories.base import Base, TimestampMixin

# --------------------------------------------------------------------- User


class User(Base, TimestampMixin):
    """사용자. MVP(AUTH_MODE=none)에서는 id=0 시스템 사용자 1건만 사용."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")


# ------------------------------------------------------------------ Project


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    project_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # AUTH_MODE=none: nullable. AUTH_MODE=basic 전환 시 NOT NULL 마이그레이션 필요.
    owner_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    project_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    training_jobs: Mapped[list[TrainingJob]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ------------------------------------------------------------------ Dataset


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="datasets")
    training_jobs: Mapped[list[TrainingJob]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# -------------------------------------------------------------- TrainingJob


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    training_job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[str] = mapped_column(String(20), nullable=False)  # classification|regression
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    excluded_columns_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metric_key: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending|running|completed|failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    run_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="training_jobs")
    dataset: Mapped[Dataset] = relationship(back_populates="training_jobs")
    models: Mapped[list[Model]] = relationship(
        back_populates="training_job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# -------------------------------------------------------------------- Model


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_job_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("training_jobs.training_job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_score: Mapped[float | None] = mapped_column(nullable=True)
    metric_summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    is_best: Mapped[bool] = mapped_column(default=False, nullable=False)
    # 아티팩트 경로는 ID 확정 후 채우므로 nullable (PLAN §4.3a 보상 로직).
    model_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    preprocessing_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    feature_schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
    )

    training_job: Mapped[TrainingJob] = relationship(back_populates="models")
    prediction_jobs: Mapped[list[PredictionJob]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ------------------------------------------------------------- PredictionJob


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    prediction_job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_type: Mapped[str] = mapped_column(String(20), nullable=False)  # form|file
    input_file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    result_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
    )

    model: Mapped[Model] = relationship(back_populates="prediction_jobs")


# ----------------------------------------------------------------- AuditLog


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # AUTH_MODE=none: 시스템 사용자(id=0) 또는 NULL 허용.
    user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    detail_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


SYSTEM_USER_ID = 0
SYSTEM_USER_LOGIN_ID = "system"
SYSTEM_USER_NAME = "시스템"
SYSTEM_USER_ROLE = "system"


__all__ = [
    "User",
    "Project",
    "Dataset",
    "TrainingJob",
    "Model",
    "PredictionJob",
    "AuditLog",
    "SYSTEM_USER_ID",
    "SYSTEM_USER_LOGIN_ID",
    "SYSTEM_USER_NAME",
    "SYSTEM_USER_ROLE",
]
