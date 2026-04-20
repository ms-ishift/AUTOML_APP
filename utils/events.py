"""구조화 로그 / AuditLog 이벤트명 상수 (IMPLEMENTATION_PLAN §1.7).

사용 패턴::

    from utils.events import Event
    log_event(logger, Event.DATASET_UPLOADED, project_id=1, size=1024)

포맷: ``<domain>.<verb>`` (소문자, 점 구분).
AuditLog.action_type 에도 동일 상수를 재사용한다.
"""

from __future__ import annotations


class Event:
    """로그/감사 이벤트 상수."""

    # 프로젝트
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"

    # 데이터셋
    DATASET_UPLOADED = "dataset.uploaded"
    DATASET_UPLOAD_FAILED = "dataset.upload_failed"
    DATASET_DELETED = "dataset.deleted"

    # 학습 잡
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    MODEL_TRAIN_FAILED = "training.model_failed"  # 개별 알고리즘 실패

    # 모델/아티팩트
    MODEL_SAVED = "model.saved"
    MODEL_DELETED = "model.deleted"
    ARTIFACT_SAVE_FAILED = "artifact.save_failed"

    # 예측
    PREDICTION_STARTED = "prediction.started"
    PREDICTION_COMPLETED = "prediction.completed"
    PREDICTION_FAILED = "prediction.failed"
