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
    TRAINING_PREPROCESSING_CUSTOMIZED = "training.preprocessing_customized"  # FR-055~058, §9.8
    TRAINING_ALGORITHMS_FILTERED = "training.algorithms_filtered"  # FR-067, §10.3
    TRAINING_TUNING_DOWNGRADED = "training.tuning_downgraded"  # §10.3 (튜닝 본체는 §11)

    # 모델/아티팩트
    MODEL_SAVED = "model.saved"
    MODEL_DELETED = "model.deleted"
    MODEL_INFLUENCE_COMPUTED = "model.influence_computed"
    ARTIFACT_SAVE_FAILED = "artifact.save_failed"
    MODEL_LEGACY_PREPROCESSING_LOADED = "model.legacy_preprocessing_loaded"  # §9.8 하위호환

    # 예측
    PREDICTION_STARTED = "prediction.started"
    PREDICTION_COMPLETED = "prediction.completed"
    PREDICTION_FAILED = "prediction.failed"
