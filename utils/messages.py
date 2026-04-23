"""사용자 노출 한글 메시지 카탈로그 (IMPLEMENTATION_PLAN §1.7).

규칙 (automl-project.mdc):
- 페이지/서비스는 한글 리터럴을 직접 쓰지 말고 ``Msg`` 상수를 참조한다.
- 파라미터가 필요한 메시지는 함수로 노출한다.
- 로깅 이벤트명은 ``utils.events.Event`` 에서 따로 관리.
"""

from __future__ import annotations


class Msg:
    """UI 사용자 노출 메시지 상수."""

    # 공통
    SAVE_SUCCESS = "저장이 완료되었습니다."
    SAVE_FAILED = "저장에 실패했습니다."
    DELETE_CONFIRM = "정말 삭제하시겠습니까? 연결된 데이터도 함께 삭제됩니다."
    UNEXPECTED_ERROR = "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    # 프로젝트 (FR-010~019)
    PROJECT_REQUIRED = "먼저 프로젝트를 선택하거나 생성해 주세요."
    PROJECT_CREATED = "프로젝트가 생성되었습니다."
    PROJECT_UPDATED = "프로젝트가 수정되었습니다."
    PROJECT_DELETED = "프로젝트가 삭제되었습니다."
    PROJECT_NAME_REQUIRED = "프로젝트명을 입력해 주세요."
    PROJECT_NAME_TOO_LONG = "프로젝트명은 100자 이내로 입력해 주세요."
    PROJECT_DESC_TOO_LONG = "설명은 500자 이내로 입력해 주세요."

    # 데이터셋 (FR-030~036)
    DATASET_REQUIRED = "먼저 데이터셋을 업로드해 주세요."
    UPLOAD_SUCCESS = "업로드가 완료되었습니다."
    UPLOAD_FAILED = "업로드에 실패했습니다."
    FILE_EMPTY = "빈 파일입니다. 데이터가 포함된 파일을 업로드해 주세요."
    FILE_PARSE_FAILED = "파일을 읽을 수 없습니다. 파일 형식과 인코딩을 확인해 주세요."
    HEADER_MISSING = "헤더 행이 비어 있습니다. 컬럼명이 있는 파일을 업로드해 주세요."
    DUPLICATED_COLUMNS = "중복된 컬럼명이 있습니다. 컬럼명을 고유하게 만들어 주세요."

    # 학습 (FR-040~049)
    TARGET_REQUIRED = "예측 대상(타깃) 컬럼을 선택해 주세요."
    TASK_TYPE_REQUIRED = "작업 유형(분류/회귀)을 선택해 주세요."
    ALGORITHM_REQUIRED = "학습할 알고리즘을 하나 이상 선택해 주세요."
    TRAINING_STARTED = "학습을 시작합니다."
    TRAINING_COMPLETED = "학습이 완료되었습니다."
    TRAINING_FAILED = "학습에 실패했습니다."
    TRAINING_RESULT_REQUIRED = "먼저 학습을 실행해 주세요."

    # 고급 전처리 (FR-055~058, §9.9)
    PREPROCESSING_ADVANCED_TITLE = "고급 전처리 (선택)"
    PREPROCESSING_CUSTOM_BADGE = "⚙️ 커스텀 전처리 적용됨"
    PREPROCESSING_PREVIEW_TITLE = "피처 변환 미리보기"
    PREPROCESSING_PREVIEW_HINT = "아래 설정을 고정한 상태에서 미리 계산됩니다 (실제 학습은 수행하지 않음)."
    PREPROCESSING_SMOTE_UNAVAILABLE = (
        "imbalanced-learn 이 설치되지 않아 SMOTE 를 사용할 수 없습니다."
    )
    PREPROCESSING_SMOTE_CLASSIFICATION_ONLY = "SMOTE 는 분류 작업에서만 사용할 수 있습니다."
    PREPROCESSING_PREVIEW_AUTO_DOWNGRADED = (
        "고카디널리티로 판단되어 자동으로 frequency 인코딩으로 강등된 컬럼이 있습니다."
    )

    # 모델 (FR-070~075)
    MODEL_SAVED = "모델이 저장되었습니다."
    MODEL_SAVE_REQUIRES_SUCCESS = "성공한 모델만 저장할 수 있습니다."
    MODEL_DELETED = "모델이 삭제되었습니다."
    MODEL_REQUIRED = "먼저 저장된 모델을 선택해 주세요."

    # 예측 (FR-080~089)
    PREDICTION_STARTED = "예측을 시작합니다."
    PREDICTION_COMPLETED = "예측이 완료되었습니다."
    PREDICTION_FAILED = "예측에 실패했습니다."
    INPUT_MISSING_COLUMNS = "필수 입력 컬럼이 누락되었습니다."


def upload_extension_not_allowed(ext: str, allowed: tuple[str, ...]) -> str:
    """허용되지 않은 확장자 메시지."""
    allowed_str = ", ".join(f".{e}" for e in allowed)
    return f"허용되지 않는 파일 형식입니다(.{ext}). 사용 가능: {allowed_str}."


def upload_too_large(size_mb: float, max_mb: int) -> str:
    """업로드 크기 초과 메시지."""
    return f"파일 크기({size_mb:.1f}MB)가 허용치({max_mb}MB)를 초과합니다."


def entity_not_found(name: str, entity_id: int | str) -> str:
    """엔터티 조회 실패 메시지."""
    return f"{name}(id={entity_id})를 찾을 수 없습니다."


def missing_columns(columns: list[str]) -> str:
    """예측 입력에서 누락된 컬럼 메시지."""
    return f"{Msg.INPUT_MISSING_COLUMNS} (누락: {', '.join(columns)})"
