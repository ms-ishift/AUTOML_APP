"""도메인 예외 계층 (IMPLEMENTATION_PLAN §1.2).

원칙:
- Service/ML 경계에서 예상 가능한 오류는 반드시 AppError 하위로 변환 후 raise.
- UI 는 `except AppError` 로 받고 `str(e)` 로 한글 메시지를 그대로 표시.
- `__cause__` 로 원인 예외를 체이닝해 로깅 시 traceback 보존.
"""

from __future__ import annotations


class AppError(Exception):
    """앱 공통 예외. ``user_message`` 는 UI 표시용 한글 메시지."""

    default_message = "처리 중 오류가 발생했습니다."

    def __init__(
        self,
        user_message: str | None = None,
        *,
        cause: Exception | None = None,
    ) -> None:
        self.user_message = user_message or self.default_message
        super().__init__(self.user_message)
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return self.user_message


class ValidationError(AppError):
    """입력값 형식/범위 오류."""

    default_message = "입력값이 올바르지 않습니다."


class NotFoundError(AppError):
    """대상 엔터티를 찾을 수 없음."""

    default_message = "요청한 데이터를 찾을 수 없습니다."


class MLTrainingError(AppError):
    """학습 파이프라인 실패."""

    default_message = "학습을 완료하지 못했습니다."


class PredictionInputError(AppError):
    """예측 입력 스키마 불일치 (FR-083)."""

    default_message = "예측 입력 데이터가 올바르지 않습니다."


class StorageError(AppError):
    """파일 저장/로드 I/O 실패."""

    default_message = "파일 저장/로드 중 오류가 발생했습니다."
