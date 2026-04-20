from __future__ import annotations

import pytest

from utils.errors import (
    AppError,
    MLTrainingError,
    NotFoundError,
    PredictionInputError,
    StorageError,
    ValidationError,
)


def test_default_message_used_when_not_provided() -> None:
    err = ValidationError()
    assert str(err) == ValidationError.default_message


def test_custom_message_overrides_default() -> None:
    err = MLTrainingError("학습 실패 세부 사유")
    assert str(err) == "학습 실패 세부 사유"
    assert err.user_message == "학습 실패 세부 사유"


def test_cause_is_chained() -> None:
    cause = ValueError("boom")
    err = StorageError("저장 실패", cause=cause)
    assert err.__cause__ is cause


@pytest.mark.parametrize(
    "cls",
    [ValidationError, NotFoundError, MLTrainingError, PredictionInputError, StorageError],
)
def test_all_subclasses_inherit_app_error(cls: type[AppError]) -> None:
    assert issubclass(cls, AppError)
    with pytest.raises(AppError):
        raise cls()
