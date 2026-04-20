from __future__ import annotations

import pytest
import streamlit as st

from utils.session_utils import (
    SessionKey,
    ToastMsg,
    clear_state,
    consume_flashes,
    current_user_id,
    flash,
    get_state,
    set_state,
)


@pytest.fixture(autouse=True)
def _clean_session():
    try:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
    except Exception:
        pass
    yield
    try:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
    except Exception:
        pass


def test_set_and_get_state() -> None:
    set_state(SessionKey.CURRENT_PROJECT_ID, 1)
    assert get_state(SessionKey.CURRENT_PROJECT_ID) == 1


def test_clear_state_removes_keys() -> None:
    set_state(SessionKey.CURRENT_PROJECT_ID, 1)
    set_state(SessionKey.CURRENT_DATASET_ID, 2)
    clear_state(SessionKey.CURRENT_PROJECT_ID)
    assert get_state(SessionKey.CURRENT_PROJECT_ID) is None
    assert get_state(SessionKey.CURRENT_DATASET_ID) == 2


def test_flash_queue_and_consume() -> None:
    flash("success", "저장됨")
    flash("warning", "주의")

    queued = consume_flashes()
    assert queued == [
        ToastMsg(level="success", message="저장됨"),
        ToastMsg(level="warning", message="주의"),
    ]
    assert consume_flashes() == []


def test_current_user_id_defaults_to_zero() -> None:
    assert current_user_id() == 0
    set_state(SessionKey.CURRENT_USER_ID, 42)
    assert current_user_id() == 42
