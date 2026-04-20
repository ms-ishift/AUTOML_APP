"""Streamlit 세션 상태 헬퍼 (FR-002, FR-003, IMPLEMENTATION_PLAN §1.5).

규칙 (streamlit-ui.mdc):
- 세션 상태 키는 임의 문자열 대신 ``SessionKey`` 상수만 사용.
- 플래시(토스트) 메시지는 ``flash()`` 로 큐잉 → ``consume_flashes()`` 로 1회 소비.
- ``require_project`` / ``require_dataset`` 는 전제 조건이 없으면 ``st.stop()`` 으로 렌더링 중단.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import streamlit as st

from utils.messages import Msg

ToastLevel = Literal["success", "warning", "error", "info"]


class SessionKey:
    """세션 상태의 공식 키. 페이지/서비스는 이 상수만 사용한다."""

    CURRENT_PROJECT_ID: Final[str] = "current_project_id"
    CURRENT_DATASET_ID: Final[str] = "current_dataset_id"
    LAST_TRAINING_JOB_ID: Final[str] = "last_training_job_id"
    CURRENT_MODEL_ID: Final[str] = "current_model_id"  # §6.5 → §6.6 전달용
    FLASH: Final[str] = "flash"
    CURRENT_USER_ID: Final[str] = "current_user_id"  # AUTH_MODE=none → 0


@dataclass(frozen=True, slots=True)
class ToastMsg:
    """플래시 큐 단위. level 은 streamlit 함수 매핑에 사용."""

    level: ToastLevel
    message: str


def get_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


def clear_state(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(k, None)


def flash(level: ToastLevel, message: str) -> None:
    """다음 렌더링 사이클에서 표시할 메시지를 큐잉한다."""
    queue: list[ToastMsg] = st.session_state.setdefault(SessionKey.FLASH, [])
    queue.append(ToastMsg(level=level, message=message))


def consume_flashes() -> list[ToastMsg]:
    """큐에 쌓인 플래시 메시지를 모두 꺼내고 비운다."""
    queue: list[ToastMsg] = st.session_state.pop(SessionKey.FLASH, [])
    return list(queue)


def current_user_id() -> int:
    """현재 사용자 id. AUTH_MODE=none 에서는 시스템 사용자 id=0 (PLAN §2.2a)."""
    return int(st.session_state.get(SessionKey.CURRENT_USER_ID, 0))


def require_project() -> int:
    """현재 선택된 project_id 를 반환. 없으면 안내 후 렌더링 중단."""
    pid = get_state(SessionKey.CURRENT_PROJECT_ID)
    if pid is None:
        st.info(Msg.PROJECT_REQUIRED)
        st.stop()
    return int(pid)


def require_dataset() -> int:
    """현재 선택된 dataset_id 를 반환. 없으면 안내 후 렌더링 중단."""
    did = get_state(SessionKey.CURRENT_DATASET_ID)
    if did is None:
        st.info(Msg.DATASET_REQUIRED)
        st.stop()
    return int(did)
