"""공용 Flash / Toast 렌더러 (FR-003, IMPLEMENTATION_PLAN §1.6).

페이지 최상단에서 ``render_flashes()`` 를 한 번 호출하면 직전 사이클에서 쌓인 메시지를 모두 출력한다.
"""

from __future__ import annotations

import streamlit as st

from utils.session_utils import consume_flashes


def render_flashes() -> None:
    """세션 큐에 쌓인 플래시 메시지를 한 번에 표시하고 비운다."""
    for toast in consume_flashes():
        if toast.level == "success":
            st.success(toast.message)
        elif toast.level == "warning":
            st.warning(toast.message)
        elif toast.level == "error":
            st.error(toast.message)
        else:
            st.info(toast.message)
