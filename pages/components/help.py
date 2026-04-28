"""인앱 매뉴얼(헬프) 컴포넌트 (IMPLEMENTATION_PLAN §6.8).

설계 요약:

- 각 페이지 상단에 ``render_help("03_training")`` 한 줄을 추가하면
  ``docs/manual/03_training.md`` 내용을 ``st.expander("❓ 이 페이지 도움말")`` 로
  렌더한다.
- 콘텐츠는 repo 의 ``docs/manual/*.md`` 를 단일 출처(SSOT) 로 삼는다.
  파일이 없으면 조용히 no-op — 페이지 본체 흐름을 절대 막지 않는다.
- 파일 mtime 을 캐시 키에 포함시켜 Streamlit 재실행 시에도 수정 내용이
  즉시 반영된다 (`@st.cache_data` 무효화).
- ``ml`` / ``services`` 에 의존하지 않는 순수 UI 컴포넌트 — 매뉴얼 허브
  페이지(``pages/00_manual.py``) 에서도 재사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import streamlit as st

from utils.log_utils import get_logger

logger = get_logger(__name__)

# ``__file__`` 기준 두 단계 상위 = 프로젝트 루트.
_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
MANUAL_DIR: Final[Path] = _ROOT / "docs" / "manual"


# ----------------------------------------------------- 공통 로더


@st.cache_data(show_spinner=False)
def _read_manual_text(path_str: str, mtime_ns: int) -> str:  # noqa: ARG001 - mtime 은 캐시 키 전용
    """디스크에서 매뉴얼 MD 를 읽어 캐시한다.

    ``mtime_ns`` 는 캐시 키로만 쓰이지만 인자로 유지해야 Streamlit 의
    ``@st.cache_data`` 가 파일 수정 시 캐시를 버린다.
    """
    return Path(path_str).read_text(encoding="utf-8")


def load_manual(key: str) -> str | None:
    """매뉴얼 MD 한 건을 로드. 없으면 ``None``.

    ``key`` 는 ``docs/manual/<key>.md`` 의 파일명(확장자 제외) 과 동일.
    트래버설 방지를 위해 ``/`` 가 포함된 키는 거부.
    """
    if "/" in key or ".." in key:
        logger.warning("help.invalid_key", extra={"key": key})
        return None
    path = MANUAL_DIR / f"{key}.md"
    if not path.is_file():
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
        return _read_manual_text(str(path), mtime_ns)
    except OSError:  # pragma: no cover - 파일시스템 단발성 에러
        logger.exception("help.read_failed", extra={"key": key})
        return None


def list_manual_keys() -> list[str]:
    """``docs/manual/*.md`` 에 실재하는 키 목록(정렬됨).

    ``_`` 로 시작하는 파일은 허브 페이지가 내부적으로 쓰는 부분 템플릿으로
    취급해 목록에서 제외한다 (예: ``_hub_intro.md``).
    """
    if not MANUAL_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in MANUAL_DIR.glob("*.md") if p.is_file() and not p.name.startswith("_")
    )


# ----------------------------------------------------- 공개 렌더 API


def render_help(
    key: str,
    *,
    title: str = "❓ 이 페이지 도움말",
    default_open: bool = False,
) -> bool:
    """페이지 상단에 접을 수 있는 도움말 expander 를 렌더한다.

    반환값은 "도움말이 실제로 렌더됐는가" — MD 파일이 없으면 ``False`` 를
    돌려주고 화면엔 아무 것도 그리지 않는다. 이 동작 덕에 페이지는 헬프 유무와
    무관하게 안전하게 동작한다.
    """
    text = load_manual(key)
    if text is None:
        return False
    with st.expander(title, expanded=default_open):
        st.markdown(text)
    return True


def render_manual_section(
    key: str,
    *,
    anchor: str | None = None,
) -> bool:
    """허브 페이지용 — expander 대신 섹션 앵커 + 본문만 렌더.

    ``anchor`` 는 HTML id 가 되어 TOC 링크에서 점프할 수 있다.
    """
    text = load_manual(key)
    if text is None:
        return False
    if anchor:
        # HTML anchor (Streamlit 이 markdown 안의 raw HTML 일부 허용)
        st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)
    st.markdown(text)
    return True


__all__ = [
    "MANUAL_DIR",
    "list_manual_keys",
    "load_manual",
    "render_help",
    "render_manual_section",
]
