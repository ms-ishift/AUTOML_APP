"""매뉴얼 허브 페이지 (IMPLEMENTATION_PLAN §6.8, FR-100).

전체 매뉴얼 MD 를 한 화면에 나열하고, 상단 검색창으로 섹션을 필터링한다.
홈의 *"문서 보기"* 버튼과 사이드바 매뉴얼 링크의 종착점이다.

UX 결정:
- 좌측 상단에 TOC, 본문은 순차 렌더. 앵커 링크로 점프.
- 검색은 단순 부분일치(대소문자 무시). 매칭이 일어난 섹션만 본문에 남는다.
- 개별 섹션의 세부 ``st.expander`` 는 쓰지 않는다 — 검색 후 내용이 직접 보여야
  스캐닝 속도가 빠르기 때문.
- DB/Project 가 없어도 열린다(진입 장벽 제거) — 사이드바는 가능한 정보만 표시.

규약 (`.cursor/rules/streamlit-ui.mdc`):
- Service 는 ``project_service.get_project`` 만 선택적으로 호출해 사이드바에 프로젝트명을 표시.
- 매뉴얼 콘텐츠는 ``docs/manual/*.md`` 의 단일 출처.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import streamlit as st

from pages.components.help import (
    list_manual_keys,
    load_manual,
    render_manual_section,
)
from pages.components.layout import (
    configure_page,
    render_page_header,
    render_sidebar,
)
from services import project_service
from utils.db_utils import is_db_initialized
from utils.errors import AppError, NotFoundError
from utils.session_utils import SessionKey, get_state

if TYPE_CHECKING:
    from services.dto import ProjectDTO


PAGE_TITLE: Final[str] = "매뉴얼"
PAGE_CAPTION: Final[str] = (
    "각 페이지 사용법과 트러블슈팅을 한 화면에서 검색·확인합니다."
)

# MD 파일 키 → 사이드바 TOC 라벨 매핑.
# 리스트에 없는 파일은 자동으로 "기타" 로 취급.
TOC_LABELS: dict[str, str] = {
    "00_overview": "개요",
    "01_projects": "프로젝트 관리",
    "02_dataset_upload": "데이터 업로드",
    "03_training": "학습",
    "04_results": "결과 비교",
    "05_models": "모델 관리",
    "06_prediction": "예측",
    "07_admin": "이력/관리자",
    "troubleshooting": "트러블슈팅",
}


def _load_current_project(db_ready: bool) -> ProjectDTO | None:
    if not db_ready:
        return None
    project_id = get_state(SessionKey.CURRENT_PROJECT_ID)
    if project_id is None:
        return None
    try:
        return project_service.get_project(int(project_id))
    except (NotFoundError, AppError):
        return None


def _label_for(key: str) -> str:
    return TOC_LABELS.get(key, key)


def _filter_keys(keys: list[str], query: str) -> list[str]:
    """검색어가 본문에 포함된 키만 남긴다. 빈 쿼리는 전체 반환."""
    q = query.strip().lower()
    if not q:
        return keys
    kept: list[str] = []
    for key in keys:
        text = load_manual(key) or ""
        if q in text.lower() or q in _label_for(key).lower():
            kept.append(key)
    return kept


def _render_toc(keys: list[str]) -> None:
    st.markdown("#### 목차")
    if not keys:
        st.caption("검색 결과 없음")
        return
    for key in keys:
        st.markdown(f"- [{_label_for(key)}](#{key})")


def _render_body(keys: list[str]) -> None:
    if not keys:
        st.info("검색어와 일치하는 섹션이 없습니다. 검색어를 바꾸거나 비워 주세요.")
        return
    for i, key in enumerate(keys):
        if i > 0:
            st.divider()
        rendered = render_manual_section(key, anchor=key)
        if not rendered:
            # 파일은 있었지만 삭제됐거나 읽기 실패
            st.warning(f"`{key}.md` 를 찾지 못했습니다.")


def main() -> None:
    configure_page(PAGE_TITLE)

    db_ready = is_db_initialized()
    current_project = _load_current_project(db_ready)

    render_sidebar(current_project=current_project, db_ready=db_ready)
    render_page_header(PAGE_TITLE, caption=PAGE_CAPTION)

    intro = load_manual("_hub_intro")
    if intro:
        st.markdown(intro)

    all_keys = list_manual_keys()
    if not all_keys:
        st.error(
            "`docs/manual/` 디렉터리에 매뉴얼 파일이 없습니다. "
            "저장소가 온전한지 확인하세요."
        )
        return

    query = st.text_input(
        "검색",
        value="",
        placeholder="예: CSV, 베스트, libomp, FR-073",
        key="manual_search_query",
    )

    filtered = _filter_keys(all_keys, query)

    left, right = st.columns([1, 3], gap="large")
    with left:
        _render_toc(filtered)
    with right:
        _render_body(filtered)


if __name__ == "__main__":
    main()
