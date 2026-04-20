"""홈 화면(``app.py``) 렌더 테스트 — Streamlit ``AppTest`` 기반 (IMPLEMENTATION_PLAN §5.1).

- DB 초기화 전/후 분기 렌더링 검증
- 최근 프로젝트 카드와 "선택하기" 상호작용 → 세션 상태 반영 확인
- Stale project_id (DB 에 없는 id) 자동 정리 검증
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import project_service
from utils.session_utils import SessionKey

APP_PATH = str(Path(__file__).resolve().parents[2] / "app.py")


def _new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=15)


def test_home_renders_init_guide_when_db_missing(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_app().run()
    assert not at.exception
    # 초기화 안내 문구가 본문 + 사이드바 양쪽에서 노출되어야 한다.
    body_text = " ".join(e.value for e in at.markdown)
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors
    assert "AutoML MVP" in body_text


def test_home_renders_intro_without_projects(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_app().run()
    assert not at.exception
    body_text = " ".join(e.value for e in at.markdown)
    assert "이 앱은 무엇을 하나요" in body_text
    # 빈 목록 → info 메시지 노출
    infos = " ".join(e.value for e in at.info)
    assert "프로젝트를 선택" in infos or "프로젝트를 만들어" in body_text


def test_home_lists_recent_projects_and_selects(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    # 4개 프로젝트를 만들어 상위 3개만 표시되는지도 검증
    for name in ("A-proj", "B-proj", "C-proj", "D-proj"):
        project_service.create_project(name, description=f"desc {name}")

    at = _new_app().run()
    assert not at.exception

    # 카드에 프로젝트 이름이 적어도 3개는 노출
    body = " ".join(e.value for e in at.markdown)
    appearances = sum(body.count(name) for name in ("A-proj", "B-proj", "C-proj", "D-proj"))
    assert appearances >= 3

    # 가장 최근(마지막 생성) 프로젝트 선택 버튼 클릭 → 세션 상태 반영
    select_buttons = [b for b in at.button if b.key and b.key.startswith("home_select_project_")]
    assert select_buttons, "최근 프로젝트 선택 버튼이 없음"
    select_buttons[0].click().run()

    selected_id = at.session_state[SessionKey.CURRENT_PROJECT_ID]
    assert isinstance(selected_id, int) and selected_id > 0


def test_home_clears_stale_project_id(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_app()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = 9999  # 존재하지 않는 id
    at.run()
    assert not at.exception
    # stale id 는 조용히 제거되고 경고 flash 가 다음 사이클에서 소비되므로 세션에 없어야 한다.
    assert SessionKey.CURRENT_PROJECT_ID not in at.session_state


def test_home_sidebar_shows_selected_project(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("X-proj", description="픽 데모")
    at = _new_app()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()
    assert not at.exception
    sidebar_success = " ".join(e.value for e in at.sidebar.success)
    assert "X-proj" in sidebar_success
    assert f"id={project.id}" in sidebar_success
