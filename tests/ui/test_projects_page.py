"""프로젝트 관리 페이지(``pages/01_projects.py``) 렌더/상호작용 테스트.

IMPLEMENTATION_PLAN §6.1 의 수용 기준을 AppTest 로 검증:
- 생성 → 목록 갱신 → 선택 → 사이드바에 반영
- 수정 플로우 (인라인 폼)
- 삭제 플로우 + cascade 확인 / cascade 거부 시 ValidationError → flash error
- DB 미초기화 가드
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from repositories import dataset_repository, project_repository
from repositories.base import session_scope
from services import project_service
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "01_projects.py")


def _new_page() -> AppTest:
    return AppTest.from_file(PAGE_PATH, default_timeout=15)


# ----------------------------------------------------------- basic rendering


def test_page_renders_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # is_db_initialized 를 Project 페이지 쪽 import 사이트에서 오버라이드.
    # streamlit AppTest 는 페이지 모듈을 재로딩하므로 pages.* 네임스페이스에 직접 패치하지 않고
    # utils.db_utils.is_db_initialized 를 False 로 만들면 페이지 쪽에서 import 시점에 그대로 참조한다.
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_page_renders_empty_list(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    infos = " ".join(e.value for e in at.info)
    assert "아직 생성된 프로젝트가 없습니다" in infos


# --------------------------------------------------------------- create flow


def test_create_project_success_updates_list_and_selection(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    # 이름 입력 + 생성 클릭
    at.text_input(key="create_name").set_value("첫 프로젝트")
    at.text_area(key="create_desc").set_value("설명이에요")
    at.button[0].click().run()

    # 플래시(success) 가 남아 있어야 하고 세션에 current_project_id 가 설정됨
    current_id = at.session_state[SessionKey.CURRENT_PROJECT_ID]
    assert isinstance(current_id, int) and current_id > 0

    # 목록 렌더에 프로젝트명이 표시
    body = " ".join(m.value for m in at.markdown)
    assert "첫 프로젝트" in body
    # 사이드바에 성공 뱃지
    sidebar_success = " ".join(m.value for m in at.sidebar.success)
    assert "첫 프로젝트" in sidebar_success


def test_create_project_empty_name_shows_validation_error(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    at.text_input(key="create_name").set_value("   ")  # 공백만
    at.button[0].click().run()
    # 다음 사이클에서 flash(error) 가 렌더된다 (현재 사이클에도 rerun 이 일어나지 않음).
    errors = " ".join(e.value for e in at.error)
    assert "프로젝트명" in errors


def test_create_project_duplicate_name_fails(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project_service.create_project("중복 이름")
    at = _new_page().run()
    at.text_input(key="create_name").set_value("중복 이름")
    at.button[0].click().run()
    errors = " ".join(e.value for e in at.error)
    assert "같은 이름의 프로젝트가 있습니다" in errors


# -------------------------------------------------------------- select flow


def test_select_button_sets_current_project(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    p1 = project_service.create_project("A-proj")
    project_service.create_project("B-proj")

    at = _new_page().run()
    # p1 선택 버튼 클릭
    select_buttons = [b for b in at.button if b.key == f"project_select_{p1.id}"]
    assert select_buttons
    select_buttons[0].click().run()

    assert at.session_state[SessionKey.CURRENT_PROJECT_ID] == p1.id
    sidebar_success = " ".join(m.value for m in at.sidebar.success)
    assert "A-proj" in sidebar_success


def test_current_project_row_shows_marker_and_disabled_select(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("current-proj")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()

    body = " ".join(m.value for m in at.markdown)
    assert "★ current-proj" in body
    select_btn = next(b for b in at.button if b.key == f"project_select_{project.id}")
    assert select_btn.disabled is True


# ---------------------------------------------------------------- edit flow


def test_edit_flow_updates_name_and_description(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("before", description="old desc")

    at = _new_page().run()
    edit_btn = next(b for b in at.button if b.key == f"project_edit_{project.id}")
    edit_btn.click().run()

    # 인라인 폼이 노출되어야 한다
    at.text_input(key="edit_name").set_value("after")
    at.text_area(key="edit_desc").set_value("new desc")
    # 저장 버튼 클릭 (폼 안의 submit)
    save_btn = next(b for b in at.button if b.label == "저장" and getattr(b, "form_id", None))
    save_btn.click().run()

    # Service 레벨로 확인
    refreshed = project_service.get_project(project.id)
    assert refreshed.name == "after"
    assert refreshed.description == "new desc"


def test_edit_flow_cancel_keeps_original(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("keep-me", description="keep-desc")
    at = _new_page().run()
    edit_btn = next(b for b in at.button if b.key == f"project_edit_{project.id}")
    edit_btn.click().run()

    at.text_input(key="edit_name").set_value("changed")
    cancel_btn = next(b for b in at.button if b.label == "취소" and getattr(b, "form_id", None))
    cancel_btn.click().run()

    refreshed = project_service.get_project(project.id)
    assert refreshed.name == "keep-me"
    # 편집 모드 세션 플래그도 정리됨
    assert "projects_edit_target_id" not in at.session_state


# -------------------------------------------------------------- delete flow


def test_delete_without_children_succeeds(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("to-delete")
    at = _new_page().run()
    del_btn = next(b for b in at.button if b.key == f"project_delete_{project.id}")
    del_btn.click().run()

    # 확인 블록의 "삭제 실행" 버튼 클릭
    confirm = next(b for b in at.button if b.key == "delete_confirm_btn")
    confirm.click().run()

    # DB 에서 실제로 사라짐
    with session_scope() as session:
        assert project_repository.get(session, project.id) is None


def test_delete_with_children_requires_cascade(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    # 연결 데이터셋 1건을 직접 시드해 children 상태 만들기
    project = project_service.create_project("parent")
    target = tmp_storage / "datasets" / f"{uuid.uuid4().hex}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(classification_csv, target)
    with session_scope() as session:
        dataset_repository.insert(
            session,
            project_id=project.id,
            file_name=classification_csv.name,
            file_path=str(target),
            row_count=10,
            column_count=3,
        )

    # 1차: cascade=False 로 시도 → ValidationError → flash 로 노출, 프로젝트 유지
    at = _new_page().run()
    del_btn = next(b for b in at.button if b.key == f"project_delete_{project.id}")
    del_btn.click().run()
    # 기본 cascade=True 체크 상태이므로 의도적으로 해제
    cascade_cb = at.checkbox(key="delete_cascade_flag")
    cascade_cb.set_value(False)
    confirm = next(b for b in at.button if b.key == "delete_confirm_btn")
    confirm.click().run()

    errors = " ".join(e.value for e in at.error)
    assert "연결된 리소스" in errors
    with session_scope() as session:
        assert project_repository.get(session, project.id) is not None

    # 2차: cascade=True 로 재시도 → 성공
    at2 = _new_page().run()
    del_btn2 = next(b for b in at2.button if b.key == f"project_delete_{project.id}")
    del_btn2.click().run()
    # cascade 기본 True 유지
    confirm2 = next(b for b in at2.button if b.key == "delete_confirm_btn")
    confirm2.click().run()

    with session_scope() as session:
        assert project_repository.get(session, project.id) is None


def test_delete_current_project_clears_session(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("current-to-delete")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()

    del_btn = next(b for b in at.button if b.key == f"project_delete_{project.id}")
    del_btn.click().run()
    confirm = next(b for b in at.button if b.key == "delete_confirm_btn")
    confirm.click().run()

    assert SessionKey.CURRENT_PROJECT_ID not in at.session_state
