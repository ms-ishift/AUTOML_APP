"""데이터 업로드 페이지(``pages/02_dataset_upload.py``) AppTest 검증.

테스트 범위 (IMPLEMENTATION_PLAN §6.2 수용 기준):
- DB 미초기화 가드 / 프로젝트 미선택 가드
- 업로드 → 목록 갱신 → 프리뷰 + 프로파일 렌더
- 선택 상호작용 (``SessionKey.CURRENT_DATASET_ID``)
- 삭제 확인 플로우 + 현재 선택 해제
- stale ``CURRENT_DATASET_ID`` 자동 정리

업로드 실행 자체는 `dataset_service.upload_dataset` 가 Service 테스트(22건)에서 이미 검증된 만큼
UI 쪽은 **직접 Service 호출로 시드**한 뒤 렌더/상호작용만 확인한다. 파일 제출 흐름은 패치로 대체.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from repositories import dataset_repository
from repositories.base import session_scope
from services import dataset_service, project_service
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "02_dataset_upload.py")


@dataclass
class FakeUpload:
    """Streamlit ``UploadedFile`` 호환 duck-type."""

    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def _new_page() -> AppTest:
    return AppTest.from_file(PAGE_PATH, default_timeout=15)


def _seed_project_with_dataset(csv_path: Path, *, name: str = "ds-proj") -> tuple[int, int]:
    project = project_service.create_project(name)
    upload = FakeUpload(name=csv_path.name, data=csv_path.read_bytes())
    dto = dataset_service.upload_dataset(project.id, upload)
    return project.id, dto.id


# ----------------------------------------------------------------- guards


def test_page_renders_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_page_requires_project_when_no_selection(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 프로젝트를 선택" in warnings


# ---------------------------------------------------- list / preview render


def test_list_renders_uploaded_datasets(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()
    assert not at.exception

    # 사이드바에 프로젝트 표시 + 본문에 파일명 노출
    body = " ".join(m.value for m in at.markdown)
    assert classification_csv.name in body
    # 업로드 폼 subheader + 사이드바에 프로젝트명이 노출
    subheader_text = " ".join(s.value for s in at.subheader)
    sidebar_text = " ".join(s.value for s in at.sidebar.success)
    assert "ds-proj" in subheader_text or "ds-proj" in sidebar_text
    # 기본 선택이 없으면 프리뷰 섹션은 아직 렌더되지 않는다
    assert SessionKey.CURRENT_DATASET_ID not in at.session_state
    assert any(b.key == f"dataset_select_{dataset_id}" for b in at.button)


def test_preview_and_profile_tabs_render_when_dataset_selected(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()
    assert not at.exception

    # 2개의 탭(샘플 데이터 / 컬럼 프로파일)이 렌더된다
    tab_labels = [t.label for t in at.tabs]
    assert "샘플 데이터" in tab_labels
    assert "컬럼 프로파일" in tab_labels

    # 프로파일 탭의 메트릭이 행/컬럼 수를 담고 있어야 한다
    metric_labels = [m.label for m in at.metric]
    assert "행 수" in metric_labels
    assert "컬럼 수" in metric_labels


# --------------------------------------------------------- interactions


def test_select_button_sets_current_dataset(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    select_btn = next(b for b in at.button if b.key == f"dataset_select_{dataset_id}")
    select_btn.click().run()

    assert at.session_state[SessionKey.CURRENT_DATASET_ID] == dataset_id
    body = " ".join(m.value for m in at.markdown)
    assert "★ " in body


def test_upload_form_empty_submit_flashes_warning(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("empty-upload-proj")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()

    # form 의 submit 버튼만 클릭 (파일 없음)
    submit_btns = [b for b in at.button if b.label == "업로드"]
    assert submit_btns, "업로드 버튼을 찾지 못했습니다"
    submit_btns[0].click().run()

    warnings = " ".join(w.value for w in at.warning)
    assert "업로드할 파일을 선택해 주세요" in warnings


# ------------------------------------------------------------- delete flow


def test_delete_dataset_removes_record_and_file(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    del_btn = next(b for b in at.button if b.key == f"dataset_delete_{dataset_id}")
    del_btn.click().run()

    confirm = next(b for b in at.button if b.key == "dataset_delete_confirm_btn")
    confirm.click().run()

    with session_scope() as session:
        assert dataset_repository.get(session, dataset_id) is None


def test_delete_current_dataset_clears_session(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()

    del_btn = next(b for b in at.button if b.key == f"dataset_delete_{dataset_id}")
    del_btn.click().run()
    confirm = next(b for b in at.button if b.key == "dataset_delete_confirm_btn")
    confirm.click().run()

    assert SessionKey.CURRENT_DATASET_ID not in at.session_state


def test_delete_cancel_keeps_record(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed_project_with_dataset(classification_csv)
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    del_btn = next(b for b in at.button if b.key == f"dataset_delete_{dataset_id}")
    del_btn.click().run()

    cancel = next(b for b in at.button if b.key == "dataset_delete_cancel_btn")
    cancel.click().run()

    with session_scope() as session:
        assert dataset_repository.get(session, dataset_id) is not None


# ----------------------------------------------------- stale session cleanup


def test_stale_current_dataset_id_is_cleared(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("stale-ds")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = 9999  # 없는 id
    at.run()

    assert SessionKey.CURRENT_DATASET_ID not in at.session_state
