"""관리자 페이지(``pages/07_admin.py``) AppTest 검증.

검증 포인트 (IMPLEMENTATION_PLAN §6.7, FR-090~093):
- DB 가드 (프로젝트 가드는 둘 필요 없음 — 관리자 뷰)
- 빈 DB → 통계 카드 0 + "이력 없음" 안내
- 학습 이력 탭 → 프로젝트명/상태/베스트 정보 포함한 테이블 렌더 (slow, 실제 학습 수반)
- 필터 selectbox 존재 (프로젝트/상태/기간)
- 최근 실패 탭 → audit_log 시드 1건 이상이면 렌더
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml.schemas import TrainingConfig
from repositories import audit_repository
from repositories.base import session_scope
from services import dataset_service, project_service, training_service

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "07_admin.py")


@dataclass
class FakeUpload:
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def _new_page() -> AppTest:
    return AppTest.from_file(PAGE_PATH, default_timeout=120)


# -------------------------------------------------------------------- guards


def test_admin_page_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_admin_page_empty_db_shows_zero_stats_and_empty_info(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    # 5개 이상 metric (stats + 실패 2)
    assert len(at.metric) >= 5
    # 학습/예측 탭 모두 "이력이 없습니다" 안내
    info_texts = " ".join(i.value for i in at.info)
    assert "학습 이력이 없습니다" in info_texts


def test_admin_page_filter_widgets_exist(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    keys = {s.key for s in at.selectbox}
    assert {
        "admin_project_picker",
        "admin_status_picker",
        "admin_period_picker",
    }.issubset(keys)


# ----------------------------------------------------------- happy training


@pytest.mark.slow
def test_admin_page_training_tab_lists_job_with_project_name(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project = project_service.create_project("admin-cls")
    ds = dataset_service.upload_dataset(
        project.id,
        FakeUpload(name=classification_csv.name, data=classification_csv.read_bytes()),
    )
    training_service.run_training(
        TrainingConfig(
            dataset_id=ds.id,
            task_type="classification",
            target_column="species",
        )
    )

    at = _new_page().run()
    assert not at.exception

    # dataframe 1개 이상 노출 (학습 탭이 첫 번째)
    assert len(at.dataframe) >= 1
    # 통계 카드 중 "학습 잡" 에 최소 1
    metric_values = [str(m.value) for m in at.metric]
    assert any(v.strip() == "1" for v in metric_values)


# --------------------------------------------------------- failure listing


def test_admin_page_failure_tab_renders_seeded_audit(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    with session_scope() as session:
        audit_repository.write(
            session,
            action_type="training.failed",
            target_type="TrainingJob",
            target_id=1,
            detail={"error": "synthetic"},
        )

    at = _new_page().run()
    assert not at.exception

    # "최근 실패" 탭 내용이 기본 렌더되어 있진 않을 수 있음 — 탭 전환 없이도
    # admin_service.list_recent_failures 는 불리므로 dataframe 이 존재해야 함.
    # AppTest 는 모든 탭 콘텐츠를 평면적으로 평가한다.
    dataframes = at.dataframe
    assert dataframes, "감사 로그 시드 후 실패 탭에 최소 1개 dataframe 이 있어야 함"


def test_admin_page_project_filter_selection_is_preserved(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("filter-pres")
    at = _new_page().run()

    picker = at.selectbox(key="admin_project_picker")
    target_label = f"{project.name} (id={project.id})"
    picker.select(target_label).run()

    assert at.session_state["admin_filter_project_id"] == project.id
