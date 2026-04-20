"""모델 관리 페이지(``pages/05_models.py``) AppTest 검증.

검증 포인트 (IMPLEMENTATION_PLAN §6.5, FR-074·FR-075):
- DB / 프로젝트 가드
- 모델 없음 안내 + "best only" 필터
- 모델 목록 렌더 (알고리즘, 주 지표, 베스트 배지)
- 상세 펼침 (입력 스키마 + metrics_summary)
- "예측하러 가기" → `SessionKey.CURRENT_MODEL_ID` 세팅 + (§6.6 미구현) info flash 폴백
- 삭제 확인 플로우 (확정 → DB/파일 정리, 취소 → 유지)

실제 `run_training` 으로 아티팩트를 시드하는 happy path 는 slow 마커.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml.schemas import TrainingConfig
from services import dataset_service, model_service, project_service, training_service
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "05_models.py")


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


def _seed_trained(
    csv: Path,
    *,
    target: str,
    name: str,
) -> tuple[int, int]:
    project = project_service.create_project(name)
    dto = dataset_service.upload_dataset(
        project.id, FakeUpload(name=csv.name, data=csv.read_bytes())
    )
    result = training_service.run_training(
        TrainingConfig(
            dataset_id=dto.id,
            task_type="classification",
            target_column=target,
        )
    )
    return project.id, result.job_id


# -------------------------------------------------------------------- guards


def test_models_page_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_models_page_requires_project(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 프로젝트를 선택" in warnings


def test_models_page_shows_empty_message_when_no_models(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("no-models")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()
    info_texts = " ".join(i.value for i in at.info)
    assert "아직 저장된 모델이 없습니다" in info_texts


# -------------------------------------------------------------- happy listing


@pytest.mark.slow
def test_models_page_lists_models_with_best_badge(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-list")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    assert not at.exception
    # 모든 학습 모델이 목록에 노출된다 (실패 포함)
    body = " ".join(m.value for m in at.markdown)
    models = model_service.list_models(project_id)
    assert models, "학습 직후 모델이 1건 이상 있어야 함"
    for m in models:
        assert m.algo_name in body
    # ★ 베스트 배지 렌더 확인
    assert any("★" in m.value for m in at.markdown)


@pytest.mark.slow
def test_models_page_best_only_filter(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-best-only")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    total_markdown_before = " ".join(m.value for m in at.markdown)
    total_models = model_service.list_models(project_id)
    # 필터 on → best 만 표시
    toggle = at.toggle(key="models_filter_best_only")
    toggle.set_value(True).run()
    body_after = " ".join(m.value for m in at.markdown)

    best = [m for m in total_models if m.is_best]
    non_best = [m for m in total_models if not m.is_best]

    for m in best:
        assert m.algo_name in body_after
    # non-best 이름은 숨겨진다 (이름 우연 일치를 피하기 위해 before 존재 + after 미존재 한 쌍만 체크)
    for m in non_best:
        assert m.algo_name in total_markdown_before
        assert m.algo_name not in body_after


# ------------------------------------------------------------------ detail


@pytest.mark.slow
def test_models_page_detail_expands_schema_and_metrics(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-detail")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    models = model_service.list_models(project_id)
    success = next(
        (m for m in models if m.metric_score is not None),
        None,
    )
    assert success is not None, "성공 모델이 1건 이상 필요"
    btn = next(b for b in at.button if b.key == f"models_detail_btn_{success.id}")
    btn.click().run()

    markdowns = " ".join(m.value for m in at.markdown)
    assert "입력 스키마" in markdowns
    assert "metrics_summary" in markdowns
    # 예측하러 가기 버튼 존재
    assert any(b.key == f"models_goto_predict_{success.id}" for b in at.button)


# ----------------------------------------------- go-predict sets session key


@pytest.mark.slow
def test_models_page_goto_predict_sets_current_model_id(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-goto")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    models = model_service.list_models(project_id)
    success = next(m for m in models if m.metric_score is not None)

    # 상세 열고 예측 이동 버튼 클릭
    next(b for b in at.button if b.key == f"models_detail_btn_{success.id}").click().run()
    next(b for b in at.button if b.key == f"models_goto_predict_{success.id}").click().run()

    assert at.session_state[SessionKey.CURRENT_MODEL_ID] == success.id
    # §6.6 폴백 info flash (switch_page 실패 → info)
    info_texts = " ".join(i.value for i in at.info)
    assert "예측 페이지는" in info_texts


# --------------------------------------------------------- delete flows


@pytest.mark.slow
def test_models_page_delete_removes_record_and_files(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-delete")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    models_before = model_service.list_models(project_id)
    target = next(m for m in models_before if m.metric_score is not None)

    # 삭제 버튼 → 인라인 컨펌 → 확정
    next(b for b in at.button if b.key == f"models_delete_btn_{target.id}").click().run()
    next(b for b in at.button if b.key == f"models_delete_confirm_{target.id}").click().run()

    assert not at.exception
    after = model_service.list_models(project_id)
    assert all(m.id != target.id for m in after)
    # 아티팩트 디렉터리도 제거됐는지 확인
    assert not (tmp_storage / "models" / str(target.id)).exists()
    success_texts = " ".join(s.value for s in at.success)
    assert "모델이 삭제" in success_texts


@pytest.mark.slow
def test_models_page_delete_cancel_keeps_record(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _ = _seed_trained(classification_csv, target="species", name="cls-delete-cancel")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.run()

    models_before = model_service.list_models(project_id)
    target = next(m for m in models_before if m.metric_score is not None)

    next(b for b in at.button if b.key == f"models_delete_btn_{target.id}").click().run()
    next(b for b in at.button if b.key == f"models_delete_cancel_{target.id}").click().run()

    after = model_service.list_models(project_id)
    assert any(m.id == target.id for m in after)
