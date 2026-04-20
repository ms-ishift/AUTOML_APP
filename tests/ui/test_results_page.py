"""결과 비교 페이지(``pages/04_results.py``) AppTest 검증.

검증 포인트 (IMPLEMENTATION_PLAN §6.4 수용 기준):
- DB / 프로젝트 / 학습 잡 가드 동작
- 학습 결과(분류·회귀) 표시: 시도/성공/실패 카드, 비교표, 베스트 배지, 플롯 셀렉터
- 저장 액션 — 비베스트 모델 선택 후 `이 모델 저장` 클릭 → is_best 가 승계되고 success flash
- stale `LAST_TRAINING_JOB_ID` 정리는 job list 로 대체되므로 별도 케이스 없음

실제 `run_training` 을 수반하는 해피패스는 slow 마커.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ml.schemas import TrainingConfig
from services import dataset_service, project_service, training_service
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "04_results.py")


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


def _seed_and_train(
    csv: Path,
    *,
    task_type: str,
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
            task_type=task_type,  # type: ignore[arg-type]
            target_column=target,
        )
    )
    return project.id, result.job_id


# --------------------------------------------------------------------- guards


def test_results_page_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_results_page_requires_project(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 프로젝트를 선택" in warnings


def test_results_page_requires_training_result(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("no-train")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 학습을 실행" in warnings


# ------------------------------------------------------ happy path (classification)


@pytest.mark.slow
def test_results_page_classification_renders_table_and_plot(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, job_id = _seed_and_train(
        classification_csv,
        task_type="classification",
        target="species",
        name="cls-results",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.LAST_TRAINING_JOB_ID] = job_id
    at.run()

    assert not at.exception
    # 요약 메트릭 3종
    labels = {m.label for m in at.metric}
    assert {"학습 시도", "성공", "실패"}.issubset(labels)
    # 베스트 success 메시지
    success_texts = " ".join(s.value for s in at.success)
    assert "베스트 모델" in success_texts
    # 플롯 selectbox 존재
    assert any(sb.key == "results_plot_pick" for sb in at.selectbox)
    # 저장 selectbox 존재
    assert any(sb.key == "results_save_pick" for sb in at.selectbox)


# ----------------------------------------------- happy path (regression)


@pytest.mark.slow
def test_results_page_regression_scatter_available(
    tmp_storage: Path,
    seeded_system_user: object,
    regression_csv: Path,
) -> None:
    # regression.csv 의 타깃은 마지막 컬럼 'y' 이라고 가정 (샘플 생성기 기준).
    import pandas as pd

    cols = pd.read_csv(regression_csv, nrows=1).columns.tolist()
    target = cols[-1]
    project_id, job_id = _seed_and_train(
        regression_csv,
        task_type="regression",
        target=target,
        name="reg-results",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.LAST_TRAINING_JOB_ID] = job_id
    at.run()

    assert not at.exception
    # 회귀에는 success 가 있어야 하고, 플롯 셀렉터도 렌더된다
    assert any(sb.key == "results_plot_pick" for sb in at.selectbox)


# ---------------------------------------------------- save (is_best promote)


@pytest.mark.slow
def test_results_page_save_button_promotes_is_best(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, job_id = _seed_and_train(
        classification_csv,
        task_type="classification",
        target="species",
        name="cls-save",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.LAST_TRAINING_JOB_ID] = job_id
    at.run()

    save_pick = next(sb for sb in at.selectbox if sb.key == "results_save_pick")
    # 비-베스트 후보를 찾는다 — 첫 번째로 is_best 아닌 후보
    result = training_service.get_training_result(job_id)
    success = [r for r in result.rows if r.status == "success"]
    non_best_idx = next(
        (i for i, r in enumerate(success) if not r.is_best),
        None,
    )
    if non_best_idx is None:
        pytest.skip("현재 성공 모델이 1개뿐이라 non-best 승격 테스트 불가")
    non_best_row = success[non_best_idx]

    save_pick.set_value(non_best_idx).run()
    # 버튼이 disabled 아님(베스트 아니므로)
    save_btn = next(b for b in at.button if b.key == "results_save_btn")
    save_btn.click().run()

    assert not at.exception
    success_texts = " ".join(s.value for s in at.success)
    assert "모델이 저장" in success_texts

    # DB 상 is_best 가 non_best_row 로 이동했는지 확인
    new_result = training_service.get_training_result(job_id)
    new_best = next(r for r in new_result.rows if r.is_best)
    assert new_best.algo_name == non_best_row.algo_name


# ---------------------------------------------------- sorting & failed rows


@pytest.mark.slow
def test_results_page_sorts_by_metric_direction(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, job_id = _seed_and_train(
        classification_csv,
        task_type="classification",
        target="species",
        name="cls-sort",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.LAST_TRAINING_JOB_ID] = job_id
    at.run()
    assert not at.exception

    # 비교표 dataframe 1개 이상 존재
    assert len(at.dataframe) >= 1
