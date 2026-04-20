"""학습 페이지(``pages/03_training.py``) AppTest 검증.

테스트 범위 (IMPLEMENTATION_PLAN §6.3 수용 기준):
- DB 미초기화 가드 / 프로젝트 미선택 가드 / 데이터셋 없을 때 안내
- 설정 폼 기본값 (타깃 첫 컬럼, 제외 컬럼 suggest_excluded 반영, 지표 옵션 = task_type 별 튜플)
- 분류/회귀 샘플 CSV 각 1회 `학습 실행` → ``LAST_TRAINING_JOB_ID`` 세팅 + 요약 렌더
- 폼 유효성: ``ValidationError`` (타깃==excluded 등) 는 service 쪽이 발생시키므로 UI 에서는 정상적인 기본값으로 실행만 확인

실제 ``run_training`` 을 도는 테스트는 scikit-learn 학습을 수반하므로 "smoke 급" 1~2건만 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import dataset_service, project_service
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "03_training.py")


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
    return AppTest.from_file(PAGE_PATH, default_timeout=90)


def _seed(csv: Path, *, name: str) -> tuple[int, int]:
    project = project_service.create_project(name)
    upload = FakeUpload(name=csv.name, data=csv.read_bytes())
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


def test_page_requires_dataset_when_none_uploaded(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("empty-train")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 데이터셋을 업로드" in warnings


# ---------------------------------------------- form defaults (classification)


def test_form_defaults_classification(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed(classification_csv, name="cls-defaults")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()
    assert not at.exception

    # 문제 유형 라디오 → 기본 classification
    task_radio = at.radio(key="training_task_type")
    assert task_radio.value == "classification"
    # 기준 지표 옵션이 분류 튜플을 따른다
    metric_select = at.selectbox(key="training_metric_key")
    assert tuple(metric_select.options) == ("accuracy", "f1", "roc_auc")


def test_form_defaults_regression_after_task_change(
    tmp_storage: Path,
    seeded_system_user: object,
    regression_csv: Path,
) -> None:
    project_id, dataset_id = _seed(regression_csv, name="reg-defaults")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()

    # task_type 을 regression 으로 전환
    at.radio(key="training_task_type").set_value("regression").run()
    metric_select = at.selectbox(key="training_metric_key")
    assert tuple(metric_select.options) == ("rmse", "mae", "r2")


# -------------------------------------------------- suggested excluded hint


def test_form_default_excludes_identifier_column(
    tmp_storage: Path,
    seeded_system_user: object,
    tmp_path: Path,
) -> None:
    """첫 컬럼이 타깃일 때, 식별자로 추정되는 컬럼이 **최초 렌더 시** 기본 제외에 포함된다.

    Streamlit 은 위젯 상태가 한 번 만들어지면 이후 `default=` 인자를 무시하므로
    target 을 **런타임에 바꿔가며** 기본 제외를 검증하는 것은 불가능하다.
    → target 을 고정한 상태의 *초회 렌더* 만 검증한다.
    """
    import pandas as pd

    project = project_service.create_project("with-id")
    # 첫 컬럼을 target 으로 두어 기본 선택이 그대로 유지되게 한다.
    df = pd.DataFrame(
        {
            "target": [i % 2 for i in range(100)],
            "id": list(range(100)),  # unique_ratio=1.0 → 식별자 힌트
            "num": [i % 5 for i in range(100)],
            "cat": [i % 3 for i in range(100)],
        }
    )
    csv = tmp_path / "with_id.csv"
    df.to_csv(csv, index=False)
    upload = FakeUpload(name=csv.name, data=csv.read_bytes())
    dto = dataset_service.upload_dataset(project.id, upload)

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dto.id
    at.run()

    excluded_ms = at.multiselect(key="training_excluded_cols")
    assert "id" in excluded_ms.value
    assert "num" not in excluded_ms.value
    assert "cat" not in excluded_ms.value


# --------------------------------------------------- happy paths (E2E train)


@pytest.mark.slow
def test_training_classification_happy_path(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, dataset_id = _seed(classification_csv, name="cls-train")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()

    # iris 계열: target 기본 첫 컬럼 → 마지막 컬럼으로 변경 필요
    target_select = at.selectbox(key="training_target_col")
    # classification.csv 의 target 컬럼명은 'species' 로 가정, 마지막 컬럼을 사용하도록 변경
    last_col = target_select.options[-1]
    target_select.set_value(last_col).run()

    submit = next(b for b in at.button if b.key == "training_submit_btn")
    submit.click().run()

    assert not at.exception
    assert SessionKey.LAST_TRAINING_JOB_ID in at.session_state
    job_id = at.session_state[SessionKey.LAST_TRAINING_JOB_ID]
    assert isinstance(job_id, int) and job_id > 0
    # 요약 카드 렌더 여부 (메트릭 위젯 존재)
    labels = [m.label for m in at.metric]
    assert "학습 시도" in labels


@pytest.mark.slow
def test_training_regression_happy_path(
    tmp_storage: Path,
    seeded_system_user: object,
    regression_csv: Path,
) -> None:
    project_id, dataset_id = _seed(regression_csv, name="reg-train")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()

    at.radio(key="training_task_type").set_value("regression").run()
    # 회귀 샘플(diabetes 계열)의 마지막 컬럼을 타깃으로
    target_select = at.selectbox(key="training_target_col")
    target_select.set_value(target_select.options[-1]).run()

    submit = next(b for b in at.button if b.key == "training_submit_btn")
    submit.click().run()

    assert not at.exception
    assert SessionKey.LAST_TRAINING_JOB_ID in at.session_state
    job_id = at.session_state[SessionKey.LAST_TRAINING_JOB_ID]
    assert isinstance(job_id, int) and job_id > 0


# ---------------------------------------------------- failure path surface


def test_training_invalid_target_surfaces_error(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service 에서 AppError 가 발생하면 카드가 error 상태 + flash error 로 표기."""
    from utils.errors import MLTrainingError

    def _boom(*args, **kwargs):
        raise MLTrainingError("학습 중 내부 오류 (테스트)")

    project_id, dataset_id = _seed(classification_csv, name="cls-fail")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    at.run()

    # 제출 직전에 service 를 패치
    monkeypatch.setattr("services.training_service.run_training", _boom)
    submit = next(b for b in at.button if b.key == "training_submit_btn")
    submit.click().run()

    # flash 는 다음 사이클에 렌더되고, `st.status` 는 error 상태 라벨을 붙인다.
    # AppTest 에서 status 위젯 라벨 검사가 애매하므로 flash 를 소비하기 위해 한번 더 rerun.
    at.run()
    errors = " ".join(e.value for e in at.error)
    assert "학습에 실패" in errors or "학습 중 내부 오류" in errors
    assert SessionKey.LAST_TRAINING_JOB_ID not in at.session_state
