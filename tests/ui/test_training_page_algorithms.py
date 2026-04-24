"""§10.5 알고리즘 선택 expander UI 검증 (pages/03_training.py, FR-067~069).

범위:
- 기본 진입 시 "전체 선택" 상태 → TrainingConfig.algorithms=None (v0.2.0 동치).
- task 전환 시 세션 잔존 이름은 자동 제거(stale 방지).
- multiselect 로 일부 해제 시 run_log 에 'algorithms=' 포함.
- CatBoost unavailable monkeypatch 시 caption 에 사유 노출 + 후보 목록에는 부재.
- ml.registry 직접 import 경계: 페이지는 training_service 만 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import dataset_service, project_service, training_service
from services.dto import AlgorithmInfoDTO
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


def _seed_csv(csv: Path, name: str) -> tuple[int, int]:
    project = project_service.create_project(name)
    upload = FakeUpload(name=csv.name, data=csv.read_bytes())
    dto = dataset_service.upload_dataset(project.id, upload)
    return project.id, dto.id


def _prepare_page(project_id: int, dataset_id: int) -> AppTest:
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    return at


def _algo_session_key(task_type: str) -> str:
    # pages/03_training.py 와 동일한 규약.
    return f"training_selected_algorithms::{task_type}"


# -------------------------------------------------------- defaults / boundary


def test_default_full_selection_is_backward_compatible(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """기본 진입 = 전체 선택 = algorithms=None 동치 (UI 내부 상태 검증)."""
    project_id, dataset_id = _seed_csv(classification_csv, "algo-defaults")
    at = _prepare_page(project_id, dataset_id).run()
    assert not at.exception

    key = _algo_session_key("classification")
    assert key in at.session_state
    selected = at.session_state[key]
    available = [i.name for i in training_service.list_algorithms("classification") if i.available]
    assert set(selected) == set(available), (selected, available)


def test_page_does_not_import_ml_registry_directly() -> None:
    """§10.4 레이어 경계: pages/03_training.py 는 ml.registry 를 import 하지 않는다."""
    with open(PAGE_PATH, encoding="utf-8") as f:
        source = f.read()
    assert "from ml.registry" not in source
    assert "import ml.registry" not in source


# ----------------------------------------------------------- task switching


def test_task_switch_drops_stale_algorithm_names(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """task 전환 시 상대 task 에 없는 이름은 자동 제거된다.

    - classification 에서 'logistic_regression' 를 남겨두고 regression 으로 전환 →
      regression 가용 목록에 없는 이름은 세션에서 제거돼야 한다.
    """
    project_id, dataset_id = _seed_csv(classification_csv, "algo-switch")
    at = _prepare_page(project_id, dataset_id).run()

    # classification 상태: 의도적으로 logistic_regression 1개만 유지 (부분 선택)
    at.multiselect(key=_algo_session_key("classification")).set_value(["logistic_regression"]).run()

    # regression 으로 전환.
    at.radio(key="training_task_type").set_value("regression").run()

    key = _algo_session_key("regression")
    assert key in at.session_state
    reg_selected = at.session_state[key]
    reg_available = [i.name for i in training_service.list_algorithms("regression") if i.available]
    # regression 에는 logistic_regression 이 없음. 그러나 regression 세션은 별도 키라 "전체"로 세팅된다.
    assert "logistic_regression" not in reg_selected
    assert set(reg_selected) == set(reg_available)


# ------------------------------------------------------ custom subset → audit


@pytest.mark.slow
def test_partial_selection_propagates_to_run_log(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """2개만 선택 후 학습 실행 → TrainingJob.run_log 에 algorithms=[...] 포함.

    또한 training.algorithms_filtered AuditLog 가 정확히 1건 기록되어야 한다.
    """
    from repositories import audit_repository, training_repository
    from repositories.base import session_scope

    project_id, dataset_id = _seed_csv(classification_csv, "algo-partial")
    at = _prepare_page(project_id, dataset_id).run()

    # 가용 중 2개만 선택 (registry core 에 항상 있는 이름)
    at.multiselect(key=_algo_session_key("classification")).set_value(
        ["logistic_regression", "random_forest"]
    ).run()

    # 학습 실행
    at.button(key="training_submit_btn").click().run()
    assert not at.exception

    assert SessionKey.LAST_TRAINING_JOB_ID in at.session_state
    job_id = at.session_state[SessionKey.LAST_TRAINING_JOB_ID]
    assert job_id is not None

    with session_scope() as session:
        job = training_repository.get(session, int(job_id))
        assert job is not None
        run_log = job.run_log or ""
        assert "algorithms=" in run_log
        assert "logistic_regression" in run_log
        assert "random_forest" in run_log

        logs = audit_repository.list_logs(session, action_type="training.algorithms_filtered")
        job_logs = [
            log for log in logs if log.target_type == "TrainingJob" and log.target_id == int(job_id)
        ]
        assert len(job_logs) == 1


# ------------------------------------ optional backend unavailable (caption)


def test_unavailable_optional_backend_shows_caption(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`list_algorithms` 결과에 catboost 가 unavailable 로 포함되면 caption 에 사유가 노출된다.

    선택 가능한 multiselect 후보 목록에는 부재해야 한다.
    """
    project_id, dataset_id = _seed_csv(classification_csv, "algo-optional")

    # list_algorithms 결과에 unavailable 1건 삽입.
    original = training_service.list_algorithms

    def _patched(task_type: str) -> list[AlgorithmInfoDTO]:
        out = list(original(task_type))
        # 이미 사용 가능한 catboost 가 있으면 건드리지 않고 더미 추가 분기.
        if any(i.name == "catboost" and i.available for i in out):
            out = [i for i in out if not (i.name == "catboost" and i.available)]
        out.append(
            AlgorithmInfoDTO(
                name="catboost",
                task_type=task_type,
                default_metric="",
                is_optional_backend=True,
                available=False,
                unavailable_reason="패키지 미설치 (pip install -r requirements-optional.txt)",
            )
        )
        return out

    monkeypatch.setattr(training_service, "list_algorithms", _patched)

    at = _prepare_page(project_id, dataset_id).run()

    # multiselect 옵션 목록에는 catboost 가 부재
    key = _algo_session_key("classification")
    selected = at.session_state[key] if key in at.session_state else []  # noqa: SIM401
    assert "catboost" not in selected

    # caption 문자열 어딘가에 사유 문구가 들어있어야 함.
    caption_texts = [c.value for c in at.caption if c.value]
    assert any("catboost" in c and "pip install" in c for c in caption_texts), caption_texts
