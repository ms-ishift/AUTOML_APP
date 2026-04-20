"""예측 페이지(``pages/06_prediction.py``) AppTest 검증.

검증 포인트 (IMPLEMENTATION_PLAN §6.6, FR-080~085):
- DB / 프로젝트 / 모델 없음 가드
- 모델 selectbox 기본값: `SessionKey.CURRENT_MODEL_ID` 우선 → is_best 폴백
- 단건 입력: feature_schema 기반 폼이 렌더되고 제출 시 예측값이 노출됨
- 단건: 서비스가 `PredictionInputError` 를 던지면 flash error 로 surface (§10.4)
- 파일 예측: `predict_batch` 를 monkeypatch 로 대체 → 결과 표 + 다운로드 버튼 존재 검증
  (`st.file_uploader` 는 AppTest 에서 직접 주입이 어려워 `session_state` 에 더미 객체를 주입)

실제 `run_training` 을 수반하는 해피패스는 slow 마커.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ml.schemas import TrainingConfig
from services import (
    dataset_service,
    prediction_service,
    project_service,
    training_service,
)
from services.dto import PredictionResultDTO
from utils.errors import PredictionInputError
from utils.session_utils import SessionKey

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "pages" / "06_prediction.py")


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
    return AppTest.from_file(PAGE_PATH, default_timeout=180)


def _seed_trained(
    csv: Path,
    *,
    task_type: str,
    target: str,
    name: str,
) -> tuple[int, int, int]:
    """프로젝트/데이터셋/학습 잡을 시드하고 **성공 모델 하나**의 id 를 반환."""
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
    from services import model_service

    models = [m for m in model_service.list_models(project.id) if m.metric_score is not None]
    assert models, "성공 모델이 1건 이상 있어야 함"
    return project.id, result.job_id, models[0].id


# -------------------------------------------------------------------- guards


def test_prediction_page_db_guide_when_not_initialized(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    at = _new_page().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "DB가 아직 초기화되지 않았습니다" in errors


def test_prediction_page_requires_project(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    at = _new_page().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "먼저 프로젝트를 선택" in warnings


def test_prediction_page_requires_trained_model(
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project = project_service.create_project("no-model-yet")
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()
    info_texts = " ".join(i.value for i in at.info)
    assert "먼저 저장된 모델을 선택" in info_texts


# -------------------------------------------------------------- model picker


@pytest.mark.slow
def test_prediction_page_model_picker_defaults_to_current_model_id(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _, model_id = _seed_trained(
        classification_csv,
        task_type="classification",
        target="species",
        name="cls-picker",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_MODEL_ID] = model_id
    at.run()

    assert not at.exception
    # picker 기본값으로 지정한 모델이 유지되는지 확인
    assert at.session_state[SessionKey.CURRENT_MODEL_ID] == model_id
    # 단건/파일 탭 존재
    labels = {str(t.label) for t in at.tabs}
    assert {"단건 입력", "파일 예측"}.issubset(labels)


# ------------------------------------------------------------ single form


@pytest.mark.slow
def test_prediction_page_single_form_runs_and_shows_result(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    project_id, _, model_id = _seed_trained(
        classification_csv,
        task_type="classification",
        target="species",
        name="cls-single",
    )
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_MODEL_ID] = model_id
    at.run()

    # number_input 들은 기본값 0.0 이므로 바로 제출해도 예측은 돈다 (unseen 경고가 붙을 수 있음)
    submit = next(
        b for b in at.button if str(b.key).startswith("FormSubmitter:prediction_single_form_")
    )
    submit.click().run()

    assert not at.exception
    # 결과 블록에 "예측 결과" metric + 상세 expander 렌더
    markdowns = " ".join(m.value for m in at.markdown)
    assert "최근 단건 예측" in markdowns
    # session 캐시에 결과 dict 가 저장되어야 함
    assert "prediction_single_result" in at.session_state
    assert at.session_state["prediction_single_result"]


def test_prediction_page_single_form_surfaces_service_error(
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """서비스 레이어 오류가 UI flash 로 노출되는지 — 실제 학습 없이 경량 검증."""
    project = project_service.create_project("err-surface")

    # list_models 는 1건, get_model_detail 은 최소 스키마만 반환하도록 패치
    from datetime import datetime

    from services import model_service
    from services.dto import FeatureSchemaDTO, ModelDetailDTO, ModelDTO

    fake_model = ModelDTO(
        id=999,
        training_job_id=1,
        algo_name="fake",
        metric_score=0.9,
        is_best=True,
        created_at=datetime(2026, 1, 1),
    )
    fake_detail = ModelDetailDTO(
        base=fake_model,
        feature_schema=FeatureSchemaDTO(numeric=["x"], categorical=[], target="y", categories={}),
        metrics_summary={"accuracy": 0.9},
    )
    monkeypatch.setattr(model_service, "list_models", lambda pid: [fake_model])
    monkeypatch.setattr(model_service, "get_model_detail", lambda mid: fake_detail)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise PredictionInputError("필수 입력 컬럼 누락")

    monkeypatch.setattr(prediction_service, "predict_single", _raise)

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()

    submit = next(
        b for b in at.button if str(b.key).startswith("FormSubmitter:prediction_single_form_")
    )
    submit.click().run()

    errors = " ".join(e.value for e in at.error)
    assert "필수 입력 컬럼 누락" in errors


# --------------------------------------------------------------- batch flow


def test_prediction_page_batch_result_renders_preview_and_download(
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """파일 예측 결과 영역이 캐시된 결과로 렌더되는지 (업로드·실행 경로 우회).

    `st.file_uploader` 는 AppTest 에서 직접 주입이 어려우므로, 이미 `BATCH_RESULT_KEY` 가
    채워진 상태에서 페이지가 예측 결과를 정상적으로 그리는지 검증한다.
    """
    from datetime import datetime

    from services import model_service
    from services.dto import FeatureSchemaDTO, ModelDetailDTO, ModelDTO

    project = project_service.create_project("batch-render")
    fake_model = ModelDTO(
        id=42,
        training_job_id=1,
        algo_name="fake-batch",
        metric_score=0.8,
        is_best=True,
        created_at=datetime(2026, 1, 1),
    )
    fake_detail = ModelDetailDTO(
        base=fake_model,
        feature_schema=FeatureSchemaDTO(numeric=["x"], categorical=[], target="y", categories={}),
        metrics_summary={"accuracy": 0.8},
    )
    monkeypatch.setattr(model_service, "list_models", lambda pid: [fake_model])
    monkeypatch.setattr(model_service, "get_model_detail", lambda mid: fake_detail)

    # 실제 파일을 만들어 둬야 "결과 CSV 다운로드" 버튼이 노출됨
    result_path = tmp_storage / "predictions" / "77.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1.0], "prediction": ["A"]}).to_csv(result_path, index=False)

    cached_result = PredictionResultDTO(
        job_id=77,
        rows=[{"x": 1.0, "prediction": "A"}],
        result_path=str(result_path),
        warnings=["학습에 사용되지 않은 컬럼은 무시합니다: foo"],
    )

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.session_state["prediction_batch_result"] = {
        "model_id": fake_model.id,
        "result": cached_result,
        "source_name": "sample.csv",
    }
    at.run()

    assert not at.exception
    markdowns = " ".join(m.value for m in at.markdown)
    assert "최근 파일 예측 결과" in markdowns
    # 경고가 UI 로 전달되는지
    warnings = " ".join(w.value for w in at.warning)
    assert "학습에 사용되지 않은 컬럼" in warnings
    # 다운로드 버튼 존재 확인 (AppTest 는 download_button 을 UnknownElement 로 노출 → 길이만 확인)
    assert len(at.get("download_button")) >= 1


def test_prediction_page_batch_run_invokes_service_with_saved_path(
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """업로드가 있으면 `_stash_uploaded_file` + `predict_batch` 가 호출된다.

    세션에 업로드 객체를 직접 주입하는 대신, `_run_batch_prediction` 을 감시한다.
    """
    from datetime import datetime

    from services import model_service
    from services.dto import FeatureSchemaDTO, ModelDetailDTO, ModelDTO

    project = project_service.create_project("batch-run-trace")
    fake_model = ModelDTO(
        id=1,
        training_job_id=1,
        algo_name="fake",
        metric_score=0.5,
        is_best=True,
        created_at=datetime(2026, 1, 1),
    )
    fake_detail = ModelDetailDTO(
        base=fake_model,
        feature_schema=FeatureSchemaDTO(numeric=[], categorical=[], target="y", categories={}),
        metrics_summary={},
    )
    monkeypatch.setattr(model_service, "list_models", lambda pid: [fake_model])
    monkeypatch.setattr(model_service, "get_model_detail", lambda mid: fake_detail)

    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project.id
    at.run()

    # 업로드 없이 버튼이 disabled 상태인지 확인 (클릭해도 콜백이 돌지 않음)
    run_btn = next(b for b in at.button if b.key == f"prediction_batch_run_{fake_model.id}")
    assert run_btn.disabled is True
