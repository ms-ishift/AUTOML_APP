"""Service 레이어 end-to-end 왕복 통합 테스트 (IMPLEMENTATION_PLAN §4.6).

목적:
- 개별 Service 테스트에서는 각자의 경계만 검증했다. 이 테스트는 사용자 관점의 **전체 유스케이스
  왕복** (프로젝트 생성 → 데이터셋 업로드 → 학습 → 모델 핀 → 단건/배치 예측 → 삭제) 을
  단일 시나리오로 묶어 리그레션을 조기 탐지한다.
- UI(Streamlit) 없이 Service 함수만으로 시나리오가 완결되는지 (``.cursor/rules/service-layer.mdc``)
  도 함께 증빙한다.

시나리오:
1. ``project_service.create_project`` 로 프로젝트 생성
2. ``dataset_service.upload_dataset`` 로 iris CSV 업로드 (duck-typing ``FakeUpload``)
3. ``training_service.run_training`` 으로 분류 학습 (on_progress 호출 카운트까지 확인)
4. ``model_service.list_models`` / ``get_model_detail`` 로 결과 조회
5. ``model_service.save_model`` 으로 is_best 를 다른 모델로 수동 전환
6. ``prediction_service.predict_single`` 로 단건 예측 + 확률 합≈1
7. ``prediction_service.predict_batch`` 로 파일 예측 → CSV 저장 확인
8. ``model_service.delete_model`` 로 관련 파일/레코드 정리
9. ``project_service.delete_project`` 로 cascade 삭제 — 남은 데이터셋이 있으므로 가드 발동 (Project Service 의 cascade 거부 규약 재확인)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from ml.schemas import TrainingConfig
from repositories import (
    audit_repository,
    model_repository,
    prediction_repository,
)
from repositories.base import session_scope
from services import (
    dataset_service,
    model_service,
    prediction_service,
    project_service,
    training_service,
)
from services.prediction_service import PREDICTION_COLUMN, PROBABILITY_PREFIX
from utils.errors import ValidationError


@dataclass
class FakeUpload:
    """Streamlit ``UploadedFile`` 호환 duck-typing 객체."""

    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def test_full_training_and_prediction_roundtrip(
    classification_csv: Path,
    tmp_storage: Path,
    tmp_path: Path,
    seeded_system_user: object,
) -> None:
    # --- 1) 프로젝트 생성 -----------------------------------------------
    project = project_service.create_project("E2E 프로젝트", description="end-to-end roundtrip")
    assert project.id > 0

    # --- 2) 데이터셋 업로드 ---------------------------------------------
    upload = FakeUpload(name=classification_csv.name, data=classification_csv.read_bytes())
    dataset = dataset_service.upload_dataset(project.id, upload)
    assert dataset.project_id == project.id
    assert dataset.row_count > 0 and dataset.column_count > 0
    # DTO 는 file_path 를 노출하지 않으므로 repository 경유로 간접 확인.
    from repositories import dataset_repository

    with session_scope() as session:
        dataset_entity = dataset_repository.get(session, dataset.id)
        assert dataset_entity is not None
        saved_csv_path = Path(dataset_entity.file_path)
    assert saved_csv_path.exists()
    assert saved_csv_path.is_relative_to(tmp_storage / "datasets")

    # --- 3) 학습 실행 + 진행 콜백 ---------------------------------------
    progress_events: list[tuple[str, float]] = []

    training_result = training_service.run_training(
        TrainingConfig(
            dataset_id=dataset.id,
            task_type="classification",
            target_column="species",
        ),
        on_progress=lambda stage, ratio: progress_events.append((stage, ratio)),
    )
    assert training_result.best_algo is not None
    assert len(training_result.rows) >= 2
    # DTO 는 job status 를 직접 노출하지 않으므로 repository 로 확인.
    from repositories import training_repository

    with session_scope() as session:
        job_entity = training_repository.get(session, training_result.job_id)
        assert job_entity is not None
        assert job_entity.status == "completed"
    # 진행 콜백은 최소 preprocessing / split / train / score / save / completed 6단계 이상
    stages = {e[0] for e in progress_events}
    assert {"preprocessing", "split", "score", "save", "completed"}.issubset(stages)
    # ratio 는 단조 증가 (동률 허용)
    ratios = [e[1] for e in progress_events]
    assert ratios == sorted(ratios)
    assert ratios[-1] == pytest.approx(1.0)

    # --- 4) 모델 목록/상세 ----------------------------------------------
    models = model_service.list_models(project.id)
    success_models = [m for m in models if m.metric_score is not None]
    assert len(success_models) >= 2, "pin 테스트를 위해 최소 2개의 성공 모델 필요"

    best_initial = next(m for m in success_models if m.is_best)
    detail = model_service.get_model_detail(best_initial.id)
    assert detail.feature_schema.target == "species"
    assert detail.feature_schema.numeric
    assert detail.metrics_summary  # 분류 → accuracy/f1 등 최소 1개

    # --- 5) 다른 모델을 수동 저장 (is_best 전환) -------------------------
    non_best = next(m for m in success_models if not m.is_best)
    promoted = model_service.save_model(non_best.id)
    assert promoted.is_best is True

    with session_scope() as session:
        job_models = list(model_repository.list_by_training_job(session, training_result.job_id))
        best_rows = [m for m in job_models if m.is_best]
        assert len(best_rows) == 1
        assert best_rows[0].model_id == non_best.id

    # --- 6) 단건 예측 -----------------------------------------------------
    df = pd.read_csv(classification_csv)
    single_payload = df.iloc[0].drop("species").to_dict()
    single_result = prediction_service.predict_single(promoted.id, single_payload)
    assert len(single_result.rows) == 1
    single_row = single_result.rows[0]
    assert PREDICTION_COLUMN in single_row
    prob_keys = [k for k in single_row if k.startswith(PROBABILITY_PREFIX)]
    assert len(prob_keys) == 3  # iris 3 클래스
    assert abs(sum(single_row[k] for k in prob_keys) - 1.0) < 1e-6
    assert single_result.result_path is None

    # --- 7) 배치 예측 -----------------------------------------------------
    batch_input = tmp_path / "batch_e2e.csv"
    df.drop(columns=["species"]).to_csv(batch_input, index=False)
    batch_result = prediction_service.predict_batch(promoted.id, batch_input)
    assert batch_result.result_path is not None
    result_csv = Path(batch_result.result_path)
    assert result_csv.exists()
    assert result_csv.parent == tmp_storage / "predictions"

    # 저장된 CSV 검증: 입력 행 수와 동일, prediction 컬럼 포함
    stored = pd.read_csv(result_csv)
    assert len(stored) == len(df)
    assert PREDICTION_COLUMN in stored.columns

    # PredictionJob 기록 (단건 + 배치) = 2건, 모두 completed
    with session_scope() as session:
        jobs = list(prediction_repository.list_by_model(session, promoted.id))
        assert len(jobs) == 2
        assert {j.status for j in jobs} == {"completed"}
        assert {j.input_type for j in jobs} == {"form", "file"}

    # --- 8) 모델 삭제 (파일/레코드 cleanup) ------------------------------
    to_delete = non_best.id
    model_dir = tmp_storage / "models" / str(to_delete)
    assert model_dir.exists()
    model_service.delete_model(to_delete)
    assert not model_dir.exists()
    with session_scope() as session:
        assert model_repository.get(session, to_delete) is None
        # cascade 로 연결된 PredictionJob 들도 함께 삭제됨
        remaining_jobs = list(prediction_repository.list_by_model(session, to_delete))
        assert remaining_jobs == []
        # 배치 결과 CSV 도 best-effort 로 정리됨
        assert not result_csv.exists()

    # --- 9) 프로젝트 cascade 가드 -----------------------------------------
    # 데이터셋이 남아 있으므로 프로젝트 직접 삭제는 Project Service 규약상 차단.
    with pytest.raises(ValidationError):
        project_service.delete_project(project.id)

    # 데이터셋부터 삭제 → 학습 잡/모델 cascade → 프로젝트 삭제 가능
    dataset_service.delete_dataset(dataset.id)
    project_service.delete_project(project.id)

    with session_scope() as session:
        audits = audit_repository.list_logs(session)
        action_types = {a.action_type for a in audits}
    # 해당 시나리오에서 발생한 핵심 이벤트가 감사 로그에 전부 남아야 한다.
    assert {
        "project.created",
        "dataset.uploaded",
        "training.completed",
        "model.saved",
        "prediction.started",
        "prediction.completed",
        "model.deleted",
        "dataset.deleted",
        "project.deleted",
    }.issubset(action_types)
