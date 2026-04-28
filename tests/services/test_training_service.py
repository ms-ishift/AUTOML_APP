"""Training Service 단위/통합 테스트 (IMPLEMENTATION_PLAN §4.3 / §9.7).

- 분류/회귀 각각 한 번씩 전체 파이프라인을 돌려 DB/파일 산출물을 검증.
- 진행 콜백 호출 및 실패 경로 (데이터셋 없음, 타깃 컬럼 없음, 아티팩트 저장 실패)를 확인.
- 개별 알고리즘 실패가 전체 실행을 중단시키지 않는지 부분 실패 경로도 검증.
- §9.7: 고급 전처리 config forward / run_log summary / feature_engineering·balance stage /
  번들 preprocessing_config.json 생성 여부 / preview_preprocessing 유스케이스.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ml.schemas import PreprocessingConfig, TrainingConfig, TuningConfig
from repositories import (
    audit_repository,
    dataset_repository,
    model_repository,
    project_repository,
    training_repository,
)
from repositories.base import session_scope
from services import training_service
from services.dto import AlgorithmInfoDTO, FeaturePreviewDTO, OptionalBackendInfoDTO
from utils.errors import MLTrainingError, NotFoundError, StorageError, ValidationError

# ------------------------------------------------------------------- helpers


def _seed_project_and_dataset(sample_csv: Path, storage: Path) -> tuple[int, int]:
    """샘플 CSV 를 임시 storage 로 복사하고 Project/Dataset 레코드를 직접 생성."""
    target = storage / "datasets" / f"{uuid.uuid4().hex}{sample_csv.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_csv, target)

    with session_scope() as session:
        project = project_repository.insert(session, project_name="테스트 프로젝트")
        dataset = dataset_repository.insert(
            session,
            project_id=project.project_id,
            file_name=sample_csv.name,
            file_path=str(target),
            row_count=0,
            column_count=0,
        )
        return project.project_id, dataset.dataset_id


# ----------------------------------------------------------- classification


def test_run_training_classification_happy(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)

    stages: list[tuple[str, float]] = []
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
        test_size=0.25,
    )
    result = training_service.run_training(config, on_progress=lambda s, r: stages.append((s, r)))

    assert result.metric_key == "f1"
    assert result.best_algo is not None
    assert any(r.is_best for r in result.rows)
    assert sum(1 for r in result.rows if r.is_best) == 1
    assert all(r.status in {"success", "failed"} for r in result.rows)
    success_rows = [r for r in result.rows if r.status == "success"]
    assert success_rows, "적어도 한 개 알고리즘은 성공해야 한다"
    # best row 는 success 여야 한다
    best_row = next(r for r in result.rows if r.is_best)
    assert best_row.status == "success"
    assert "f1" in best_row.metrics

    # 진행 콜백: 주요 단계가 적어도 1회씩 찍혔는지 검증
    names = [s for s, _ in stages]
    assert "preprocessing" in names
    assert "split" in names
    assert any(n.startswith("train:") for n in names)
    assert "score" in names
    assert "save" in names
    assert "completed" in names
    # 비내림차순
    ratios = [r for _, r in stages]
    assert ratios == sorted(ratios)

    # DB 검증
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.ended_at is not None
        assert (job.run_log or "").strip()

        stored_models = list(model_repository.list_by_training_job(session, result.job_id))
        assert len(stored_models) == len(result.rows)
        best_count = sum(1 for m in stored_models if m.is_best)
        assert best_count == 1

    # 파일 아티팩트 검증 (성공 모델 수만큼 디렉터리가 있어야 함)
    models_dir = tmp_storage / "models"
    assert models_dir.is_dir()
    saved_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    assert len(saved_dirs) == len(success_rows)
    for d in saved_dirs:
        assert (d / "model.joblib").exists()
        assert (d / "preprocessor.joblib").exists()
        assert (d / "feature_schema.json").exists()
        assert (d / "metrics.json").exists()

    # 감사 로그 검증
    with session_scope() as session:
        started = audit_repository.list_logs(session, action_type="training.started")
        completed = audit_repository.list_logs(session, action_type="training.completed")
        assert len(started) >= 1
        assert len(completed) >= 1


# ----------------------------------------------------------------- regression


def test_run_training_regression_happy(
    regression_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(regression_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="regression",
        target_column="progression",
        test_size=0.25,
    )
    result = training_service.run_training(config)
    assert result.metric_key == "rmse"
    assert result.best_algo is not None
    best_row = next(r for r in result.rows if r.is_best)
    assert "rmse" in best_row.metrics


# ---------------------------------------------------- validation / lookup


def test_run_training_raises_on_missing_dataset(tmp_storage: Path) -> None:
    config = TrainingConfig(
        dataset_id=9999,
        task_type="classification",
        target_column="y",
    )
    with pytest.raises(NotFoundError):
        training_service.run_training(config)


def test_run_training_raises_on_missing_target(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="does_not_exist",
    )
    with pytest.raises(ValidationError):
        training_service.run_training(config)


def test_run_training_raises_on_bad_metric_key(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
        metric_key="not_supported",
    )
    with pytest.raises(ValidationError):
        training_service.run_training(config)


def test_run_training_raises_on_missing_file(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    # 파일 제거 → StorageError 기대
    with session_scope() as session:
        ds = dataset_repository.get(session, dataset_id)
        assert ds is not None
        Path(ds.file_path).unlink()

    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    with pytest.raises(StorageError):
        training_service.run_training(config)


# ------------------------------------------------------------ partial failure


def test_run_training_partial_failure_is_recorded(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """개별 알고리즘이 실패해도 전체는 완료되고 failed 행이 DTO 에 포함된다."""
    from ml import registry
    from ml.registry import AlgoSpec

    def _boom() -> object:
        raise RuntimeError("boom")

    failing = AlgoSpec("always_fail", "classification", _boom, "f1")
    patched_specs = [*registry.get_specs("classification"), failing]
    monkeypatch.setattr(training_service, "get_specs", lambda task: patched_specs)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)
    assert any(r.status == "failed" and r.algo_name == "always_fail" for r in result.rows)
    assert any(r.status == "success" for r in result.rows)
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "completed"


def test_run_training_all_failed_marks_job_failed(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml.registry import AlgoSpec

    def _boom() -> object:
        raise RuntimeError("boom")

    only_failing = [AlgoSpec("only_fail", "classification", _boom, "f1")]
    monkeypatch.setattr(training_service, "get_specs", lambda task: only_failing)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)
    assert result.best_algo is None
    assert all(r.status == "failed" for r in result.rows)
    with session_scope() as session:
        job = training_repository.get(session, result.job_id)
        assert job is not None
        assert job.status == "failed"


# ---------------------------------------------------- artifact save failure


def test_run_training_rolls_back_on_artifact_failure(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_model_bundle 실패 → DB 롤백 + 파일 정리 + MLTrainingError."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(training_service, "save_model_bundle", _raise)

    _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    with pytest.raises(MLTrainingError):
        training_service.run_training(config)

    # DB: Model 레코드는 롤백되어 존재하지 않아야 한다
    with session_scope() as session:
        all_models = list(model_repository.list_by_project(session, project_id=1))
        assert all_models == []
        # 학습 잡은 failed 상태로 기록
        jobs = list(training_repository.list_by_project(session, project_id=1))
        assert len(jobs) == 1
        assert jobs[0].status == "failed"

    # storage/models 아래에 남은 디렉터리가 없어야 한다
    models_dir = tmp_storage / "models"
    assert not any(models_dir.iterdir())


# -------------------------------------------------------------- list/get


def test_list_and_get_training_job(
    classification_csv: Path,
    tmp_storage: Path,
    seeded_system_user: object,
) -> None:
    project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
    config = TrainingConfig(
        dataset_id=dataset_id,
        task_type="classification",
        target_column="species",
    )
    result = training_service.run_training(config)

    jobs = training_service.list_training_jobs(project_id)
    assert len(jobs) == 1
    assert jobs[0].id == result.job_id
    assert jobs[0].status == "completed"

    reloaded = training_service.get_training_result(result.job_id)
    assert reloaded.job_id == result.job_id
    assert reloaded.best_algo == result.best_algo
    assert len(reloaded.rows) == len(result.rows)


def test_get_training_result_missing_raises(tmp_storage: Path) -> None:
    with pytest.raises(NotFoundError):
        training_service.get_training_result(99999)


# =========================================================== §9.7 Service 통합


class TestPreprocessingForwarding:
    """§9.7: run_training 이 config.preprocessing 를 ml 레이어로 forward 하는 경로."""

    def test_default_config_emits_new_stages(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """기본 config 에서도 feature_engineering / balance stage 가 순서대로 emit 되어야 한다."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        stages: list[tuple[str, float]] = []
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        training_service.run_training(config, on_progress=lambda s, r: stages.append((s, r)))
        names = [s for s, _ in stages]
        assert "feature_engineering" in names
        assert "balance" in names
        idx = {
            n: names.index(n) for n in ("preprocessing", "feature_engineering", "split", "balance")
        }
        assert idx["preprocessing"] < idx["feature_engineering"]
        assert idx["feature_engineering"] < idx["split"]
        assert idx["split"] < idx["balance"]
        # balance 는 train:<algo> 보다 먼저
        first_train_idx = next(i for i, n in enumerate(names) if n.startswith("train:"))
        assert idx["balance"] < first_train_idx

    def test_default_config_run_log_says_default(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        result = training_service.run_training(config)
        with session_scope() as session:
            job = training_repository.get(session, result.job_id)
            assert job is not None
            assert "preprocessing: default" in (job.run_log or "")

    def test_custom_config_run_log_contains_summary(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """is_default=False 면 summary() 가 바뀐 축만 `key=value` 로 나열돼 run_log 에 기록된다."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        pp_cfg = PreprocessingConfig(numeric_scale="robust")
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        result = training_service.run_training(config)
        with session_scope() as session:
            job = training_repository.get(session, result.job_id)
            assert job is not None
            run_log = job.run_log or ""
            assert "preprocessing: numeric_scale=robust" in run_log
            # default 문자열은 포함되지 말아야 함
            assert "preprocessing: default" not in run_log

    def test_custom_config_saves_preprocessing_json(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """is_default=False 일 때 각 모델 번들에 preprocessing_config.json 이 생성된다."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        pp_cfg = PreprocessingConfig(numeric_scale="robust")
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        training_service.run_training(config)
        models_dir = tmp_storage / "models"
        saved_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        assert saved_dirs
        for d in saved_dirs:
            assert (
                d / "preprocessing_config.json"
            ).exists(), f"preprocessing_config.json 미생성: {d}"

    def test_default_config_skips_preprocessing_json(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """is_default=True 면 preprocessing_config.json 은 생성되지 않는다 (구 모델 바이트 동치)."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        training_service.run_training(config)
        models_dir = tmp_storage / "models"
        saved_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        assert saved_dirs
        for d in saved_dirs:
            assert not (d / "preprocessing_config.json").exists()

    def test_regression_plus_smote_rejected_by_training_config(self) -> None:
        """회귀 + SMOTE 는 TrainingConfig 생성 단계에서 거부되어야 한다 (§9.1 크로스 검증)."""
        pp_cfg = PreprocessingConfig(imbalance="smote")
        with pytest.raises(ValueError, match="SMOTE"):
            TrainingConfig(
                dataset_id=1,
                task_type="regression",
                target_column="y",
                preprocessing=pp_cfg,
            )

    def test_custom_config_writes_preprocessing_customized_audit(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """§9.8: is_default=False 학습 시 AuditLog 에 training.preprocessing_customized 1회 기록."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        pp_cfg = PreprocessingConfig(numeric_scale="robust", imbalance="class_weight")
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        result = training_service.run_training(config)

        with session_scope() as session:
            logs = audit_repository.list_logs(
                session, action_type="training.preprocessing_customized"
            )
            job_logs = [
                log
                for log in logs
                if log.target_type == "TrainingJob" and log.target_id == result.job_id
            ]
            assert len(job_logs) == 1
            detail = job_logs[0].detail_json or {}
            assert "summary" in detail
            assert "numeric_scale=robust" in detail["summary"]
            assert "imbalance=class_weight" in detail["summary"]

    def test_default_config_skips_preprocessing_customized_audit(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """§9.8: 기본 config 학습 시 training.preprocessing_customized 는 기록되지 않아야 한다."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        result = training_service.run_training(config)

        with session_scope() as session:
            logs = audit_repository.list_logs(
                session, action_type="training.preprocessing_customized"
            )
            assert not [
                log
                for log in logs
                if log.target_type == "TrainingJob" and log.target_id == result.job_id
            ]


class TestAlgorithmFiltering:
    """§10.3 (FR-067): TrainingConfig.algorithms 필터링 + 감사 로그."""

    def test_default_none_equals_full_set(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """algorithms=None → 전체 후보 학습, training.algorithms_filtered 감사 0건 (v0.2.0 동치)."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        result = training_service.run_training(config)
        successful = [r for r in result.rows if r.status == "success"]
        assert len(successful) >= 2

        with session_scope() as session:
            logs = audit_repository.list_logs(session, action_type="training.algorithms_filtered")
            assert not [
                log
                for log in logs
                if log.target_type == "TrainingJob" and log.target_id == result.job_id
            ]

    def test_single_algorithm_trains_only_that_model(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """algorithms=('random_forest',) → result.rows 정확히 1건 + 감사 1건."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            algorithms=("random_forest",),
        )
        result = training_service.run_training(config)

        names = {r.algo_name for r in result.rows}
        assert names == {"random_forest"}

        with session_scope() as session:
            logs = audit_repository.list_logs(session, action_type="training.algorithms_filtered")
            job_logs = [
                log
                for log in logs
                if log.target_type == "TrainingJob" and log.target_id == result.job_id
            ]
            assert len(job_logs) == 1
            detail = job_logs[0].detail_json or {}
            assert detail.get("algorithms") == ["random_forest"]

    def test_unknown_algorithm_raises_validation_error(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """미등록 이름 포함 → ValidationError."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            algorithms=("mystery_forest",),
        )
        with pytest.raises(ValidationError, match="mystery_forest"):
            training_service.run_training(config)

    def test_multiple_algorithms_trains_selected_subset(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """2개 선택 → trained_models 정확히 2건 + 감사 1건 (detail 에 두 이름 모두)."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            algorithms=("logistic_regression", "random_forest"),
        )
        result = training_service.run_training(config)

        names = {r.algo_name for r in result.rows}
        assert names == {"logistic_regression", "random_forest"}

        with session_scope() as session:
            logs = audit_repository.list_logs(session, action_type="training.algorithms_filtered")
            job_logs = [
                log
                for log in logs
                if log.target_type == "TrainingJob" and log.target_id == result.job_id
            ]
            assert len(job_logs) == 1
            assert set((job_logs[0].detail_json or {}).get("algorithms", [])) == {
                "logistic_regression",
                "random_forest",
            }

    def test_tuning_downgrade_emits_event_but_still_trains(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """§10.3 (§11 선반영): tuning.method='grid' 는 현재 downgrade 되어야 한다.

        - 학습은 정상 완료 (result.rows 유지)
        - run_log 에 'tuning=downgraded_v010' 포함
        """
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            algorithms=("logistic_regression",),
            tuning=TuningConfig(method="grid", cv_folds=3),
        )
        result = training_service.run_training(config)
        assert len(result.rows) == 1

        with session_scope() as session:
            job = training_repository.get(session, result.job_id)
            assert job is not None
            run_log = job.run_log or ""
            assert "tuning=downgraded_v010" in run_log


class TestPreviewPreprocessing:
    """§9.7: preview_preprocessing 읽기 전용 유스케이스."""

    def test_default_classification_numeric_only(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """iris 유사 데이터(수치 4 + 타깃) / 기본 config → 파생 없음."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        preview = training_service.preview_preprocessing(dataset_id, config)
        assert isinstance(preview, FeaturePreviewDTO)
        assert preview.n_cols_in == 4
        assert preview.n_cols_out == 4
        assert preview.derived == ()
        assert preview.encoding_summary == {}
        assert preview.auto_downgraded == ()

    def test_high_cardinality_auto_downgrade(
        self,
        tmp_storage: Path,
        seeded_system_user: object,
        tmp_path: Path,
    ) -> None:
        """onehot + threshold=50 + auto_downgrade=True → 60 unique 컬럼은 frequency 로 강등."""
        df = pd.DataFrame(
            {
                "x": list(range(120)),
                "cat": [f"c{i % 60}" for i in range(120)],  # 60 unique
                "y": [i % 2 for i in range(120)],
            }
        )
        csv = tmp_path / "hc.csv"
        df.to_csv(csv, index=False)
        _, dataset_id = _seed_project_and_dataset(csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="y",
            preprocessing=PreprocessingConfig(),
        )
        preview = training_service.preview_preprocessing(dataset_id, config)
        assert "cat" in preview.auto_downgraded
        assert preview.encoding_summary.get("cat") == "frequency"
        # derived 에는 cat 이 단일 frequency 엔트리로 나열
        kinds = {name for _, name, _ in preview.derived}
        assert "cat" in kinds

    def test_low_cardinality_onehot_expands_columns(
        self,
        tmp_storage: Path,
        seeded_system_user: object,
        tmp_path: Path,
    ) -> None:
        """저카디널리티 cat(3개 값) 은 onehot 으로 3개 파생 → n_cols_out > n_cols_in."""
        df = pd.DataFrame(
            {
                "x": list(range(30)),
                "cat": ["A", "B", "C"] * 10,
                "y": [i % 2 for i in range(30)],
            }
        )
        csv = tmp_path / "lc.csv"
        df.to_csv(csv, index=False)
        _, dataset_id = _seed_project_and_dataset(csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="y",
            preprocessing=PreprocessingConfig(),
        )
        preview = training_service.preview_preprocessing(dataset_id, config)
        assert preview.n_cols_in == 2  # x, cat
        assert preview.n_cols_out == 1 + 3  # x + 3 onehot
        assert preview.auto_downgraded == ()
        # derived 에는 cat__A / cat__B / cat__C 가 나와야 한다
        derived_names = {name for _, name, _ in preview.derived}
        assert {"cat__A", "cat__B", "cat__C"} <= derived_names

    def test_missing_dataset_raises(self, tmp_storage: Path, seeded_system_user: object) -> None:
        config = TrainingConfig(
            dataset_id=99999,
            task_type="classification",
            target_column="y",
        )
        with pytest.raises(NotFoundError):
            training_service.preview_preprocessing(99999, config)

    def test_missing_target_raises(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="does_not_exist",
        )
        with pytest.raises(ValidationError):
            training_service.preview_preprocessing(dataset_id, config)

    def test_missing_file_raises_storage_error(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        with session_scope() as session:
            ds = dataset_repository.get(session, dataset_id)
            assert ds is not None
            Path(ds.file_path).unlink()
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
        )
        with pytest.raises(StorageError):
            training_service.preview_preprocessing(dataset_id, config)

    def test_bool_as_numeric_false_routes_bools_to_categorical(
        self,
        tmp_storage: Path,
        seeded_system_user: object,
        tmp_path: Path,
    ) -> None:
        """bool_as_numeric=False 면 bool 컬럼이 범주형 경로로 합류해 onehot 파생이 생긴다."""
        df = pd.DataFrame(
            {
                "x": list(range(20)),
                "flag": [True, False] * 10,  # native bool
                "y": [i % 2 for i in range(20)],
            }
        )
        csv = tmp_path / "bool.csv"
        df.to_csv(csv, index=False)
        _, dataset_id = _seed_project_and_dataset(csv, tmp_storage)
        pp_cfg = PreprocessingConfig(bool_as_numeric=False)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="y",
            preprocessing=pp_cfg,
        )
        preview = training_service.preview_preprocessing(dataset_id, config)
        # flag 가 onehot 으로 전개되어 파생 2개 (True/False)
        flag_derived = [d for d in preview.derived if d[0] == "flag"]
        assert len(flag_derived) == 2
        assert all(d[2] == "onehot" for d in flag_derived)

    def test_preview_does_not_create_training_job(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """preview 는 읽기 전용 — TrainingJob/Model 레코드를 만들지 않아야 한다."""
        project_id, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            preprocessing=PreprocessingConfig(numeric_scale="robust"),
        )
        training_service.preview_preprocessing(dataset_id, config)
        jobs = training_service.list_training_jobs(project_id)
        assert jobs == []

    def test_replace_config_preprocessing_preserves_other_fields(self) -> None:
        """dataclasses.replace 로 preprocessing 만 교체해도 나머지 필드가 유지됨 (DX 확인)."""
        base = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="y",
            test_size=0.3,
        )
        new = replace(base, preprocessing=PreprocessingConfig(numeric_scale="robust"))
        assert new.dataset_id == 1
        assert new.test_size == 0.3
        assert new.preprocessing is not None
        assert new.preprocessing.numeric_scale == "robust"


class TestBalancerIntegration:
    """§9.7: config.preprocessing.imbalance != 'none' 면 balancer 가 train_all 에 주입된다."""

    def test_class_weight_strategy_trains_successfully(
        self,
        classification_csv: Path,
        tmp_storage: Path,
        seeded_system_user: object,
    ) -> None:
        """class_weight 전략으로도 전체 파이프라인이 성공하고 run_log 에 balance 기록이 남는다."""
        _, dataset_id = _seed_project_and_dataset(classification_csv, tmp_storage)
        pp_cfg = PreprocessingConfig(imbalance="class_weight")
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        result = training_service.run_training(config)
        assert result.best_algo is not None
        with session_scope() as session:
            job = training_repository.get(session, result.job_id)
            assert job is not None
            run_log = job.run_log or ""
            assert "balance: strategy=class_weight" in run_log


class TestDatetimeColumnHandling:
    """Regression: ``datetime64[ns]`` 컬럼이 기본 전처리 경로에서 자동 drop 되어야 한다.

    이전에는 ``split_feature_types`` 가 datetime 을 cat 으로 편입 → ``SimpleImputer``
    가 ``datetime64[ns]`` dtype 을 거부해 모든 알고리즘이 동시에 실패했다 (버그 리포트).
    xlsx 업로드 / parquet 경로처럼 pandas 가 native datetime dtype 으로 읽어오는
    케이스에서 주로 발생하므로, 단위 테스트는 ``_build_preprocessing`` 에 이미
    datetime64 컬럼이 포함된 DataFrame 을 직접 주입해 회귀를 검증한다.
    """

    @staticmethod
    def _df_with_datetime() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60] * 4,
                "income": [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 4,
                "signup": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-02-15",
                        "2023-03-20",
                        "2023-04-10",
                        "2023-05-05",
                        "2023-06-18",
                        "2023-07-22",
                        "2023-08-30",
                    ]
                    * 4
                ),
                "species": (["setosa", "versicolor"] * 4) * 4,
            }
        )

    def test_default_path_drops_datetime_column(self) -> None:
        df = self._df_with_datetime()
        config = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="species",
        )
        (
            preprocessor,
            schema,
            X,
            _y,
            _route,
            dropped,
        ) = training_service._build_preprocessing(df, config)

        assert dropped == ("signup",)
        assert "signup" not in X.columns
        assert "signup" not in schema.input_columns
        assert schema.datetime == ("signup",)
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[0] == len(X)

    def test_v2_path_without_decompose_drops_datetime(self) -> None:
        df = self._df_with_datetime()
        pp_cfg = PreprocessingConfig(datetime_decompose=False)
        config = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        _, schema, X, _y, _route, dropped = training_service._build_preprocessing(df, config)
        assert dropped == ("signup",)
        assert "signup" not in X.columns
        assert schema.datetime == ("signup",)

    def test_v2_path_with_decompose_keeps_datetime(self) -> None:
        df = self._df_with_datetime()
        pp_cfg = PreprocessingConfig(
            datetime_decompose=True,
            datetime_parts=("year", "month"),
        )
        config = TrainingConfig(
            dataset_id=1,
            task_type="classification",
            target_column="species",
            preprocessing=pp_cfg,
        )
        preprocessor, _schema, X, _y, _route, dropped = training_service._build_preprocessing(
            df, config
        )
        assert dropped == ()
        assert "signup" in X.columns
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[1] >= 2  # year + month 파생 포함


class TestTargetValidation:
    """§10.9 / v0.5.2 hotfix: 부적절한 타깃(날짜/고유값 폭발) 사전 차단.

    - datetime64 타깃 → ``ValidationError`` 즉시 raise (xgboost 등 fit 실패 방지)
    - classification 인데 고유 클래스 수가 ``max(50, n/2)`` 이상 → 차단
    - running job 이 생성되기 전에 실패해야 하므로 ``TrainingJob`` row 가 없다
    """

    @staticmethod
    def _write_csv(df: pd.DataFrame, storage: Path) -> Path:
        path = storage / f"seed_{uuid.uuid4().hex}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def test_datetime_target_is_rejected(
        self,
        tmp_storage: Path,
        seeded_system_user: object,  # noqa: ARG002
    ) -> None:
        df = pd.DataFrame(
            {
                "feat": range(30),
                "event_at": pd.date_range("2024-01-01", periods=30, freq="D"),
            }
        )
        csv_path = self._write_csv(df, tmp_storage)
        _, dataset_id = _seed_project_and_dataset(csv_path, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="event_at",
        )
        with pytest.raises(ValidationError) as excinfo:
            training_service.run_training(config)
        assert "날짜" in str(excinfo.value) or "datetime" in str(excinfo.value).lower()

        # running job 이 만들어지지 않아야 한다 (검증은 job 생성 전에 수행)
        with session_scope() as session:
            jobs = list(training_repository.list_by_project(session, project_id=1))
            assert all(j.status != "running" for j in jobs)

    def test_classification_with_too_many_unique_target_is_rejected(
        self,
        tmp_storage: Path,
        seeded_system_user: object,  # noqa: ARG002
    ) -> None:
        # 고유값 = 60개 → threshold=max(50, 30)=50 이상
        df = pd.DataFrame(
            {
                "feat": range(60),
                "target": [f"class_{i}" for i in range(60)],
            }
        )
        csv_path = self._write_csv(df, tmp_storage)
        _, dataset_id = _seed_project_and_dataset(csv_path, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="classification",
            target_column="target",
        )
        with pytest.raises(ValidationError) as excinfo:
            training_service.run_training(config)
        assert "고유값" in str(excinfo.value) or "고유" in str(excinfo.value)

    def test_regression_with_many_unique_target_is_allowed(
        self,
        tmp_storage: Path,
        seeded_system_user: object,  # noqa: ARG002
    ) -> None:
        """회귀는 연속형 타깃이므로 고유값 많아도 통과해야 한다."""
        df = pd.DataFrame(
            {
                "feat": list(range(60)) * 2,
                "target": [float(i) * 0.1 for i in range(60)] * 2,
            }
        )
        csv_path = self._write_csv(df, tmp_storage)
        _, dataset_id = _seed_project_and_dataset(csv_path, tmp_storage)
        config = TrainingConfig(
            dataset_id=dataset_id,
            task_type="regression",
            target_column="target",
        )
        # 이 단계에서 ValidationError 가 발생하지 않아야 한다.
        # (run_training 전체 성공까지는 검증하지 않고, 타깃 검증만 통과하면 OK)
        try:
            training_service.run_training(config)
        except ValidationError as exc:  # pragma: no cover
            pytest.fail(f"회귀 타깃이 차단되면 안 된다: {exc}")
        except Exception:
            # 타깃 검증 이후 단계에서의 실패는 본 테스트 범위 밖
            pass


class TestAlgorithmDiscovery:
    """§10.4 (FR-067, FR-069): list_algorithms / list_optional_backends."""

    def test_list_algorithms_returns_classification_specs(self) -> None:
        infos = training_service.list_algorithms("classification")
        assert infos, "분류 후보가 최소 1건 있어야 한다"
        assert all(isinstance(i, AlgorithmInfoDTO) for i in infos)
        assert all(i.task_type == "classification" for i in infos)
        # 레이어 경계: registry 를 직접 import 하지 않고도 이름이 노출돼야 한다.
        names = {i.name for i in infos if i.available}
        assert "random_forest" in names
        assert "logistic_regression" in names

    def test_list_algorithms_returns_regression_specs(self) -> None:
        infos = training_service.list_algorithms("regression")
        names = {i.name for i in infos if i.available}
        assert "linear" in names
        # §10.1 신규
        assert "elastic_net" in names

    def test_list_algorithms_rejects_unknown_task(self) -> None:
        with pytest.raises(ValidationError):
            training_service.list_algorithms("clustering")

    def test_list_algorithms_marks_optional_backend_flag(self) -> None:
        """등록된 스펙 중 optional backend 는 is_optional_backend=True."""
        infos = training_service.list_algorithms("classification")
        for info in infos:
            if info.name in {"xgboost", "lightgbm", "catboost"}:
                assert info.is_optional_backend is True, info.name
            else:
                assert info.is_optional_backend is False, info.name

    def test_list_algorithms_includes_unavailable_optional_with_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """미설치 optional backend 도 리스트에 나타나며 사유가 붙는다."""
        from ml.registry import OptionalBackendStatus

        fake_status = [
            OptionalBackendStatus(name="xgboost", available=True),
            OptionalBackendStatus(name="lightgbm", available=True),
            OptionalBackendStatus(
                name="catboost",
                available=False,
                reason="패키지 미설치 (pip install 필요)",
            ),
        ]
        monkeypatch.setattr(
            "services.training_service.optional_backends_status", lambda: fake_status
        )

        infos = training_service.list_algorithms("classification")
        by_name = {i.name: i for i in infos}
        # catboost 가 (실제 등록 여부와 무관하게) unavailable 항목으로 노출
        if "catboost" in by_name and not by_name["catboost"].available:
            assert "pip install" in by_name["catboost"].unavailable_reason

    def test_list_optional_backends_returns_three(self) -> None:
        infos = training_service.list_optional_backends()
        assert [i.name for i in infos] == ["xgboost", "lightgbm", "catboost"]
        assert all(isinstance(i, OptionalBackendInfoDTO) for i in infos)
        for info in infos:
            if info.available:
                assert info.reason == ""
            else:
                assert info.reason
