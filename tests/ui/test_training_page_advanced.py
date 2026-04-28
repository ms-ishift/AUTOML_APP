"""§9.9 고급 전처리 expander UI 검증 (pages/03_training.py, FR-055~058).

범위:
- 기본 진입 시 expander 는 접혀 있고 위젯은 기본값 → `PreprocessingConfig.is_default == True`
- 스케일 `robust` 선택 후 학습 실행 → TrainingJob.run_log 에 `numeric_scale=robust` 포함 (@slow)
- 회귀 + SMOTE 조합은 UI 에서 SMOTE 옵션이 노출되지 않음 (옵션 축소 + 안내)
- imblearn 미설치 모킹 → 분류에서도 SMOTE 옵션 제거 + 사유 caption
- 미리보기 버튼 → FeaturePreviewDTO 렌더 (파생 onehot 확장 케이스)
- 커스텀 값 설정 시 커스텀 뱃지 caption 노출
- 회귀 전환 후 previous SMOTE 세션 상태는 강제 "none" 으로 정규화
- 미리보기에서 FeaturePreviewDTO.auto_downgraded 가 존재하면 info 노출
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from services import dataset_service, project_service
from services.dto import FeaturePreviewDTO
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


def _mixed_csv(tmp_path: Path, *, rows: int = 100) -> Path:
    """수치 + 범주(저 카디널리티) + 고카디널리티 + bool + datetime 을 포함하는 혼합 CSV.

    - `cat_low` (3 카테고리) → onehot 분해 → n_cols_out 증가
    - `cat_high` (>50 nunique) → 고카디널리티 자동 강등 시나리오
    - `is_active` (bool) / `dt` (datetime) → 추후 decompose 옵션 검증용
    """
    df = pd.DataFrame(
        {
            "target": [i % 2 for i in range(rows)],
            "num_a": [i * 0.5 for i in range(rows)],
            "num_b": [(i % 7) + 0.1 for i in range(rows)],
            "cat_low": [["x", "y", "z"][i % 3] for i in range(rows)],
            # nunique=50 이지만 unique_ratio=0.5 라 suggest_excluded 대상이 아니다.
            "cat_high": [f"v{i % 50}" for i in range(rows)],
            "is_active": [bool(i % 2) for i in range(rows)],
            "dt": pd.date_range("2024-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        }
    )
    csv = tmp_path / "mixed.csv"
    df.to_csv(csv, index=False)
    return csv


def _prepare_page(project_id: int, dataset_id: int) -> AppTest:
    at = _new_page()
    at.session_state[SessionKey.CURRENT_PROJECT_ID] = project_id
    at.session_state[SessionKey.CURRENT_DATASET_ID] = dataset_id
    return at


# ----------------------------------------------------------------- defaults


def test_expander_defaults_are_backward_compatible(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """기본 진입 시 PP_ 위젯 값들은 PreprocessingConfig() 와 동치여야 한다."""
    from ml.schemas import PreprocessingConfig

    project_id, dataset_id = _seed_csv(classification_csv, "pp-defaults")
    at = _prepare_page(project_id, dataset_id).run()
    assert not at.exception

    # 위젯들이 세션에 초기 세팅됨
    s = at.session_state
    cfg = PreprocessingConfig(
        numeric_impute=s["pp_numeric_impute"],
        numeric_scale=s["pp_numeric_scale"],
        outlier=s["pp_outlier"],
        categorical_impute=s["pp_categorical_impute"],
        categorical_encoding=s["pp_categorical_encoding"],
        highcard_auto_downgrade=s["pp_highcard_auto_downgrade"],
        highcard_threshold=s["pp_highcard_threshold"],
        datetime_decompose=s["pp_datetime_decompose"],
        bool_as_numeric=s["pp_bool_as_numeric"],
        imbalance=s["pp_imbalance"],
    )
    assert cfg.is_default


# --------------------------------------------------------- regression path


def test_regression_hides_smote_option(
    tmp_storage: Path,
    seeded_system_user: object,
    regression_csv: Path,
) -> None:
    """회귀로 전환하면 불균형 전략 radio 의 옵션이 ('none',) 로 축소된다."""
    project_id, dataset_id = _seed_csv(regression_csv, "pp-reg")
    at = _prepare_page(project_id, dataset_id).run()
    at.radio(key="training_task_type").set_value("regression").run()

    imbalance_radio = at.radio(key="pp_imbalance")
    assert tuple(imbalance_radio.options) == ("none",)


def test_regression_forces_imbalance_to_none_even_if_session_had_smote(
    tmp_storage: Path,
    seeded_system_user: object,
    regression_csv: Path,
) -> None:
    """세션에 과거 smote 값이 남아있어도 회귀에서는 none 으로 정규화된다."""
    project_id, dataset_id = _seed_csv(regression_csv, "pp-reg-smote")
    at = _prepare_page(project_id, dataset_id)
    # 이전 세션에 smote 가 선택되었던 상황 모사
    at.session_state["pp_imbalance"] = "smote"
    at.run()
    at.radio(key="training_task_type").set_value("regression").run()

    assert at.session_state["pp_imbalance"] == "none"


# -------------------------------------------------- imblearn unavailable


def test_smote_option_hidden_when_imblearn_missing(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SMOTE_AVAILABLE=False 를 모킹하면 분류에서도 smote 옵션이 보이지 않는다."""
    import ml.balancing as bal_mod

    monkeypatch.setattr(bal_mod, "SMOTE_AVAILABLE", False)

    project_id, dataset_id = _seed_csv(classification_csv, "pp-noimb")
    at = _prepare_page(project_id, dataset_id).run()

    imbalance_radio = at.radio(key="pp_imbalance")
    assert "smote" not in imbalance_radio.options
    # caption 에 미설치 안내가 나와야 한다 (Msg.PREPROCESSING_SMOTE_UNAVAILABLE).
    captions = " ".join(c.value for c in at.caption)
    assert "imbalanced-learn" in captions


# ----------------------------------------------------------- preview flow


def test_preview_button_renders_feature_preview(
    tmp_storage: Path,
    seeded_system_user: object,
    tmp_path: Path,
) -> None:
    """미리보기 버튼 → FeaturePreviewDTO 메트릭 3종 + 파생 테이블 렌더.

    onehot 인코딩으로 cat_low(3 카테고리) 가 여러 열로 분해되어 n_cols_out > n_cols_in 를 기대.
    """
    csv = _mixed_csv(tmp_path)
    project_id, dataset_id = _seed_csv(csv, "pp-preview")
    at = _prepare_page(project_id, dataset_id).run()
    # 타깃을 명시적으로 target 으로 고정 (첫 컬럼이 target 이므로 기본 그대로)
    target_select = at.selectbox(key="training_target_col")
    assert target_select.value == "target"

    preview_btn = next(b for b in at.button if b.key == "pp_preview_btn")
    preview_btn.click().run()

    labels = [m.label for m in at.metric]
    assert "원본 열" in labels
    assert "변환 후 열" in labels
    assert "파생 피처" in labels

    assert "pp_preview_result" in at.session_state
    result = at.session_state["pp_preview_result"]
    assert isinstance(result, FeaturePreviewDTO)
    assert result.n_cols_out > result.n_cols_in


def test_preview_auto_downgrade_shows_info(
    tmp_storage: Path,
    seeded_system_user: object,
    tmp_path: Path,
) -> None:
    """cat_high 가 자동 강등되면 auto_downgraded 가 채워지고 st.info 로 안내된다."""
    csv = _mixed_csv(tmp_path)
    project_id, dataset_id = _seed_csv(csv, "pp-downgrade")
    at = _prepare_page(project_id, dataset_id)
    # 임계값을 5 로 낮춰 cat_low(3)는 유지, cat_high(100)만 강등되도록
    at.session_state["pp_highcard_threshold"] = 5
    at.run()

    preview_btn = next(b for b in at.button if b.key == "pp_preview_btn")
    preview_btn.click().run()

    assert "pp_preview_result" in at.session_state
    result = at.session_state["pp_preview_result"]
    assert isinstance(result, FeaturePreviewDTO)
    assert "cat_high" in result.auto_downgraded
    info_text = " ".join(i.value for i in at.info)
    assert "frequency" in info_text


# ---------------------------------------------------------- custom badge


def test_custom_preprocessing_badge_caption_appears(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """numeric_scale='robust' 를 세팅하면 커스텀 뱃지 caption 이 노출된다."""
    project_id, dataset_id = _seed_csv(classification_csv, "pp-badge")
    at = _prepare_page(project_id, dataset_id)
    at.session_state["pp_numeric_scale"] = "robust"
    at.run()

    captions = " ".join(c.value for c in at.caption)
    assert "커스텀 전처리" in captions


# ---------------------------------------------------------- happy training


@pytest.mark.slow
def test_robust_scale_propagates_to_run_log(
    tmp_storage: Path,
    seeded_system_user: object,
    classification_csv: Path,
) -> None:
    """스케일 robust 로 학습 실행 → TrainingJob.run_log 에 numeric_scale=robust 포함."""
    from repositories import training_repository
    from repositories.base import session_scope

    project_id, dataset_id = _seed_csv(classification_csv, "pp-robust")
    at = _prepare_page(project_id, dataset_id)
    at.session_state["pp_numeric_scale"] = "robust"
    at.run()

    # iris 의 target 은 마지막 컬럼(species) → 선택 변경
    target_select = at.selectbox(key="training_target_col")
    target_select.set_value(target_select.options[-1]).run()

    submit = next(b for b in at.button if b.key == "training_submit_btn")
    submit.click().run()
    assert not at.exception

    job_id = at.session_state[SessionKey.LAST_TRAINING_JOB_ID]
    with session_scope() as session:
        job = training_repository.get(session, job_id)
        assert job is not None
        assert "numeric_scale=robust" in (job.run_log or "")
