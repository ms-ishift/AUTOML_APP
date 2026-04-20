"""Dataset Service 통합 테스트 (IMPLEMENTATION_PLAN §4.2).

- 실제 sqlite + 임시 STORAGE_DIR 에서 upload → profile/preview → list → delete 전체 흐름 검증.
- Streamlit ``UploadedFile`` 은 duck-typing 으로 가짜 객체(``FakeUpload``) 를 주입.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from repositories.models import AuditLog, Dataset, User
from services import dataset_service, project_service
from services.dto import DatasetDTO, DatasetProfileDTO
from utils.errors import NotFoundError, StorageError, ValidationError
from utils.events import Event

# ------------------------------------------------------- fake upload object


@dataclass
class FakeUpload:
    """Streamlit ``UploadedFile`` 과 호환되는 최소 duck-typing 객체."""

    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


def _load_upload(path: Path, name: str | None = None) -> FakeUpload:
    return FakeUpload(name=name or path.name, data=path.read_bytes())


def _new_project(name: str = "ds-proj") -> int:
    return project_service.create_project(name).id


# ----------------------------------------------------------------- helpers


def _audit_actions(sqlite_engine) -> list[str]:
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        return [row.action_type for row in session.query(AuditLog).all()]


def _dataset_rows(sqlite_engine) -> list[Dataset]:
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        return list(session.query(Dataset).all())


# ------------------------------------------------------------ upload: happy


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_happy_path(
    sqlite_engine,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project()
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    assert isinstance(dto, DatasetDTO)
    assert dto.project_id == project_id
    assert dto.file_name == classification_csv.name
    assert dto.row_count > 0
    assert dto.column_count > 0
    assert Event.DATASET_UPLOADED in _audit_actions(sqlite_engine)


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_writes_uuid_file(
    tmp_storage: Path,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project()
    dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    target_dir = tmp_storage / "datasets" / str(project_id)
    files = list(target_dir.iterdir())
    assert len(files) == 1
    saved = files[0]
    assert saved.suffix == ".csv"
    assert saved.name != classification_csv.name  # UUID 재명명


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_stores_profile_in_schema_json(
    sqlite_engine,
    seeded_system_user: User,
    regression_csv: Path,
) -> None:
    project_id = _new_project("reg-proj")
    dto = dataset_service.upload_dataset(project_id, _load_upload(regression_csv))

    profile = dataset_service.get_dataset_profile(dto.id)
    assert isinstance(profile, DatasetProfileDTO)
    assert profile.rows == dto.row_count
    assert profile.cols == dto.column_count
    assert len(profile.columns) == dto.column_count
    assert all(c.name for c in profile.columns)


# ------------------------------------------------------------ upload: failure


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_rejects_missing_project(
    sqlite_engine,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.upload_dataset(9999, _load_upload(classification_csv))
    assert Event.DATASET_UPLOAD_FAILED in _audit_actions(sqlite_engine)
    assert Event.DATASET_UPLOADED not in _audit_actions(sqlite_engine)


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_rejects_bad_extension(
    sqlite_engine,
    seeded_system_user: User,
) -> None:
    project_id = _new_project("bad-ext")
    bad = FakeUpload(name="notes.txt", data=b"hello")

    with pytest.raises(ValidationError):
        dataset_service.upload_dataset(project_id, bad)

    assert _dataset_rows(sqlite_engine) == []
    assert Event.DATASET_UPLOAD_FAILED in _audit_actions(sqlite_engine)


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_cleans_file_on_parse_failure(
    tmp_storage: Path,
    sqlite_engine,
    seeded_system_user: User,
) -> None:
    project_id = _new_project("bad-header")
    # 중간 컬럼명이 비어 있어 pandas 가 "Unnamed: 1" 로 읽고, validate_columns 가 차단.
    bad_csv = b"a,,b\n1,2,3\n4,5,6\n"
    bad = FakeUpload(name="bad.csv", data=bad_csv)

    with pytest.raises(ValidationError):
        dataset_service.upload_dataset(project_id, bad)

    target_dir = tmp_storage / "datasets" / str(project_id)
    leftover = list(target_dir.iterdir()) if target_dir.exists() else []
    assert leftover == []
    assert _dataset_rows(sqlite_engine) == []


@pytest.mark.usefixtures("tmp_storage")
def test_upload_dataset_rejects_empty_file(
    sqlite_engine,
    seeded_system_user: User,
) -> None:
    project_id = _new_project("empty-file")
    empty = FakeUpload(name="empty.csv", data=b"")

    with pytest.raises(ValidationError):
        dataset_service.upload_dataset(project_id, empty)
    assert _dataset_rows(sqlite_engine) == []


# ----------------------------------------------------------------- preview


@pytest.mark.usefixtures("tmp_storage")
def test_preview_dataset_returns_top_n(
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("preview-p")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    rows = dataset_service.preview_dataset(dto.id, n=5)
    assert isinstance(rows, list)
    assert len(rows) == 5
    # JSON-호환 dict 인지 확인
    import json

    json.dumps(rows)


@pytest.mark.usefixtures("tmp_storage")
def test_preview_dataset_clamps_to_max(
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("preview-clamp")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    rows = dataset_service.preview_dataset(dto.id, n=10_000)
    assert len(rows) <= dataset_service.PREVIEW_MAX_ROWS


def test_preview_dataset_validates_n(seeded_system_user: User) -> None:
    with pytest.raises(ValidationError):
        dataset_service.preview_dataset(1, n=0)


@pytest.mark.usefixtures("tmp_storage")
def test_preview_dataset_normalizes_nan_to_none(
    tmp_storage: Path,
    seeded_system_user: User,
) -> None:
    project_id = _new_project("nan-csv")
    csv_bytes = b"x,y\n1,a\n,b\n3,\n"
    dto = dataset_service.upload_dataset(project_id, FakeUpload(name="na.csv", data=csv_bytes))

    rows = dataset_service.preview_dataset(dto.id, n=5)
    assert rows[1]["x"] is None
    assert rows[2]["y"] is None


def test_preview_dataset_missing_dataset_raises(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.preview_dataset(9999)


# ------------------------------------------------------------------ profile


@pytest.mark.usefixtures("tmp_storage")
def test_get_dataset_profile_rebuilds_from_file_when_schema_missing(
    sqlite_engine,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("rebuild-profile")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        ds = session.get(Dataset, dto.id)
        assert ds is not None
        ds.schema_json = None
        session.commit()

    profile = dataset_service.get_dataset_profile(dto.id)
    assert profile.rows == dto.row_count
    assert profile.cols == dto.column_count


def test_get_dataset_profile_missing_dataset(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.get_dataset_profile(9999)


@pytest.mark.usefixtures("tmp_storage")
def test_get_dataset_profile_missing_file_raises_storage_error(
    sqlite_engine,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("missing-file")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        ds = session.get(Dataset, dto.id)
        assert ds is not None
        Path(ds.file_path).unlink(missing_ok=True)
        ds.schema_json = None
        session.commit()

    with pytest.raises(StorageError):
        dataset_service.get_dataset_profile(dto.id)


# -------------------------------------------------------------------- list


@pytest.mark.usefixtures("tmp_storage")
def test_list_datasets_orders_newest_first(
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("list-proj")
    first = dataset_service.upload_dataset(
        project_id, _load_upload(classification_csv, name="first.csv")
    )
    second = dataset_service.upload_dataset(
        project_id, _load_upload(classification_csv, name="second.csv")
    )

    rows = dataset_service.list_datasets(project_id)
    assert [r.id for r in rows] == [second.id, first.id]


def test_list_datasets_missing_project_raises(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.list_datasets(9999)


# ------------------------------------------------------------------ delete


@pytest.mark.usefixtures("tmp_storage")
def test_delete_dataset_removes_file_and_row(
    tmp_storage: Path,
    sqlite_engine,
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("del-proj")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    target_dir = tmp_storage / "datasets" / str(project_id)
    assert any(target_dir.iterdir())

    dataset_service.delete_dataset(dto.id)

    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    with factory() as session:
        assert session.get(Dataset, dto.id) is None
    assert list(target_dir.iterdir()) == []
    assert Event.DATASET_DELETED in _audit_actions(sqlite_engine)


def test_delete_dataset_missing_raises_not_found(seeded_system_user: User) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.delete_dataset(9999)


# ---------------------------------------------------------- layer-boundary


def test_dataset_service_is_streamlit_free() -> None:
    import inspect

    source = inspect.getsource(dataset_service)
    assert "streamlit" not in source
    assert "import st " not in source


# ---------------------------------------------------------- cascade guard
# (project_service.delete_project(cascade=False) 가 dataset 존재 시 차단되는지 확인)


@pytest.mark.usefixtures("tmp_storage")
def test_project_delete_without_cascade_blocks_dataset(
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    project_id = _new_project("cascade-check")
    dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    with pytest.raises(ValidationError):
        project_service.delete_project(project_id, cascade=False)


# ---------------------------------------------------------------- smoke


@pytest.mark.usefixtures("tmp_storage")
def test_upload_reads_dataframe_consistently(
    seeded_system_user: User,
    classification_csv: Path,
) -> None:
    """DB 에 저장된 row/col 이 실제 pandas 로드 결과와 동일한지 확인."""
    expected = pd.read_csv(classification_csv)
    project_id = _new_project("consistency")
    dto = dataset_service.upload_dataset(project_id, _load_upload(classification_csv))

    assert dto.row_count == expected.shape[0]
    assert dto.column_count == expected.shape[1]


# ------------------------------------------------- suggest_excluded_columns


@pytest.mark.usefixtures("tmp_storage")
def test_suggest_excluded_columns_flags_identifier(
    seeded_system_user: User,
    tmp_path: Path,
) -> None:
    """고유값 비율이 임계 이상인 컬럼이 힌트로 반환된다."""
    project_id = _new_project("sug-id")
    df = pd.DataFrame(
        {
            "id": list(range(100)),  # unique_ratio=1.0 → 힌트
            "age": [i % 20 for i in range(100)],
            "target": [i % 2 for i in range(100)],
        }
    )
    csv_path = tmp_path / "with_id.csv"
    df.to_csv(csv_path, index=False)
    dto = dataset_service.upload_dataset(project_id, _load_upload(csv_path))

    suggestions = dataset_service.suggest_excluded_columns(dto.id)
    assert "id" in suggestions
    assert "age" not in suggestions


@pytest.mark.usefixtures("tmp_storage")
def test_suggest_excluded_columns_respects_threshold(
    seeded_system_user: User,
    tmp_path: Path,
) -> None:
    """임계값을 낮추면 더 많은 컬럼이 힌트로 잡힌다."""
    project_id = _new_project("sug-threshold")
    df = pd.DataFrame(
        {
            "id": list(range(100)),
            "category_50": [i % 50 for i in range(100)],  # unique_ratio=0.5
            "y": [i % 2 for i in range(100)],
        }
    )
    csv_path = tmp_path / "threshold.csv"
    df.to_csv(csv_path, index=False)
    dto = dataset_service.upload_dataset(project_id, _load_upload(csv_path))

    default_hint = dataset_service.suggest_excluded_columns(dto.id)
    assert default_hint == ["id"]

    loose = dataset_service.suggest_excluded_columns(dto.id, unique_ratio_threshold=0.4)
    assert "id" in loose and "category_50" in loose


def test_suggest_excluded_columns_unknown_dataset_raises(
    sqlite_engine,
    seeded_system_user: User,
) -> None:
    with pytest.raises(NotFoundError):
        dataset_service.suggest_excluded_columns(9999)
