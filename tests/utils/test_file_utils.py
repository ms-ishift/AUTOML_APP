from __future__ import annotations

from pathlib import Path

import pytest

from utils.errors import ValidationError
from utils.file_utils import (
    ALLOWED_EXTENSIONS,
    extract_extension,
    read_tabular,
    save_uploaded_file,
    validate_columns,
    validate_extension,
    validate_size,
)


class FakeUploaded:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data


def test_extract_extension_lowercase() -> None:
    assert extract_extension("DATA.CSV") == "csv"
    assert extract_extension("report.XLSX") == "xlsx"


def test_extract_extension_no_dot() -> None:
    with pytest.raises(ValidationError):
        extract_extension("nodot")


def test_validate_extension_accepts_allowed() -> None:
    for ext in ALLOWED_EXTENSIONS:
        assert validate_extension(f"file.{ext}") == ext


def test_validate_extension_rejects_other() -> None:
    with pytest.raises(ValidationError):
        validate_extension("a.pdf")


def test_validate_size_limit() -> None:
    validate_size(size_bytes=1024, max_mb=1)  # 1KB < 1MB ok
    with pytest.raises(ValidationError):
        validate_size(size_bytes=2 * 1024 * 1024, max_mb=1)


def test_save_uploaded_file_writes_bytes(tmp_path: Path) -> None:
    up = FakeUploaded("data.csv", b"a,b\n1,2\n")
    dest = save_uploaded_file(up, project_id=7, target_root=tmp_path)

    assert dest.exists()
    assert dest.parent == tmp_path / "7"
    assert dest.suffix == ".csv"
    assert dest.read_bytes() == b"a,b\n1,2\n"


def test_read_tabular_csv_ok(tmp_path: Path) -> None:
    p = tmp_path / "ok.csv"
    p.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    df = read_tabular(p)
    assert list(df.columns) == ["x", "y"]
    assert df.shape == (2, 2)


def test_read_tabular_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError):
        read_tabular(p)


def test_read_tabular_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        read_tabular(p)


def test_validate_columns_detects_duplicates() -> None:
    with pytest.raises(ValidationError):
        validate_columns(["x", "x"])


def test_validate_columns_detects_unnamed() -> None:
    with pytest.raises(ValidationError):
        validate_columns(["x", "Unnamed: 1"])


def test_validate_columns_ok() -> None:
    validate_columns(["x", "y", "z"])
