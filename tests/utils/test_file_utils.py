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


# --------------------------------------------- datetime auto-detection (fix)


class TestDatetimeAutoDetection:
    """CSV 로드 시 object dtype 의 날짜 문자열 컬럼을 ``datetime64[ns]`` 로 자동 변환.

    목적: `split_feature_types_v2` 가 datetime 그룹으로 인식해 고급 전처리의
    ``datetime_decompose`` 옵션이 실제 효과를 내도록 한다. 또한 범주형 경로로
    잘못 흘러 onehot 이 폭증하는 것을 방지.
    """

    def test_obvious_date_column_is_coerced(self, tmp_path: Path) -> None:
        import pandas as pd

        p = tmp_path / "dt.csv"
        rows = "\n".join(f"2024-0{(i % 9) + 1}-15,{i}" for i in range(20))
        p.write_text("signup,value\n" + rows + "\n", encoding="utf-8")
        df = read_tabular(p)
        assert pd.api.types.is_datetime64_any_dtype(df["signup"])
        assert pd.api.types.is_numeric_dtype(df["value"])

    def test_pure_numeric_string_column_not_converted(self, tmp_path: Path) -> None:
        import pandas as pd

        p = tmp_path / "num.csv"
        rows = "\n".join(f"{i},{i + 100}" for i in range(20))
        p.write_text("id,score\n" + rows + "\n", encoding="utf-8")
        df = read_tabular(p)
        assert not pd.api.types.is_datetime64_any_dtype(df["id"])
        assert not pd.api.types.is_datetime64_any_dtype(df["score"])

    def test_small_sample_skipped(self, tmp_path: Path) -> None:
        """non-null row 가 너무 적으면 휴리스틱이 불안정하므로 건드리지 않는다."""
        p = tmp_path / "tiny.csv"
        p.write_text("signup,v\n2024-01-01,1\n2024-02-01,2\n", encoding="utf-8")
        df = read_tabular(p)
        # 2행만 있으므로 _DATETIME_DETECT_MIN_NON_NULL 미만 → object 유지.
        assert df["signup"].dtype == "object"

    def test_mixed_non_date_stays_object(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.csv"
        lines = ["label,v"] + [f"{x},{i}" for i, x in enumerate(["foo", "bar", "baz"] * 7)]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        df = read_tabular(p)
        assert df["label"].dtype == "object"
