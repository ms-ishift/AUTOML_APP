"""파일 업로드/읽기 유틸 (FR-030, FR-034, NFR-006).

설계 메모:
- Streamlit 의 ``UploadedFile`` 을 직접 import 하지 않고 duck-typing(``_UploadedLike``)으로 처리해
  단위 테스트에서 모킹이 용이하도록 한다.
- 저장 경로: ``<storage>/datasets/<project_id>/<uuid>.<ext>`` (datasets 루트는 주입 가능).
- 검증 순서: 확장자 → 크기 → 저장. 파싱 검증은 ``read_tabular`` 단계에서 수행.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path
from typing import Protocol

import pandas as pd

from config.settings import settings
from utils.errors import StorageError, ValidationError
from utils.messages import (
    Msg,
    upload_extension_not_allowed,
    upload_too_large,
)

ALLOWED_EXTENSIONS: tuple[str, ...] = ("csv", "xlsx")


class _UploadedLike(Protocol):
    """Streamlit ``UploadedFile`` 과 호환되는 최소 인터페이스."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


def extract_extension(filename: str) -> str:
    """파일명에서 소문자 확장자만 꺼낸다."""
    if "." not in filename:
        raise ValidationError(upload_extension_not_allowed("", ALLOWED_EXTENSIONS))
    return filename.rsplit(".", 1)[-1].lower()


def validate_extension(
    filename: str,
    allowed: tuple[str, ...] = ALLOWED_EXTENSIONS,
) -> str:
    """확장자를 검증하고 소문자 ext 를 반환 (FR-030)."""
    ext = extract_extension(filename)
    if ext not in allowed:
        raise ValidationError(upload_extension_not_allowed(ext, allowed))
    return ext


def validate_size(size_bytes: int, max_mb: int | None = None) -> None:
    """업로드 크기 제한 검증 (FR-030, 기본값: settings.MAX_UPLOAD_MB)."""
    limit = settings.MAX_UPLOAD_MB if max_mb is None else max_mb
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > limit:
        raise ValidationError(upload_too_large(size_mb, limit))


def ensure_dir(path: Path) -> Path:
    """디렉터리 생성 (idempotent)."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(
    uploaded: _UploadedLike,
    project_id: int,
    *,
    target_root: Path | None = None,
) -> Path:
    """업로드 파일을 ``<target_root>/<project_id>/<uuid>.<ext>`` 로 저장.

    ``target_root`` 기본값은 ``settings.datasets_dir``.
    """
    ext = validate_extension(uploaded.name)
    validate_size(uploaded.size)

    root = target_root or settings.datasets_dir
    dest_dir = ensure_dir(root / str(project_id))
    dest = dest_dir / f"{uuid.uuid4().hex}.{ext}"

    try:
        dest.write_bytes(uploaded.getvalue())
    except OSError as e:
        raise StorageError(Msg.UPLOAD_FAILED, cause=e) from e
    return dest


def validate_columns(columns: list[str]) -> None:
    """헤더/중복 컬럼 검증 (FR-034)."""
    stripped = [str(c).strip() for c in columns]
    if any(c == "" or c.lower().startswith("unnamed") for c in stripped):
        raise ValidationError(Msg.HEADER_MISSING)
    if len(stripped) != len(set(stripped)):
        raise ValidationError(Msg.DUPLICATED_COLUMNS)


def _read_by_ext(source: Path | BytesIO, ext: str) -> pd.DataFrame:
    if ext == "csv":
        return pd.read_csv(source)
    return pd.read_excel(source, engine="openpyxl")


def read_tabular(path: Path | str) -> pd.DataFrame:
    """CSV/XLSX 를 DataFrame 으로 로드하고 헤더/빈파일 검증까지 수행 (FR-034)."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(Msg.FILE_PARSE_FAILED)

    ext = extract_extension(p.name)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(upload_extension_not_allowed(ext, ALLOWED_EXTENSIONS))

    try:
        df = _read_by_ext(p, ext)
    except pd.errors.EmptyDataError as e:
        raise ValidationError(Msg.FILE_EMPTY, cause=e) from e
    except (ValueError, pd.errors.ParserError, OSError) as e:
        raise ValidationError(Msg.FILE_PARSE_FAILED, cause=e) from e

    if df.empty or df.shape[1] == 0:
        raise ValidationError(Msg.FILE_EMPTY)

    validate_columns([str(c) for c in df.columns])
    return df


def read_tabular_bytes(data: bytes, ext: str) -> pd.DataFrame:
    """바이트스트림에서 CSV/XLSX 로드 (검증 포함)."""
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(upload_extension_not_allowed(ext, ALLOWED_EXTENSIONS))

    try:
        df = _read_by_ext(BytesIO(data), ext)
    except pd.errors.EmptyDataError as e:
        raise ValidationError(Msg.FILE_EMPTY, cause=e) from e
    except (ValueError, pd.errors.ParserError, OSError) as e:
        raise ValidationError(Msg.FILE_PARSE_FAILED, cause=e) from e

    if df.empty or df.shape[1] == 0:
        raise ValidationError(Msg.FILE_EMPTY)

    validate_columns([str(c) for c in df.columns])
    return df
