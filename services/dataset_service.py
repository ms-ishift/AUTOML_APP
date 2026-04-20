"""Dataset Service (IMPLEMENTATION_PLAN §4.2, FR-030~035).

책임:
- 업로드 파일을 받아 확장자·크기·헤더 검증 → 디스크 저장 → 프로파일 생성 → DB insert
- 파일 I/O 와 DB 트랜잭션의 일관성 보장 (실패 시 보상 삭제)
- Repository → ORM → DTO 변환 (UI 는 ORM/DataFrame 을 직접 받지 않는다)

레이어 규약 (``.cursor/rules/service-layer.mdc``):
- Streamlit 타입 참조 금지. 업로드 파일은 duck-typing(``utils.file_utils._UploadedLike``)으로 처리.
- 사용자 노출 메시지는 ``utils.messages.Msg`` 에서만 꺼내 쓴다.
- 변경 액션은 ``audit_repository.write`` 로 감사 로그를 남긴다 (성공 + 실패).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml.profiling import ID_UNIQUE_RATIO_THRESHOLD, profile_dataframe
from repositories import audit_repository, dataset_repository, project_repository
from repositories.base import session_scope
from services.dto import ColumnProfileDTO, DatasetDTO, DatasetProfileDTO
from utils.errors import AppError, NotFoundError, StorageError, ValidationError
from utils.events import Event
from utils.file_utils import read_tabular, save_uploaded_file
from utils.log_utils import get_logger, log_event
from utils.messages import Msg, entity_not_found

if TYPE_CHECKING:
    from ml.schemas import DatasetProfile
    from utils.file_utils import _UploadedLike

logger = get_logger(__name__)

PREVIEW_DEFAULT_ROWS = 50
PREVIEW_MAX_ROWS = 500


# ---------------------------------------------------------------- internals


def _profile_to_json(profile: DatasetProfile) -> dict[str, Any]:
    """``DatasetProfile`` 를 JSON 직렬화 가능한 dict 로 변환 (DB ``schema_json``)."""
    return {
        "rows": profile.n_rows,
        "cols": profile.n_cols,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "n_missing": c.n_missing,
                "n_unique": c.n_unique,
                "missing_ratio": c.missing_ratio,
                "unique_ratio": c.unique_ratio,
            }
            for c in profile.columns
        ],
    }


def _json_to_profile_dto(data: dict[str, Any] | None) -> DatasetProfileDTO:
    """``schema_json`` → ``DatasetProfileDTO`` 역직렬화."""
    data = data or {}
    cols_raw = data.get("columns") or []
    columns = [
        ColumnProfileDTO(
            name=str(c.get("name", "")),
            dtype=str(c.get("dtype", "")),
            n_missing=int(c.get("n_missing", 0)),
            n_unique=int(c.get("n_unique", 0)),
            missing_ratio=float(c.get("missing_ratio", 0.0)),
            unique_ratio=float(c.get("unique_ratio", 0.0)),
        )
        for c in cols_raw
    ]
    return DatasetProfileDTO(
        rows=int(data.get("rows", 0)),
        cols=int(data.get("cols", len(columns))),
        columns=columns,
    )


def _cleanup_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception(
            "file.cleanup_failed",
            extra={"path": str(path)},
        )


def _audit_upload_failure(
    project_id: int,
    filename: str,
    error: AppError,
) -> None:
    """업로드 실패 감사 로그. 이 자체가 실패해도 본 예외를 가리지 않는다."""
    try:
        with session_scope() as session:
            audit_repository.write(
                session,
                action_type=Event.DATASET_UPLOAD_FAILED,
                target_type="Project",
                target_id=project_id,
                detail={"file_name": filename, "error": str(error)},
            )
    except Exception:  # pragma: no cover - 감사 실패는 조용히 로깅만
        logger.exception("audit.write_failed", extra={"project_id": project_id})
    log_event(
        logger,
        Event.DATASET_UPLOAD_FAILED,
        project_id=project_id,
        file_name=filename,
        error=str(error),
    )


# ---------------------------------------------------------- Public use-cases


def upload_dataset(project_id: int, uploaded: _UploadedLike) -> DatasetDTO:
    """FR-030, FR-033, FR-034: 데이터셋 업로드.

    처리 순서:
    1) 프로젝트 존재 확인 (없으면 orphan 파일을 만들지 않는다)
    2) 확장자/크기 검증 후 ``<storage>/datasets/<project_id>/<uuid>.<ext>`` 저장
    3) ``read_tabular`` 로 헤더/빈파일/중복컬럼 검증 + DataFrame 로딩
    4) ``profile_dataframe`` 으로 프로파일 생성
    5) DB insert + 감사 로그 ``dataset.uploaded``

    실패 보상: 저장 후 어느 단계에서든 실패하면 방금 쓴 파일을 정리하고 감사 실패 로그를 남긴다.
    """
    with session_scope() as session:
        project = project_repository.get(session, project_id)
        if project is None:
            err = NotFoundError(entity_not_found("프로젝트", project_id))
            _audit_upload_failure(project_id, getattr(uploaded, "name", "?"), err)
            raise err

    saved_path: Path | None = None
    try:
        saved_path = save_uploaded_file(uploaded, project_id)
        df = read_tabular(saved_path)
        profile = profile_dataframe(df)

        with session_scope() as session:
            dataset = dataset_repository.insert(
                session,
                project_id=project_id,
                file_name=uploaded.name,
                file_path=str(saved_path),
                row_count=profile.n_rows,
                column_count=profile.n_cols,
                schema_json=_profile_to_json(profile),
            )
            audit_repository.write(
                session,
                action_type=Event.DATASET_UPLOADED,
                target_type="Dataset",
                target_id=dataset.dataset_id,
                detail={
                    "project_id": project_id,
                    "file_name": uploaded.name,
                    "rows": profile.n_rows,
                    "cols": profile.n_cols,
                },
            )
            log_event(
                logger,
                Event.DATASET_UPLOADED,
                dataset_id=dataset.dataset_id,
                project_id=project_id,
                rows=profile.n_rows,
                cols=profile.n_cols,
            )
            return DatasetDTO.from_orm(dataset)
    except AppError as err:
        _cleanup_file(saved_path)
        _audit_upload_failure(project_id, getattr(uploaded, "name", "?"), err)
        raise
    except Exception as err:  # 예기치 못한 예외도 파일 정리 후 StorageError 로 포장
        _cleanup_file(saved_path)
        wrapped = StorageError(Msg.UPLOAD_FAILED, cause=err)
        _audit_upload_failure(project_id, getattr(uploaded, "name", "?"), wrapped)
        raise wrapped from err


def get_dataset_profile(dataset_id: int) -> DatasetProfileDTO:
    """FR-033: 저장된 프로파일(``schema_json``) → ``DatasetProfileDTO`` 반환.

    schema_json 이 비어 있으면(과거 데이터 호환성) 파일을 다시 읽어 재생성한다.
    """
    with session_scope() as session:
        dataset = dataset_repository.get(session, dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", dataset_id))
        if dataset.schema_json:
            return _json_to_profile_dto(dataset.schema_json)
        file_path = Path(dataset.file_path)

    if not file_path.exists():
        raise StorageError(Msg.FILE_PARSE_FAILED)
    df = read_tabular(file_path)
    profile = profile_dataframe(df)
    return _json_to_profile_dto(_profile_to_json(profile))


def preview_dataset(dataset_id: int, n: int = PREVIEW_DEFAULT_ROWS) -> list[dict[str, Any]]:
    """FR-031: 상위 ``n`` 행을 JSON 직렬화 가능한 ``list[dict]`` 로 반환.

    - ``n`` 은 [1, ``PREVIEW_MAX_ROWS``] 범위로 강제 클램프.
    - ``NaN`` 은 JSON 변환 시 ``None`` 으로 자연스레 치환된다.
    """
    if n <= 0:
        raise ValidationError("미리보기 행 수는 1 이상이어야 합니다.")
    n = min(n, PREVIEW_MAX_ROWS)

    with session_scope() as session:
        dataset = dataset_repository.get(session, dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", dataset_id))
        file_path = Path(dataset.file_path)

    if not file_path.exists():
        raise StorageError(Msg.FILE_PARSE_FAILED)

    df = read_tabular(file_path).head(n)
    # to_json → json.loads 경로가 NaN/Timestamp/NumPy 스칼라를 모두 JSON 기본 타입으로 정리.
    payload = df.to_json(orient="records", date_format="iso", force_ascii=False)
    return list(json.loads(payload))


def suggest_excluded_columns(
    dataset_id: int,
    *,
    unique_ratio_threshold: float = ID_UNIQUE_RATIO_THRESHOLD,
) -> list[str]:
    """FR-044 힌트: 식별자 의심 컬럼(고유값 비율 ≥ 임계) 목록.

    UI 의 "제외 컬럼 기본값"으로 사용. 비즈니스 로직이므로 페이지에 인라인하지 않고
    Service 에서 노출한다. 내부적으로 ``get_dataset_profile`` 을 재사용 (schema_json
    이 비어 있으면 파일에서 재생성).
    """
    profile = get_dataset_profile(dataset_id)
    if profile.rows < 2:
        return []
    return [c.name for c in profile.columns if c.unique_ratio >= unique_ratio_threshold]


def list_datasets(project_id: int) -> list[DatasetDTO]:
    """FR-032: 프로젝트 소속 데이터셋 목록(최신순)."""
    with session_scope() as session:
        if project_repository.get(session, project_id) is None:
            raise NotFoundError(entity_not_found("프로젝트", project_id))
        rows = dataset_repository.list_by_project(session, project_id)
        return [DatasetDTO.from_orm(ds) for ds in rows]


def delete_dataset(dataset_id: int) -> None:
    """FR-035: 데이터셋 삭제 (파일 포함).

    ORM 관계에 ``cascade="all, delete-orphan"`` 가 걸려 있으므로 TrainingJob / Model 도 함께 삭제된다.
    파일 삭제는 DB 커밋 이후 best-effort 로 수행한다 (롤백 시 파일이 남아도 무결성 이슈 없음).
    """
    with session_scope() as session:
        dataset = dataset_repository.get(session, dataset_id)
        if dataset is None:
            raise NotFoundError(entity_not_found("데이터셋", dataset_id))
        file_path = Path(dataset.file_path)
        project_id = dataset.project_id
        file_name = dataset.file_name
        dataset_repository.delete(session, dataset_id)
        audit_repository.write(
            session,
            action_type=Event.DATASET_DELETED,
            target_type="Dataset",
            target_id=dataset_id,
            detail={"project_id": project_id, "file_name": file_name},
        )
        log_event(
            logger,
            Event.DATASET_DELETED,
            dataset_id=dataset_id,
            project_id=project_id,
        )

    _cleanup_file(file_path)


__all__ = [
    "PREVIEW_DEFAULT_ROWS",
    "PREVIEW_MAX_ROWS",
    "upload_dataset",
    "get_dataset_profile",
    "preview_dataset",
    "suggest_excluded_columns",
    "list_datasets",
    "delete_dataset",
]
