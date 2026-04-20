"""전역 설정 (pydantic-settings).

모든 모듈은 `from config.settings import settings` 로만 접근한다.
관련 계획서: IMPLEMENTATION_PLAN.md §1.1.

주요 매핑:
- AUTH_MODE         → FR-010 (MVP 기본 none)
- MAX_UPLOAD_MB     → FR-030
- DEFAULT_TEST_SIZE → FR-043
- RANDOM_SEED       → NFR (재현성)
- LOG_LEVEL         → NFR-007
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: Literal["dev", "prod"] = "dev"
    DATABASE_URL: str = f"sqlite:///{ROOT / 'db' / 'app.db'}"
    STORAGE_DIR: Path = Field(default_factory=lambda: ROOT / "storage")
    MAX_UPLOAD_MB: int = 200
    AUTH_MODE: Literal["none", "basic"] = "none"
    DEFAULT_TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("STORAGE_DIR", mode="before")
    @classmethod
    def _normalize_storage_dir(cls, value: object) -> Path:
        if value in (None, ""):
            return ROOT / "storage"
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p

    @field_validator("DEFAULT_TEST_SIZE")
    @classmethod
    def _check_ratio(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("DEFAULT_TEST_SIZE 는 (0, 1) 범위여야 합니다.")
        return value

    @field_validator("MAX_UPLOAD_MB")
    @classmethod
    def _check_upload_mb(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_UPLOAD_MB 는 양수여야 합니다.")
        return value

    @property
    def datasets_dir(self) -> Path:
        return self.STORAGE_DIR / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.STORAGE_DIR / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.STORAGE_DIR / "predictions"

    @property
    def logs_dir(self) -> Path:
        return self.STORAGE_DIR / "logs"

    def ensure_dirs(self) -> None:
        """모든 하위 저장소 디렉터리를 생성 (idempotent)."""
        for d in (self.datasets_dir, self.models_dir, self.predictions_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
