"""`.env` 의 MAX_UPLOAD_MB 를 `.streamlit/config.toml` 의 [server].maxUploadSize 에 동기화.

IMPLEMENTATION_PLAN 0.2b. `Makefile` 의 run 타깃이 호출한다.
settings 객체를 쓰지 않는 이유: 이 스크립트는 의존 최소화를 위해 python-dotenv 만 사용한다.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
ENV_PATH = ROOT / ".env"

DEFAULT_MAX_MB = 200
DEFAULT_CONFIG = """# 이 파일의 maxUploadSize 는 scripts/sync_streamlit_config.py 가
# .env 의 MAX_UPLOAD_MB 값으로 덮어씁니다.

[server]
maxUploadSize = 200
headless = true

[theme]
base = "light"
"""


def _load_env_max_mb() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[sync] python-dotenv 미설치: 기본값 사용", file=sys.stderr)
        return DEFAULT_MAX_MB

    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    raw = os.getenv("MAX_UPLOAD_MB")
    if not raw:
        return DEFAULT_MAX_MB
    try:
        return int(raw)
    except ValueError:
        print(f"[sync] MAX_UPLOAD_MB 값이 정수가 아님: {raw!r}. 기본값 사용", file=sys.stderr)
        return DEFAULT_MAX_MB


def _write_config(max_mb: int) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")

    content = CONFIG_PATH.read_text(encoding="utf-8")
    new_content, replaced = re.subn(
        r"(?m)^maxUploadSize\s*=\s*\d+",
        f"maxUploadSize = {max_mb}",
        content,
    )
    if replaced == 0:
        new_content = content.rstrip() + f"\n\n[server]\nmaxUploadSize = {max_mb}\n"

    CONFIG_PATH.write_text(new_content, encoding="utf-8")


def main() -> None:
    max_mb = _load_env_max_mb()
    _write_config(max_mb)
    print(f"[sync] MAX_UPLOAD_MB={max_mb} -> {CONFIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
