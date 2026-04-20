#!/usr/bin/env bash
# AutoML Streamlit MVP 개발 실행 스크립트.
# .env 의 MAX_UPLOAD_MB 를 .streamlit/config.toml 에 동기화한 뒤 Streamlit 실행.
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/sync_streamlit_config.py
exec streamlit run app.py
