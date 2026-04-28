# AutoML Streamlit MVP — 개발 명령 모음.
# IMPLEMENTATION_PLAN 0.2 참조.
#
# venv-aware: .venv 가 존재하면 그쪽 바이너리를 우선 사용한다.
# 따라서 `source .venv/bin/activate` 없이도 `make run` / `make test` 등이
# 동작한다. venv 가 없으면 시스템 PATH 의 python/pytest/streamlit 으로 fallback.

VENV := .venv
VENV_BIN := $(CURDIR)/$(VENV)/bin
PYTHON := $(shell [ -x $(VENV_BIN)/python ] && echo $(VENV_BIN)/python || command -v python3 || echo python)
ifneq ($(wildcard $(VENV_BIN)),)
export PATH := $(VENV_BIN):$(PATH)
endif

.PHONY: help venv install install-dev samples test test-fast cov lint fmt ci run run-logs docker-assert docker-build docker-run smoke bench plan-check clean doctor

help:
	@echo "Targets:"
	@echo "  venv         - .venv 가상환경 생성 (python3.11 기준)"
	@echo "  install      - requirements.txt 설치 (필요 시 venv 먼저)"
	@echo "  install-dev  - install + pre-commit 훅 설치"
	@echo "  samples      - samples/*.csv 생성 (sklearn)"
	@echo "  test         - pytest 실행 (전체, slow 포함)"
	@echo "  test-fast    - pytest -m 'not slow' (slow 제외, ~25s)"
	@echo "  cov          - pytest --cov=ml --cov=services (fail_under=60)"
	@echo "  lint         - ruff + mypy 검사"
	@echo "  fmt          - ruff --fix + black"
	@echo "  ci           - lint + cov 통합 게이트 (§7.5)"
	@echo "  run          - sync config + streamlit 실행"
	@echo "  run-logs     - app 로그 실시간 보기 (tail -F, 별 터미널에서 make run 과 병행)"
	@echo "  docker-build - Docker 이미지 빌드 (기본 태그: automl-app:latest)"
	@echo "  docker-run   - Docker 컨테이너 실행 (8501 포트 노출)"
	@echo "  bench        - NFR-003 성능 벤치 (scripts/perf_bench.py)"
	@echo "  doctor       - 실행 환경 점검 (venv / python / streamlit 해석)"
	@echo "  plan-check   - IMPLEMENTATION_PLAN.md 의 [~] 개수 점검"
	@echo "  clean        - __pycache__/캐시 정리"

venv:
	@if [ ! -x $(VENV_BIN)/python ]; then \
		echo "[venv] creating $(VENV) via python3.11"; \
		python3.11 -m venv $(VENV) || python3 -m venv $(VENV); \
	else \
		echo "[venv] $(VENV) already exists"; \
	fi

install: venv
	$(VENV_BIN)/pip install -r requirements.txt

install-dev: install
	$(VENV_BIN)/pre-commit install || echo "pre-commit 미설치: pip install pre-commit 후 다시 실행"

samples:
	$(PYTHON) scripts/generate_samples.py

test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q -m "not slow"

cov:
	$(PYTHON) -m pytest --cov=ml --cov=services --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy .

fmt:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m black .

ci: lint cov
	@echo "[ci] quality gate OK (ruff + mypy + pytest --cov fail_under=60)"

run:
	$(PYTHON) scripts/sync_streamlit_config.py
	$(PYTHON) -m streamlit run app.py

# storage 위치는 .env 의 STORAGE_DIR 등으로 바뀔 수 있으므로 settings 에서 경로를 해석한다.
# LOGFILE=/path/to/app.log 으로 고정 경로를 줄 수 있다.
run-logs:
	@set -e; \
	if [ -n "$(LOGFILE)" ]; then log="$(LOGFILE)"; else \
		log="$$($(PYTHON) -c 'from config.settings import settings; print(settings.logs_dir / "app.log")')"; \
	fi; \
	echo "[run-logs] tail -F $$log  (끄려면 Ctrl+C)"; \
	exec tail -F "$$log"

# Docker 실행 편의 타깃.
# IMAGE_TAG=your-tag PORT=8502 로 오버라이드 가능.
# docker 가 없을 때(예: make docker-build → "docker: command not found") 는
# https://docs.docker.com/get-docker/ 에서 설치하거나, macOS 는 Docker Desktop 설치·실행 후 터미널을 다시 연다.
docker-assert:
	@command -v docker >/dev/null 2>&1 || { \
		echo "ERROR: docker CLI not in PATH. Install Docker (e.g. Docker Desktop) and try again: https://docs.docker.com/get-docker/"; \
		echo "  macOS: open Docker.app once so docker is on PATH, then: make docker-build"; \
		exit 127; \
	}

docker-build: docker-assert
	@tag="$${IMAGE_TAG:-automl-app:latest}"; \
	echo "[docker-build] building $$tag"; \
	docker build -t "$$tag" .

docker-run: docker-assert
	@set -e; \
	tag="$${IMAGE_TAG:-automl-app:latest}"; \
	port="$${PORT:-8501}"; \
	echo "[docker-run] running $$tag on localhost:$$port"; \
	docker run --rm -p "$$port:8501" "$$tag"

smoke:
	$(PYTHON) -m pytest -q -m "not slow"

bench:
	$(PYTHON) scripts/perf_bench.py

doctor:
	@echo "[doctor] CURDIR     = $(CURDIR)"
	@echo "[doctor] VENV_BIN   = $(VENV_BIN) (exists=$(if $(wildcard $(VENV_BIN)),yes,no))"
	@echo "[doctor] PYTHON     = $(PYTHON)"
	@$(PYTHON) --version
	@echo "[doctor] streamlit  = $$($(PYTHON) -c 'import streamlit,inspect; print(streamlit.__file__)' 2>/dev/null || echo 'MISSING — run: make install')"
	@echo "[doctor] pytest     = $$($(PYTHON) -c 'import pytest; print(pytest.__file__)' 2>/dev/null || echo 'MISSING — run: make install')"
	@echo "[doctor] optional backends (xgboost / lightgbm):"
	@$(PYTHON) -c "from ml.registry import optional_backends_status; \
	[print(f'  - {s.name:8} {\"OK\" if s.available else \"SKIP — \" + s.reason}') for s in optional_backends_status()]" \
		|| echo "  (registry 로드 실패: make install 후 재시도)"
	@echo "[doctor] PATH       = $$PATH"

plan-check:
	@count=$$(grep -cE '^[[:space:]]*-[[:space:]]+\[~\]' IMPLEMENTATION_PLAN.md || true); \
	if [ "$$count" -gt 1 ]; then \
		echo "WARN: IMPLEMENTATION_PLAN.md 에 진행중([~]) 항목이 $$count개 (권장 1개 이하)"; \
		grep -nE '^[[:space:]]*-[[:space:]]+\[~\]' IMPLEMENTATION_PLAN.md || true; \
	else \
		echo "OK: in-progress=$$count"; \
	fi

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
