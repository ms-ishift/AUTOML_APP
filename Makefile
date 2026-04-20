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

.PHONY: help venv install install-dev samples test test-fast cov lint fmt ci run smoke bench plan-check clean doctor

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
