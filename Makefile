# AutoML Streamlit MVP — 개발 명령 모음.
# IMPLEMENTATION_PLAN 0.2 참조.

.PHONY: help install install-dev samples test test-fast cov lint fmt ci run smoke bench plan-check clean

help:
	@echo "Targets:"
	@echo "  install      - requirements.txt 설치"
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
	@echo "  plan-check   - IMPLEMENTATION_PLAN.md 의 [~] 개수 점검"
	@echo "  clean        - __pycache__/캐시 정리"

install:
	pip install -r requirements.txt

install-dev: install
	pre-commit install || echo "pre-commit 미설치: pip install pre-commit 후 다시 실행"

samples:
	python scripts/generate_samples.py

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

cov:
	pytest --cov=ml --cov=services --cov-report=term-missing

lint:
	ruff check .
	mypy .

fmt:
	ruff check . --fix
	black .

ci: lint cov
	@echo "[ci] quality gate OK (ruff + mypy + pytest --cov fail_under=60)"

run:
	python scripts/sync_streamlit_config.py
	streamlit run app.py

smoke:
	pytest -q -m "not slow"

bench:
	python scripts/perf_bench.py

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
