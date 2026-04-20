#!/usr/bin/env bash
# 코드/설정 파일을 수정하면서 IMPLEMENTATION_PLAN.md 를 함께 스테이지하지 않았으면 경고.
# 차단하지 않고 경고만 출력 (개발 편의).
set -euo pipefail

staged=$(git diff --cached --name-only || true)

has_code=$(echo "$staged" | grep -E '\.(py|toml|yaml|yml|sh)$|(^|/)Makefile$' || true)
has_plan=$(echo "$staged" | grep -E '^IMPLEMENTATION_PLAN\.md$' || true)

if [ -n "$has_code" ] && [ -z "$has_plan" ]; then
  echo ""
  echo "[plan-staged] WARN: 코드/설정 변경이 있으나 IMPLEMENTATION_PLAN.md 가 함께 스테이지되지 않았습니다."
  echo "              진행 상태([~]/[x])와 진행 로그를 갱신해 같은 커밋에 포함하는 것을 권장합니다."
  echo ""
fi

exit 0
