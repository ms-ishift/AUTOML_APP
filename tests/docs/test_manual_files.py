"""``docs/manual/*.md`` 의 무결성 / 동기화 체크.

목적:
- 각 기능 페이지(01~07) 에 대응하는 매뉴얼 파일이 존재한다.
- 공통 문서(00_overview, troubleshooting, _hub_intro) 가 존재한다.
- 기대하는 섹션 키워드가 각 MD 에 포함돼 콘텐츠가 "빈 스캐폴드" 로 퇴화하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MANUAL_DIR = Path(__file__).resolve().parents[2] / "docs" / "manual"

REQUIRED_PAGE_KEYS = (
    "00_overview",
    "01_projects",
    "02_dataset_upload",
    "03_training",
    "04_results",
    "05_models",
    "06_prediction",
    "07_admin",
    "troubleshooting",
)

REQUIRED_PRIVATE_KEYS = ("_hub_intro",)

MIN_CHARS = 400  # 최소 길이 — 스캐폴드 퇴화 감지


@pytest.mark.parametrize("key", REQUIRED_PAGE_KEYS)
def test_public_manual_file_exists_and_has_body(key: str) -> None:
    path = MANUAL_DIR / f"{key}.md"
    assert path.is_file(), f"{path.name} 매뉴얼이 없습니다."
    text = path.read_text(encoding="utf-8")
    assert len(text) >= MIN_CHARS, (
        f"{path.name} 이(가) {MIN_CHARS}자 미만 — 콘텐츠가 퇴화했습니다."
    )


@pytest.mark.parametrize("key", REQUIRED_PRIVATE_KEYS)
def test_private_template_exists(key: str) -> None:
    path = MANUAL_DIR / f"{key}.md"
    assert path.is_file(), f"{path.name} 내부 템플릿이 없습니다."


@pytest.mark.parametrize(
    "key",
    ("01_projects", "02_dataset_upload", "03_training", "04_results", "05_models", "06_prediction", "07_admin"),
)
def test_page_manual_has_user_and_developer_sections(key: str) -> None:
    """페이지별 매뉴얼은 사용자/개발자 이중 섹션을 유지해야 한다."""
    text = (MANUAL_DIR / f"{key}.md").read_text(encoding="utf-8")
    assert "### 👤" in text, f"{key}.md 에 사용자 섹션(👤) 이 없습니다."
    assert "### 🛠" in text, f"{key}.md 에 개발자 섹션(🛠) 이 없습니다."
    # 트러블슈팅 표가 없으면 사용자 경험이 부족함.
    assert "자주 겪는 오류" in text or "자주 겪는 문제" in text, (
        f"{key}.md 에 오류 복구 섹션이 없습니다."
    )


def test_troubleshooting_covers_known_risks() -> None:
    text = (MANUAL_DIR / "troubleshooting.md").read_text(encoding="utf-8")
    # 릴리즈 이전 해결한 3대 재현 문제(R-001 libomp, DB seed FK, make run venv) 가
    # 모두 안내돼야 한다.
    assert "libomp" in text
    assert "FOREIGN KEY" in text or "seed" in text.lower()
    assert "make run" in text or "make doctor" in text


def test_no_stray_manual_files() -> None:
    """정의되지 않은 키가 섞이면 알람. 신규 파일을 추가했으면 위 상수에도 등록해야 한다."""
    all_md = {p.stem for p in MANUAL_DIR.glob("*.md")}
    expected = set(REQUIRED_PAGE_KEYS) | set(REQUIRED_PRIVATE_KEYS)
    unexpected = all_md - expected
    assert not unexpected, (
        f"등록되지 않은 매뉴얼 파일: {sorted(unexpected)}. "
        "tests/docs/test_manual_files.py 의 REQUIRED_* 상수에 추가하세요."
    )
