"""인앱 매뉴얼 컴포넌트 / 허브 페이지 테스트 (IMPLEMENTATION_PLAN §6.8).

- ``pages/components/help.py`` 의 로더·fallback 동작
- ``pages/00_manual.py`` 허브 페이지 렌더 (검색 필터 포함)
- 각 기능 페이지 상단에 도움말 expander 가 노출되는지의 스모크 체크
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pages.components import help as help_mod

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_HUB_PATH = str(PROJECT_ROOT / "pages" / "00_manual.py")


# ----------------------------------------------------- 로더 단위 테스트


def test_load_manual_returns_none_for_missing_key() -> None:
    assert help_mod.load_manual("this_key_does_not_exist_12345") is None


def test_load_manual_reads_existing_file() -> None:
    text = help_mod.load_manual("00_overview")
    assert text is not None
    assert "전체 흐름 개요" in text


def test_load_manual_rejects_path_traversal() -> None:
    assert help_mod.load_manual("../secrets") is None
    assert help_mod.load_manual("a/b") is None


def test_list_manual_keys_hides_private_files() -> None:
    keys = help_mod.list_manual_keys()
    assert "00_overview" in keys
    assert "troubleshooting" in keys
    # `_hub_intro` 는 `_` 접두사라 목록에 포함되면 안 된다.
    assert "_hub_intro" not in keys


def test_load_manual_cache_invalidates_on_mtime_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """파일 내용이 바뀌면 캐시가 풀려 새 내용이 반환돼야 한다."""
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    target = manual_dir / "unit.md"
    target.write_text("## first\n", encoding="utf-8")

    monkeypatch.setattr(help_mod, "MANUAL_DIR", manual_dir)
    help_mod._read_manual_text.clear()

    first = help_mod.load_manual("unit")
    assert first is not None and "first" in first

    # mtime_ns 가 바뀌도록 내용을 수정.
    import os
    import time

    time.sleep(0.01)
    target.write_text("## second\n", encoding="utf-8")
    os.utime(target, None)

    second = help_mod.load_manual("unit")
    assert second is not None and "second" in second


# ----------------------------------------------------- 허브 페이지 스모크


def _new_hub() -> AppTest:
    return AppTest.from_file(MANUAL_HUB_PATH, default_timeout=15)


def test_manual_hub_renders_without_db() -> None:
    """DB 가 없어도 매뉴얼 허브는 열려야 한다 (진입 장벽 제거)."""
    at = _new_hub().run()
    assert not at.exception
    body = " ".join(e.value for e in at.markdown)
    assert "전체 흐름 개요" in body  # 00_overview.md 의 첫 헤더
    assert "트러블슈팅" in body  # TOC 라벨


def test_manual_hub_search_filters_sections() -> None:
    at = _new_hub().run()
    assert not at.exception
    text_input = at.text_input(key="manual_search_query")
    text_input.set_value("libomp").run()
    assert not at.exception
    body = " ".join(e.value for e in at.markdown)
    assert "libomp" in body
    # 전혀 관련 없는 섹션 본문은 더 이상 보이지 않아야 한다.
    # 00_overview 는 libomp 단어를 쓰지 않는다.
    assert "용어 미니사전" not in body


def test_manual_hub_empty_search_shows_all_sections() -> None:
    at = _new_hub().run()
    assert not at.exception
    text_input = at.text_input(key="manual_search_query")
    text_input.set_value("").run()
    assert not at.exception
    body = " ".join(e.value for e in at.markdown)
    # 최소 3개 주요 섹션이 함께 보여야 한다.
    assert "전체 흐름 개요" in body
    assert "프로젝트 관리" in body
    assert "자주 겪는 문제" in body


# ----------------------------------------------------- 기능 페이지 연결


def test_help_expander_present_on_projects_page(
    tmp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("utils.db_utils.is_db_initialized", lambda: False)
    page_path = str(PROJECT_ROOT / "pages" / "01_projects.py")
    at = AppTest.from_file(page_path, default_timeout=15).run()
    assert not at.exception
    # render_help 는 실패 시 아무 것도 그리지 않으므로 expander 가 최소 1개 존재해야 한다.
    assert any("이 페이지 도움말" in (e.label or "") for e in at.expander)
