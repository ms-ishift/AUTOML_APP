"""ml/type_inference.py 검증 — §9.2 (FR-056).

커버 범위:
- detect_datetime_columns: datetime64 / ISO 문자열 / 혼합 파싱 / 수치 문자열 배제
- detect_bool_columns: 네이티브 / int 0-1 / Y-N / yes-no / true-false / 혼합 토큰
- detect_highcard_categorical: nunique 축 / unique_ratio 축 / 결측 존재 시
- skew_report: 치우친 분포 감지 / 상수/비수치 스킵 / threshold 경계
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.type_inference import (
    DATETIME_PARSE_SUCCESS_RATIO,
    detect_bool_columns,
    detect_datetime_columns,
    detect_highcard_categorical,
    skew_report,
)

# ---------------------------------------------------------------- datetime


def test_detect_datetime_columns_native_datetime64() -> None:
    df = pd.DataFrame(
        {
            "signup_at": pd.date_range("2024-01-01", periods=5, freq="D"),
            "age": [10, 20, 30, 40, 50],
            "city": ["seoul", "busan", "seoul", "busan", "seoul"],
        }
    )
    assert detect_datetime_columns(df) == ["signup_at"]


def test_detect_datetime_columns_iso_strings() -> None:
    df = pd.DataFrame(
        {
            "created": [
                "2024-01-01",
                "2024-02-15",
                "2024-03-20",
                "2024-04-05",
                "2024-05-11",
            ],
            "name": ["a", "b", "c", "d", "e"],
        }
    )
    assert detect_datetime_columns(df) == ["created"]


def test_detect_datetime_columns_partial_parse_failure() -> None:
    # 19/20 성공 (95% 충족) → 포함
    values = [f"2024-01-{d:02d}" for d in range(1, 20)] + ["not-a-date"]
    # 15/20 성공 (75% 미달) → 제외
    bad_values = [f"2024-02-{d:02d}" for d in range(1, 16)] + ["x"] * 5
    df = pd.DataFrame({"ok": values, "bad": bad_values})
    detected = detect_datetime_columns(df)
    assert "ok" in detected
    assert "bad" not in detected
    # 전역 상수 회귀 방지
    assert DATETIME_PARSE_SUCCESS_RATIO == 0.95


def test_detect_datetime_columns_numeric_strings_excluded() -> None:
    # 수치 문자열은 to_datetime 이 unit 해석으로 오인할 수 있어 제외
    df = pd.DataFrame({"ids": ["1", "42", "100", "7", "9"]})
    assert detect_datetime_columns(df) == []


def test_detect_datetime_columns_all_nan_column_skipped() -> None:
    df = pd.DataFrame({"empty": [None, None, None], "a": [1, 2, 3]})
    assert detect_datetime_columns(df) == []


def test_detect_datetime_columns_empty_dataframe() -> None:
    df = pd.DataFrame(
        {"x": pd.Series([], dtype="datetime64[ns]"), "y": pd.Series([], dtype=object)}
    )
    # 행은 0 이지만 datetime64 dtype 은 여전히 감지
    assert detect_datetime_columns(df) == ["x"]


# ---------------------------------------------------------------- bool


def test_detect_bool_columns_native_bool() -> None:
    df = pd.DataFrame({"is_admin": [True, False, True], "age": [1, 2, 3]})
    assert detect_bool_columns(df) == ["is_admin"]


def test_detect_bool_columns_int_0_1() -> None:
    df = pd.DataFrame(
        {
            "has_child": [0, 1, 1, 0, 1],
            "age": [10, 20, 30, 40, 50],
            "has_nan": [0, 1, None, 1, 0],  # pandas 가 float 로 올릴 것 → int 아님
        }
    )
    detected = detect_bool_columns(df)
    assert "has_child" in detected
    assert "age" not in detected
    # NaN 이 섞인 0/1 은 float dtype 이 되어 is_integer_dtype False → 제외됨.
    assert "has_nan" not in detected


def test_detect_bool_columns_yn_tokens() -> None:
    df = pd.DataFrame(
        {
            "flag_yn": ["Y", "N", "Y", "N"],
            "flag_yesno": ["yes", "no", "Yes", "NO"],
            "flag_tf": ["true", "false", "True", "FALSE"],
            "city": ["seoul", "busan", "seoul", "busan"],
        }
    )
    detected = detect_bool_columns(df)
    assert set(detected) == {"flag_yn", "flag_yesno", "flag_tf"}
    assert "city" not in detected


def test_detect_bool_columns_mixed_tokens_rejected() -> None:
    # 토큰 집합에 없는 값이 섞이면 bool 후보 아님
    df = pd.DataFrame(
        {
            "mixed": ["Y", "N", "maybe", "N"],
            "mostly_bool_but_one_off": ["true", "false", "unknown", "true"],
        }
    )
    assert detect_bool_columns(df) == []


def test_detect_bool_columns_all_nan_skipped() -> None:
    df = pd.DataFrame({"x": [None, None, None], "y": [True, False, True]})
    assert detect_bool_columns(df) == ["y"]


# ---------------------------------------------------------------- highcard


def test_detect_highcard_categorical_nunique_axis() -> None:
    # 60 행, "many" 는 60 unique → nunique > 50 → highcard
    df = pd.DataFrame(
        {
            "many": [f"id_{i}" for i in range(60)],
            "few": ["a", "b", "c"] * 20,
        }
    )
    detected = detect_highcard_categorical(
        df, ["many", "few"], nunique_threshold=50, unique_ratio_threshold=0.9
    )
    assert detected == ["many"]


def test_detect_highcard_categorical_ratio_axis() -> None:
    # 10 행, "ratioful" nunique=5 → 0.5 > 0.3 임계 → highcard (nunique 는 50 미만)
    df = pd.DataFrame(
        {
            "ratioful": ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"],
            "dominant": ["a"] * 9 + ["b"],
        }
    )
    detected = detect_highcard_categorical(
        df, ["ratioful", "dominant"], nunique_threshold=50, unique_ratio_threshold=0.3
    )
    assert detected == ["ratioful"]


def test_detect_highcard_categorical_skip_missing_columns() -> None:
    df = pd.DataFrame({"a": ["x"] * 20})  # 20 행, 1 unique → highcard 아님
    # "nonexistent" 는 조용히 스킵, "a" 는 기본 임계값으로는 highcard 아님.
    assert detect_highcard_categorical(df, ["a", "nonexistent"]) == []


def test_detect_highcard_categorical_empty_cols_iter() -> None:
    df = pd.DataFrame({"a": list(range(100))})
    assert detect_highcard_categorical(df, []) == []


# ---------------------------------------------------------------- skew


def test_skew_report_detects_right_skewed() -> None:
    # log-normal 유사 분포: 대다수 작은 값 + 극단 큰 값 1개 → |skew| 큼
    df = pd.DataFrame(
        {
            "skewed": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1000],
            "normalish": list(range(1, 11)),
        }
    )
    report = skew_report(df, ["skewed", "normalish"], abs_skew_threshold=1.0)
    assert "skewed" in report
    assert abs(report["skewed"]) >= 1.0
    # normalish 는 대칭에 가까워 threshold 미달
    assert "normalish" not in report


def test_skew_report_skip_constant_and_non_numeric() -> None:
    df = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5, 5],
            "text": ["a", "b", "c", "d", "e"],
            "skewed": [0, 0, 0, 0, 100],
        }
    )
    report = skew_report(df, ["constant", "text", "skewed", "missing"])
    # 상수 컬럼은 skew=NaN → 스킵, 비수치/존재X 도 스킵
    assert "constant" not in report
    assert "text" not in report
    assert "missing" not in report
    assert "skewed" in report


def test_skew_report_threshold_boundary() -> None:
    # 낮은 threshold 로는 normalish 도 살짝 걸릴 수 있고 엄격한 threshold 로는 제외.
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    df = pd.DataFrame({"flat": vals})
    # 완전 대칭 분포이므로 어떤 threshold 에서도 미포함
    assert skew_report(df, ["flat"], abs_skew_threshold=0.1) == {}


def test_skew_report_rounds_values() -> None:
    df = pd.DataFrame({"skewed": [0, 0, 0, 0, 100]})
    report = skew_report(df, ["skewed"], abs_skew_threshold=0.5)
    assert "skewed" in report
    v = report["skewed"]
    # 소수 6자리로 반올림되었는지 (원본은 float 정밀도)
    assert round(v, 6) == v
