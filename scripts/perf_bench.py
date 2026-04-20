"""성능 벤치 스크립트 (IMPLEMENTATION_PLAN §7.2, NFR-003).

검증 시나리오:
- 5만 행 CSV 업로드 + 프로파일 + 미리보기 (목표: 10초 이내)
- 저장된 모델로 단건 예측 (목표: 3초 이내)

실행 예::

    python scripts/perf_bench.py              # 전체 시나리오
    python scripts/perf_bench.py --rows 50000 # 업로드 시 행 수 조정

환경은 격리된 임시 ``STORAGE_DIR`` 을 사용하므로 현재 ``db/app.db`` 를 오염시키지 않는다.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class BenchResult:
    name: str
    elapsed_s: float
    target_s: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.elapsed_s <= self.target_s

    def render(self) -> str:
        status = "OK" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name:<36s} {self.elapsed_s:6.3f}s "
            f"(target {self.target_s:.1f}s) {self.detail}"
        )


def _prepare_env() -> Path:
    """임시 storage 경로 + SQLite DB 로 환경을 격리한다."""
    tmp_root = Path(tempfile.mkdtemp(prefix="automl-bench-"))
    os.environ["STORAGE_DIR"] = str(tmp_root)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_root}/bench.db"
    os.environ["AUTH_MODE"] = "none"
    return tmp_root


def _bootstrap_db() -> int:
    """스키마 생성 + 시스템 사용자 시드. project_id 를 반환."""
    import importlib

    import config.settings as cfg

    importlib.reload(cfg)
    import repositories.base as base

    importlib.reload(base)
    from repositories import project_repository
    from repositories.base import Base, engine, session_scope
    from repositories.models import (
        SYSTEM_USER_ID,
        SYSTEM_USER_LOGIN_ID,
        SYSTEM_USER_NAME,
        SYSTEM_USER_ROLE,
        User,
    )

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    with session_scope() as session:
        if session.get(User, SYSTEM_USER_ID) is None:
            session.add(
                User(
                    user_id=SYSTEM_USER_ID,
                    login_id=SYSTEM_USER_LOGIN_ID,
                    user_name=SYSTEM_USER_NAME,
                    role=SYSTEM_USER_ROLE,
                )
            )
            session.flush()
        proj = project_repository.insert(
            session,
            project_name="bench",
            description="perf bench",
            owner_user_id=SYSTEM_USER_ID,
        )
        session.flush()
        return int(proj.project_id)


class _Uploaded:
    """``_UploadedLike`` duck-type — CSV 바이트를 그대로 제공."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _make_csv_bytes(rows: int) -> bytes:
    """회귀형 피처 12개 + 타깃 1개. pandas 로 생성해 바이트로 반환."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_num = 8
    n_cat = 4
    data: dict[str, Any] = {}
    for i in range(n_num):
        data[f"num_{i}"] = rng.normal(0, 1, size=rows).round(4)
    cats = ["A", "B", "C", "D"]
    for i in range(n_cat):
        data[f"cat_{i}"] = rng.choice(cats, size=rows)
    coef = rng.normal(0, 1, size=n_num)
    target = np.zeros(rows)
    for i in range(n_num):
        target += coef[i] * data[f"num_{i}"]
    target += rng.normal(0, 0.1, size=rows)
    data["target"] = target.round(4)

    df = pd.DataFrame(data)
    from io import BytesIO

    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def bench_upload(project_id: int, rows: int) -> list[BenchResult]:
    """upload_dataset + profile + preview 왕복 측정."""
    from services import dataset_service

    payload = _make_csv_bytes(rows)
    uploaded = _Uploaded(f"bench_{rows}.csv", payload)

    t0 = time.perf_counter()
    ds = dataset_service.upload_dataset(project_id, uploaded)
    t_upload = time.perf_counter() - t0

    t0 = time.perf_counter()
    profile = dataset_service.get_dataset_profile(ds.id)
    t_profile = time.perf_counter() - t0

    preview_times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        dataset_service.preview_dataset(ds.id, n=50)
        preview_times.append(time.perf_counter() - t0)
    t_preview = statistics.median(preview_times)

    total = t_upload + t_profile + t_preview
    results = [
        BenchResult(
            f"upload_dataset ({rows:,} rows)",
            t_upload,
            10.0,
            detail=f"rows={ds.row_count}, cols={ds.column_count}",
        ),
        BenchResult(
            "get_dataset_profile",
            t_profile,
            2.0,
            detail=f"n_cols={len(profile.columns)}",
        ),
        BenchResult(
            "preview_dataset (median of 3)",
            t_preview,
            1.0,
            detail=f"min={min(preview_times):.3f}s",
        ),
        BenchResult(
            f"upload+profile+preview total ({rows:,} rows)",
            total,
            10.0,
        ),
    ]
    return results


def bench_predict_single(project_id: int) -> list[BenchResult]:
    """작은 데이터셋으로 학습 1건 → predict_single 반복 측정."""
    from io import BytesIO

    import numpy as np
    import pandas as pd

    from ml.schemas import TrainingConfig
    from services import dataset_service, prediction_service, training_service

    rng = np.random.default_rng(7)
    rows = 800
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rows).round(4),
            "x2": rng.normal(0, 1, rows).round(4),
            "cat": rng.choice(["A", "B", "C"], rows),
            "target": rng.integers(0, 2, rows),
        }
    )
    buf = BytesIO()
    df.to_csv(buf, index=False)
    ds = dataset_service.upload_dataset(project_id, _Uploaded("bench_predict.csv", buf.getvalue()))

    cfg = TrainingConfig(
        dataset_id=ds.id,
        task_type="classification",
        target_column="target",
        test_size=0.2,
    )
    t0 = time.perf_counter()
    result = training_service.run_training(cfg)
    t_train = time.perf_counter() - t0
    best = next((r for r in result.rows if r.status == "success"), None)
    assert best is not None, "학습 실패 — 벤치 중단"
    assert best.model_id is not None
    model_id = best.model_id

    payload = {"x1": 0.1, "x2": 0.2, "cat": "A"}
    single_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        prediction_service.predict_single(model_id, payload)
        single_times.append(time.perf_counter() - t0)

    return [
        BenchResult(
            "train_one_algo (logistic_regression, 800 rows)",
            t_train,
            15.0,
            detail="warmup only",
        ),
        BenchResult(
            "predict_single (cold)",
            single_times[0],
            3.0,
            detail=f"n_calls={len(single_times)}",
        ),
        BenchResult(
            "predict_single (median warm)",
            statistics.median(single_times[1:]),
            3.0,
            detail=f"min={min(single_times):.3f}s max={max(single_times):.3f}s",
        ),
    ]


def _summarize(results: list[BenchResult]) -> int:
    print()
    print("=" * 72)
    for r in results:
        print(r.render())
    print("=" * 72)
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"FAIL: {len(failed)}개 지표가 목표를 초과")
        return 1
    print("ALL OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AutoML 성능 벤치 (§7.2 NFR-003)")
    ap.add_argument("--rows", type=int, default=50_000, help="업로드 행 수")
    ap.add_argument("--skip-predict", action="store_true", help="예측 벤치 스킵")
    ns = ap.parse_args()

    tmp_root = _prepare_env()
    print(f"[setup] STORAGE_DIR={tmp_root}")
    project_id = _bootstrap_db()
    print(f"[setup] project_id={project_id}")

    results: list[BenchResult] = []
    results.extend(bench_upload(project_id, ns.rows))
    if not ns.skip_predict:
        results.extend(bench_predict_single(project_id))
    return _summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
