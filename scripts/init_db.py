"""DB 초기화 스크립트 (IMPLEMENTATION_PLAN §2.4).

사용::

    python scripts/init_db.py              # 테이블 생성 (없으면)
    python scripts/init_db.py --drop       # 기존 스키마 drop 후 재생성
    python scripts/init_db.py --seed       # 시스템 사용자 + 샘플 프로젝트 upsert
    python scripts/init_db.py --drop --seed

옵션은 조합 가능하며 멱등(idempotent) 실행을 지향한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 스크립트 단독 실행 시 프로젝트 루트를 sys.path 에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect  # noqa: E402

from repositories.base import Base, engine, session_scope  # noqa: E402
from repositories.models import (  # noqa: E402,F401
    SYSTEM_USER_ID,
    SYSTEM_USER_LOGIN_ID,
    SYSTEM_USER_NAME,
    SYSTEM_USER_ROLE,
    AuditLog,
    Dataset,
    Model,
    PredictionJob,
    Project,
    TrainingJob,
    User,
)
from utils.events import Event  # noqa: E402
from utils.log_utils import get_logger, log_event  # noqa: E402

logger = get_logger("scripts.init_db")


def _drop_all() -> None:
    logger.info("drop_all.started")
    Base.metadata.drop_all(engine)
    logger.info("drop_all.completed")


def _create_all() -> None:
    logger.info("create_all.started")
    Base.metadata.create_all(engine)
    tables = sorted(inspect(engine).get_table_names())
    logger.info(f"create_all.completed | tables={tables}")


def _seed_system_user() -> None:
    with session_scope() as session:
        user = session.get(User, SYSTEM_USER_ID)
        if user is None:
            user = User(
                user_id=SYSTEM_USER_ID,
                login_id=SYSTEM_USER_LOGIN_ID,
                user_name=SYSTEM_USER_NAME,
                role=SYSTEM_USER_ROLE,
            )
            session.add(user)
            logger.info("seed.system_user.created")
        else:
            logger.info("seed.system_user.exists")


def _seed_sample_project() -> None:
    """샘플 CSV 경로를 참조하는 예시 프로젝트 1건을 upsert."""
    sample_csv = ROOT / "samples" / "classification.csv"

    with session_scope() as session:
        exists = session.query(Project).filter(Project.project_name == "샘플: 붓꽃 분류").first()
        if exists is not None:
            logger.info("seed.sample_project.exists")
            return

        project = Project(
            project_name="샘플: 붓꽃 분류",
            description="init_db --seed 로 생성된 예시 프로젝트",
            owner_user_id=SYSTEM_USER_ID,
        )
        session.add(project)
        session.flush()
        log_event(
            logger,
            Event.PROJECT_CREATED,
            project_id=project.project_id,
            seeded=True,
            sample_csv=str(sample_csv),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoML MVP DB 초기화")
    parser.add_argument("--drop", action="store_true", help="기존 스키마 drop 후 재생성")
    parser.add_argument("--seed", action="store_true", help="시스템 사용자 + 샘플 프로젝트 upsert")
    args = parser.parse_args(argv)

    if args.drop:
        _drop_all()
    _create_all()

    if args.seed:
        _seed_system_user()
        _seed_sample_project()

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
