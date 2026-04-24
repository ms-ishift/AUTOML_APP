# AutoML Streamlit MVP — 구현 계획 (체크리스트)

**문서 버전:** 0.4
**최근 갱신:** 2026-04-23 (0.4 추가: §9 전처리 고도화 계획 수립 — FR-055~058, PreprocessingConfig + datetime/bool/고카디널리티 + class_weight/SMOTE + 피처 변환 미리보기. 0.3 기준 유지: DTO 분리, 메시지/이벤트 카탈로그, AUTH 정책, 업로드 한도 동기화, 네비 방식 고정, 비교표 규격, Makefile/pre-commit, 아티팩트 저장 순서)

**참조 문서:**
- 요구사항: [`AutoML_Streamlit_MVP.md`](./AutoML_Streamlit_MVP.md)
- 아키텍처: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- 규칙: `.cursor/rules/*`, 워크플로우: `.cursor/skills/automl-feature-dev/SKILL.md`

**진행 원칙:**
1. **상향식 구현**: 유틸 → 인프라 → ML → Service → UI 순으로 쌓아 올린다.
2. **단계별 체크인**: 각 단계 끝에서 체크리스트 전부 ✅가 되어야 다음 단계로 진입.
3. **테스트 동반**: 의미 있는 로직에는 반드시 `tests/` 짝꿍 테스트 추가.
4. **FR 추적**: 각 작업은 요구사항의 FR 번호에 매핑.
5. **계획서가 단일 진실(Single Source of Progress)**: 작업 시작 전 계획서를 먼저 읽고, 작업 종료 시 반드시 체크박스와 진행 로그를 갱신한다. 스코프 변경은 **코드보다 계획서를 먼저** 수정한다.

**진행 상태 표기**: `[ ]` 미착수 · `[~]` 진행중 · `[x]` 완료 · `[!]` 보류/이슈

---

## 계획서 유지 규칙

이 계획서는 작업 **중에도 계속 수정**하는 살아있는 문서다.

### 체크박스 갱신 타이밍

| 시점 | 동작 |
|------|------|
| 작업 시작 | 해당 항목을 `[ ]` → `[~]`로 변경. 동시에 오직 1개만 `[~]` 유지 권장. |
| 작업 완료 | `[~]` → `[x]`. 수용 기준을 실제로 검증한 뒤 체크. 미검증 체크 금지. |
| 블록 발생 | `[~]` → `[!]` + 바로 아래 줄에 `> BLOCKED: <이유> / <후속조치>` 인용 추가. |
| 스코프 변경 | 항목을 추가/삭제하기 전 본 규칙서 아래 "변경 이력"에 기록. |

### 진행 로그 포맷 (문서 최하단 섹션)

```
YYYY-MM-DD | 단계 X.Y | started|completed|blocked | <한 줄 메모>
```

- 하루 여러 건이면 여러 줄. 시간 순서로 append (삽입 금지).
- 실패/재시도는 별도 줄로 남긴다 (덮어쓰지 않음).

### 스코프 변경 규칙

새 FR을 발견했거나 파일/단계를 추가해야 하는 경우:

1. **먼저 계획서에 항목 추가** (적절한 단계/하위절에).
2. **변경 이력 섹션**에 `added`/`removed`/`moved` 기록.
3. **그다음 구현**. 역순(코드 먼저) 금지.

### Definition of Done (모든 항목 공통)

> **템플릿 체크리스트** — 각 작업 단위(§x.y)를 `[x]` 로 닫기 전에 아래를 전부 만족하는지 스스로 점검한다. 루트의 `[ ]` 는 단일 실행 아이템이 아니라 반복 적용되는 기준이므로 `plan-check` 진행중 카운트에서 제외된다 (☐ 표식 사용).

- ☐ 코드 작성 완료 및 해당 모듈 단위 테스트 통과 (`pytest <path> -q`)
- ☐ `ruff check <path>` / `mypy <path>` 0 에러 (새로 추가한 파일 범위)
- ☐ 레이어 경계 준수 (UI↛Repo/ML 직접 호출 없음, ML↛Streamlit/SQLAlchemy 없음)
- ☐ 관련 FR 번호가 docstring 또는 모듈 헤더에 기재
- ☐ 설계와 달라진 부분은 `ARCHITECTURE.md` 반영
- ☐ 계획서 체크박스 + 진행 로그 갱신
- ☐ 단계 끝에서 논리적 단위로 커밋 (제안 prefix: `feat:`, `chore:`, `test:`, `docs:`, `refactor:`)

---

## 단계 0. 프로젝트 부트스트랩

*MVP 실행 전 스캐폴드 생성. 모든 후속 단계가 이 기반 위에 쌓인다.*

### 0.1 디렉터리 스캐폴드

- [x] 루트 패키지 골격 생성
  - [x] `app.py` (빈 진입점)
  - [x] `pages/__init__.py`, `pages/components/__init__.py`
  - [x] `services/__init__.py`
  - [x] `ml/__init__.py`
  - [x] `repositories/__init__.py`
  - [x] `utils/__init__.py`
  - [x] `config/__init__.py`
  - [x] `tests/__init__.py`, `tests/ml/`, `tests/services/`, `tests/repositories/`, `tests/utils/`
- [x] 저장소/DB 폴더 + `.gitkeep`
  - [x] `storage/datasets/.gitkeep`
  - [x] `storage/models/.gitkeep`
  - [x] `storage/predictions/.gitkeep`
  - [x] `storage/logs/.gitkeep`
  - [x] `db/.gitkeep`

### 0.2 개발 환경

- [x] `python3.11 -m venv .venv` 후 `pip install -r requirements.txt` (현 시점은 최소 의존만 설치: scikit-learn, pandas, python-dotenv, streamlit)
- [x] `cp .env.example .env`
- [x] `ruff check .` / `black .` / `pytest` 통과 (단계 1 완료 시점 확인; `mypy`는 단계 2 이후 타입 보강과 함께)
  > NOTE: 아직 lint/type 툴을 설치하지 않았다. 단계 1 시작 시 `make install` 로 `requirements.txt` 전체 설치 후 이 항목을 [x] 로 갱신한다.
- [x] `scripts/dev_run.sh` 작성 (`streamlit run app.py`) + `chmod +x`
- [x] **Makefile** 추가 — `make install / install-dev / samples / test / lint / fmt / run / smoke / plan-check / clean`
  - [x] `plan-check`: `IMPLEMENTATION_PLAN.md` 의 `[~]` 개수가 1개 이하인지 검사 (장기 진행중 방지)
- [x] **pre-commit 훅** (`.pre-commit-config.yaml`)
  - [x] `ruff`, `black`, `mypy`
  - [x] 커스텀 훅 `plan-staged` (`scripts/check_plan_staged.sh`): 코드 변경이 있으면 `IMPLEMENTATION_PLAN.md` 도 같은 커밋에 포함되었는지 경고

### 0.2a Streamlit 네비게이션 방식 결정 (고정)

- [x] **선택 방식: `pages/` 디렉터리 자동 멀티페이지** (Streamlit 기본)
  - 근거: 설정 0에 가까움, 파일 단위 라우팅, MVP 규모에 충분
  - 제약: 페이지 순서는 파일명 prefix (`01_~07_`) 로 고정
- [x] `st.navigation` 은 **8단계(인증 도입) 이후 재검토** (로그인 분기 처리가 필요해질 때)
- [x] 페이지 간 이동은 `st.switch_page("pages/04_results.py")` 만 사용 (URL 하드코딩 금지) — 규칙 `.cursor/rules/automl-project.mdc` 반영 완료

### 0.2b `.streamlit/config.toml` 부트스트랩

- [x] `.streamlit/config.toml` 생성
  - [x] `[server] maxUploadSize = <settings.MAX_UPLOAD_MB>` (settings와 동기화)
  - [x] `[theme]` 기본 테마 지정 (`base = "light"`)
- [x] `scripts/sync_streamlit_config.py` — `.env` 의 `MAX_UPLOAD_MB` 를 읽어 config.toml 에 반영
- [x] `make run` 이 내부적으로 `sync_streamlit_config.py` 를 먼저 실행하도록 연결 (`dev_run.sh` 도 동일 순서)
- [x] 테스트: `.env` 의 값을 `350 → 200` 으로 토글해 config.toml 이 갱신됨을 확인

### 0.3 샘플 데이터 준비

*단계 3/4 테스트와 단계 6 수용 검증에 모두 필요하므로 먼저 확보한다.*

- [x] `samples/classification.csv` — 소형 분류용 (iris, 150 rows, target=species)
- [x] `samples/regression.csv` — 소형 회귀용 (diabetes, 442 rows, target=progression)
- [x] `scripts/generate_samples.py` — sklearn 내장 dataset → CSV 로 재현성 있게 생성
- [x] `tests/conftest.py` 에서 위 샘플 경로를 공통 fixture(`classification_csv`, `regression_csv`, `samples_dir`)로 노출

**수용 기준**
- `streamlit run app.py` 시 빈 페이지라도 오류 없이 기동됨
- 모든 `__init__.py` import 가능 (`python -c "import services, ml, repositories, utils, config"` 성공)
- `samples/*.csv` 가 git에 커밋되어 있고 `scripts/generate_samples.py` 로 재생성 가능

---

## 단계 1. 기본 유틸 / 인프라

*어떤 기능 코드도 먼저 얹히지 않는다. 유틸이 안정되어야 이후가 깨끗하다.*

### 1.1 설정 (`config/settings.py`)

- [x] pydantic-settings `Settings` 클래스
  - [x] 필드: `APP_ENV`, `DATABASE_URL`, `STORAGE_DIR`, `MAX_UPLOAD_MB`, `AUTH_MODE`, `DEFAULT_TEST_SIZE`, `RANDOM_SEED`, `LOG_LEVEL`
  - [x] `.env` 자동 로드 + 타입 검증
  - [x] `settings = Settings()` 전역 싱글톤
  - [x] 경로 필드(`STORAGE_DIR`)는 `Path`로 캐스팅 + 하위 디렉터리 자동 생성 메서드 제공

### 1.2 예외 (`utils/errors.py`)

- [x] `AppError(Exception)` 기반 클래스 (user_message 속성)
- [x] 하위 계층:
  - [x] `ValidationError` — 입력 오류
  - [x] `NotFoundError` — 엔터티 미존재
  - [x] `MLTrainingError` — 학습 실패
  - [x] `PredictionInputError` — 예측 입력 스키마 불일치
  - [x] `StorageError` — 파일/IO 오류

### 1.3 로깅 (`utils/log_utils.py`)

- [x] `get_logger(name: str) -> Logger` 팩토리
  - [x] 루트 로거 1회 초기화 (JSON or key=value 포맷)
  - [x] `storage/logs/app.log` 롤링 핸들러 + 콘솔 핸들러
  - [x] `settings.LOG_LEVEL` 반영
- [x] `log_event(logger, event: str, **extra)` 헬퍼 (구조화 로깅)

### 1.4 파일 유틸 (`utils/file_utils.py`) — FR-030, FR-034, NFR-006

- [x] `save_uploaded_file(uploaded, project_id) -> Path` (UUID 파일명 재명명)
- [x] `validate_extension(filename, allowed={"csv","xlsx"})`
- [x] `validate_size(bytes_len, max_mb)`
- [x] `read_tabular(path) -> pd.DataFrame` (CSV/XLSX 자동 분기, 헤더 검증)
- [x] `ensure_dir(path: Path)` 헬퍼

### 1.5 세션 유틸 (`utils/session_utils.py`) — FR-002, FR-003

- [x] 공식 세션 키 enum/상수 (`SessionKey`): `CURRENT_PROJECT_ID`, `CURRENT_DATASET_ID`, `LAST_TRAINING_JOB_ID`, `FLASH`
- [x] `get_state(key) -> Any | None`
- [x] `set_state(key, value) -> None`
- [x] `clear_state(*keys)`
- [x] Flash 메시지 API:
  - [x] `flash(level: Literal["success","warning","error","info"], message: str)`
  - [x] `consume_flashes() -> list[ToastMsg]` (한 번 렌더 후 비움)

### 1.6 공통 컴포넌트 (`pages/components/toast.py`)

- [x] `render_flashes()` — 사이드바/본문 상단에서 호출되는 공용 토스트 렌더러
- [x] level → `st.success/warning/error/info` 매핑

### 1.7 메시지/이벤트 카탈로그 (일관성 게이트)

하드코딩 한글 메시지와 로그 이벤트명을 **한 곳으로 모아** 페이지·서비스 전역에서 재사용한다.

- [x] `utils/messages.py` — 사용자 노출 한글 메시지 상수
  - [x] 네임스페이스 예: `class Msg: UPLOAD_SUCCESS = "업로드가 완료되었습니다."`
  - [x] 템플릿 메시지는 `def upload_too_large(max_mb: int) -> str` 형태의 함수로 제공
  - [x] 페이지/서비스는 `from utils.messages import Msg` 로만 참조 (하드코딩 금지)
- [x] `utils/events.py` — 구조화 로그 이벤트명 상수
  - [x] `class Event: PROJECT_CREATED = "project.created"`, `TRAINING_STARTED = "training.started"` …
  - [x] 감사 로그 `action_type` 도 이 상수를 재사용
- [-] 린트: `ruff` 룰로 `st.error("...")` 처럼 문자열 리터럴을 직접 쓰는 Streamlit 호출을 발견 시 경고 (개발 편의, 선택) *(차기 이월 — 현재 `utils.messages.Msg` 사용은 코드리뷰·rules(`streamlit-ui.mdc`)로 강제. ruff custom rule 은 ast 탐지기 구현 비용이 커서 MVP 외)*

### 1.8 단위 테스트 (`tests/utils/`)

- [x] `test_errors.py` — 예외 상속/메시지
- [x] `test_file_utils.py` — 확장자/크기/읽기 해피&실패 경로
- [x] `test_session_utils.py` — 키 통제, flash 소비 로직 (`st.session_state` 직접 사용)

**수용 기준**
- 다른 모듈이 `settings`, `get_logger`, 도메인 예외를 import해서 바로 쓸 수 있는 상태
- `storage/logs/app.log`에 로그가 실제로 기록됨
- CSV/XLSX 로드 + 잘못된 파일 경로에서 `ValidationError` 발생
- `Msg`, `Event` 상수가 pages/services에서 실제로 사용됨 (하드코딩 한글 리터럴 0건)

---

## 단계 2. 영속성 레이어 (DB & Repository)

*ORM을 먼저 확정해야 Service의 DTO 매핑이 깔끔하다.*

### 2.1 ORM 베이스 (`repositories/base.py`)

- [x] `engine = create_engine(settings.DATABASE_URL, future=True)`
- [x] `SessionLocal` 팩토리
- [x] `Base = DeclarativeBase`
- [x] `@contextmanager session_scope()` — commit/rollback/close 자동화
- [x] sqlite 상대 경로 정규화 + 부모 디렉터리 자동 생성 (`_resolve_sqlite_path`)
- [x] `TimestampMixin` (created_at/updated_at 자동 관리)

### 2.2 ORM 엔터티 (`repositories/models.py`) — 요구사항 §9

- [x] `User` (스키마만 준비, AUTH_MODE 정책은 아래 2.2a 참조)
- [x] `Project`
- [x] `Dataset` (project FK, schema_json 포함)
- [x] `TrainingJob` (dataset FK, task_type/target/metric/status/run_log)
- [x] `Model` (training_job FK, is_best, paths, feature_schema_json) — path 는 nullable (§4.3a 보상 로직)
- [x] `PredictionJob` (model FK, input_type, paths, status)
- [x] `AuditLog` (action_type, target, detail_json)
- [x] 관계 설정 + `created_at`/`updated_at` 자동화 + 상위 삭제 시 cascade

### 2.2a AUTH_MODE 정책 (MVP 고정)

*MVP 1차는 `AUTH_MODE=none`. 이때도 DB 스키마는 확장성을 유지한다.*

- [x] `Project.owner_user_id` → **nullable**. `AUTH_MODE=none` 일 때는 `None` 허용.
- [x] `init_db.py --seed` 시 `User(id=0, login_id="system", user_name="시스템", role="system")` 시스템 사용자 1건을 upsert
- [x] `AUTH_MODE=none` 컨텍스트에서 `AuditLog.user_id = 0` (시스템) 으로 기록 (`audit_repository.write` 기본값)
- [→] `AUTH_MODE=basic` 전환 시 마이그레이션 체크리스트를 **§8.1 로 이관** (본 섹션에서는 추적 종료):
  - [→] 기존 `owner_user_id IS NULL` Project 를 시스템 사용자 또는 특정 계정으로 귀속 — §8.1 참조
  - [→] `User.password_hash` NOT NULL 전환 — §8.1 참조
- [x] Service 레이어에서 `current_user_id()` 헬퍼(`utils/session_utils.py`) 로 사용자 컨텍스트를 일원화 (MVP 반환값=0) — §1.5 에서 완료
- [x] 상수 선언: `SYSTEM_USER_ID=0`, `SYSTEM_USER_LOGIN_ID`, `SYSTEM_USER_NAME`, `SYSTEM_USER_ROLE` (`repositories/models.py`)

### 2.3 Repository 구현

각 Repository는 **세션을 인자로 받는 함수 모듈**로 작성 (Service가 트랜잭션 주인).

- [x] `project_repository.py` — insert / update / delete / get / list_by_owner / list_all / count / exists_by_name
- [x] `dataset_repository.py` — insert / get / list_by_project / delete
- [x] `training_repository.py` — insert / update_status (status 화이트리스트 + started/ended 자동) / get / list_by_project / append_run_log
- [x] `model_repository.py` — bulk_insert / update_paths (§4.3a 보상 로직) / get / delete / list_by_training_job / list_by_project / mark_best (단일 best 강제)
- [x] `prediction_repository.py` — insert / update_status / get / list_by_model (input_type/status 화이트리스트)
- [x] `audit_repository.py` — write (user_id 기본값=SYSTEM_USER_ID) / list_logs (user/action/target/period 필터)

### 2.4 DB 초기화 스크립트 (`scripts/init_db.py`)

- [x] 엔터티 전체 메타데이터로 테이블 생성
- [x] 옵션: `--drop` 스키마 재생성
- [x] 옵션: `--seed` 시스템 사용자(id=0) + 샘플 프로젝트(`샘플: 붓꽃 분류`) upsert (멱등)
- [x] 구조화 로그 (`utils/log_utils`) 로 각 단계 기록

### 2.5 통합 테스트 (`tests/repositories/`)

- [x] `test_session_scope.py` — 커밋/롤백, 스키마/AUTH 정책 (6건)
- [x] `test_project_repository.py` — CRUD/list/count/cascade/감사로그 기본값 (11건)
- [x] `test_training_repository.py` — 상태전이/run_log/모델 bulk/예측/감사 필터 (9건)
- [x] `conftest.py` — 임시 sqlite + 스키마 생성 fixture (`sqlite_engine` / `session_factory` / `db_session`)

**수용 기준**
- [x] `python scripts/init_db.py` 실행 후 `db/app.db`에 테이블 생성 확인
- [x] `pytest tests/repositories -q` 전부 통과 (26건)
- [x] `PRAGMA foreign_keys=ON` 자동 활성화로 cascade delete 실제 동작 확인

---

## 단계 3. ML 엔진 (순수 함수 계층)

*Streamlit/DB 없이 단독 실행 가능해야 함. 이 단계가 끝나면 CLI로도 학습이 된다.*

### 3.1 도메인 스키마 분리 (ML vs Service)

**원칙**: `ml/schemas.py` 는 ML 엔진 내부 구조만, `services/dto.py` 는 UI ↔ Service 교환용 DTO.

#### 3.1a `ml/schemas.py` — ML 내부 전용 (Streamlit/DB 비의존)

- [x] `@dataclass(frozen=True, slots=True)`:
  - [x] `ColumnProfile(name, dtype, n_missing, n_unique, missing_ratio, unique_ratio)`
  - [x] `DatasetProfile(n_rows, n_cols, columns: tuple[ColumnProfile, ...])` + `column(name)` 헬퍼
  - [x] `TrainingConfig(...)` + `__post_init__` 검증 (test_size 범위, task_type, target)
  - [x] `FeatureSchema(numeric, categorical, target, categories)` + `to_dict`/`from_dict` (아티팩트 저장용)
  - [x] `ScoredModel(algo_name, status, metrics, error, train_time_ms)` + `is_success` 속성

#### 3.1b `services/dto.py` — Service 반환용 (UI가 소비)

- [x] `@dataclass(frozen=True, slots=True)`:
  - [x] `ProjectDTO` + `from_orm`
  - [x] `DatasetDTO` + `from_orm`
  - [x] `ColumnProfileDTO` / `DatasetProfileDTO`
  - [x] `TrainingJobDTO` + `from_orm`
  - [x] `ModelComparisonRowDTO`
  - [x] `TrainingResultDTO`
  - [x] `ModelDTO` + `from_orm`
  - [x] `FeatureSchemaDTO` + `ModelDetailDTO`
  - [x] `PredictionResultDTO` + `from_orm`
- [x] 원칙 문서화: **Repository → Service 경계에서 ORM → DTO 로 변환** (dto.py 모듈 doc-string 명시)
- [x] 변환 헬퍼: 각 DTO에 `from_orm(cls, entity)` classmethod

### 3.2 알고리즘 레지스트리 (`ml/registry.py`) — FR-062

- [x] `AlgoSpec(name, task, factory, default_metric)` dataclass (frozen)
- [x] 분류 카탈로그: logistic_regression, decision_tree, random_forest (+xgboost/lightgbm 선택)
- [x] 회귀 카탈로그: linear, ridge, lasso, random_forest (+xgboost/lightgbm 선택)
- [x] `get_specs(task_type)` / `get_spec(task_type, name)` / `available_algorithms(task_type)`
- [x] `random_state=settings.RANDOM_SEED` 일괄 적용
- [x] XGBoost/LightGBM import 가드: `ImportError` 뿐 아니라 `XGBoostError` 등 런타임 로드 실패(macOS libomp 미설치 등) 도 skip

### 3.3 데이터 프로파일링 (`ml/profiling.py`) — FR-033

- [x] `profile_dataframe(df) -> DatasetProfile` (missing/unique 비율 포함)
- [x] `suggest_excluded(profile, unique_ratio_threshold=0.95) -> list[str]` — 식별자 의심 컬럼 (n_rows<2 시 빈 리스트)

### 3.4 전처리 (`ml/preprocess.py`) — FR-050~053

- [x] `split_feature_types(df, target, excluded) -> (num_cols, cat_cols)`
- [x] `build_preprocessor(num_cols, cat_cols) -> ColumnTransformer`
- [x] `build_feature_schema(df, num_cols, cat_cols, target) -> FeatureSchema`
- [x] `prepare_xy(df, config) -> (X, y)` (타깃 제외/제외컬럼 제거)

### 3.5 학습 (`ml/trainers.py`) — FR-061, FR-063, NFR-004

- [x] `train_all(specs, X_train, y_train) -> list[TrainedModel]`
  - [x] 개별 실패는 `status="failed"`로 기록, 전체 중단 없음
  - [x] 각 모델별 학습 시간 측정
- [x] `split_dataset(X, y, test_size, task_type) -> (X_tr, X_te, y_tr, y_te)`
  - [x] 분류면 stratify 적용 (클래스 분포 이슈 시 fallback)

### 3.6 평가 (`ml/evaluators.py`) — FR-064, FR-071, FR-072

- [x] 분류: `accuracy, f1, roc_auc`
  - [x] 다중클래스는 `average="macro"` / roc_auc는 ovr
- [x] 회귀: `rmse, mae, r2`
- [x] `score_models(trained, X_test, y_test, task_type) -> list[ScoredModel]`
- [x] `select_best(scored, metric_key) -> ScoredModel` (방향 내장)
- [x] 시각화 보조:
  - [x] `confusion_matrix_data(y_true, y_pred, labels)` → dict 반환 (UI가 plot)
  - [x] `regression_scatter_data(y_true, y_pred)` → dict 반환

### 3.7 아티팩트 (`ml/artifacts.py`) — FR-073, §10.4

- [x] `save_model_bundle(dir: Path, estimator, preprocessor, schema: FeatureSchema, metrics: dict)` → 4개 파일 생성
- [x] `load_model_bundle(dir: Path) -> ModelBundle` (dataclass: estimator, preprocessor, schema, metrics)
- [x] `validate_prediction_input(df, schema) -> df` (FR-083: 누락/타입/추가 컬럼 처리)

### 3.8 단위 테스트 (`tests/ml/`, `tests/services/`)

- [x] `tests/ml/test_registry.py` — 카탈로그 완결성 + factory 새 인스턴스 + ml/ 의 streamlit/sqlalchemy 미사용 검증
- [x] `tests/ml/test_profiling.py` — profile 기본/결측·고유/빈 DF/식별자 제안/임계치
- [x] `tests/services/test_dto.py` — ORM→DTO 변환 / frozen 불변성 / 조합 DTO 구조
- [x] `test_preprocess.py` — 결측/범주형/스케일링/unseen 카테고리 안전성
- [x] `test_trainers.py` — 실패 모델 격리 (깨진 spec 주입) + 진행 콜백 + preprocessor clone 확인
- [x] `test_evaluators.py` — metric 방향, select_best, CM/scatter 데이터 구조
- [x] `test_artifacts.py` — save/load 라운드트립, 스키마 검증 *(`tests/ml/test_artifacts.py` 10건 — 4파일 생성 / roundtrip / missing dir·file 에러 / schema_json 검증 / `validate_prediction_input` 5시나리오(결측·강제형변환·범주 문자열화·빈 DF 거부))*
- [x] `test_pipeline_e2e.py` — iris/diabetes 샘플 CSV 로 분류+회귀 각 1회 풀 파이프라인 실행 (artifacts 제외)

**수용 기준**
- `pytest tests/ml -q` 전부 통과
- `ml/` 에서 `streamlit` / `sqlalchemy` import 없음 (`ruff` 또는 grep 확인)
- CLI 스크립트로 샘플 CSV 학습 → 아티팩트 저장 → 재로드 예측 성공

---

## 단계 4. Service 레이어 (비즈니스 로직)

*UI 없이 Service 함수만으로 유스케이스가 완결되어야 한다.*

### 4.1 Project Service (`services/project_service.py`) — FR-020~024

- [x] `create_project(name, description) -> ProjectDTO`
- [x] `list_projects() -> list[ProjectDTO]`
- [x] `get_project(project_id) -> ProjectDTO` (없으면 `NotFoundError`)
- [x] `update_project(project_id, name, description) -> ProjectDTO`
- [x] `delete_project(project_id, cascade: bool) -> None`
  - [x] cascade=False일 때 연결 리소스 존재 시 `ValidationError`
- [x] 모든 변경 액션 → `audit_repository.write`

### 4.2 Dataset Service (`services/dataset_service.py`) — FR-030~035

- [x] `upload_dataset(project_id, uploaded_file) -> DatasetDTO`
  - [x] 확장자/크기 검증 (file_utils)
  - [x] 파일 저장 (UUID 명명)
  - [x] 헤더 유효성/중복컬럼/빈파일 검증
  - [x] DB insert + 프로파일(schema_json) 저장
- [x] `get_dataset_profile(dataset_id) -> DatasetProfileDTO` (스펙상 DTO 로 상향)
- [x] `preview_dataset(dataset_id, n=50) -> list[dict]` (상위 N행, NaN→None 정규화, PREVIEW_MAX_ROWS 클램프)
- [x] `list_datasets(project_id) -> list[DatasetDTO]`
- [x] `delete_dataset(dataset_id) -> None` (파일 포함 삭제, 트랜잭션 커밋 이후 best-effort unlink)

### 4.3 Training Service (`services/training_service.py`) — FR-060~066

- [x] `run_training(config: TrainingConfig) -> TrainingResultDTO`
  - [x] 전처리 빌드 → split → train_all → score → best 선정
  - [x] TrainingJob 레코드 생성 + 상태 전이(`pending→running→completed/failed`)
  - [x] Model 레코드 저장 (비교 테이블의 각 행)
  - [x] 성공한 모델 아티팩트만 저장 (`storage/models/<model_id>/`)
  - [x] run_log에 단계별 메시지 append
- [x] `get_training_result(job_id) -> TrainingResultDTO`
- [x] `list_training_jobs(project_id) -> list[TrainingJobDTO]`
- [x] 진행률 콜백 훅: `run_training(..., on_progress: Callable[[str, float], None] | None)`
  - [x] **Streamlit 재실행 모델 제약상 스레드/비동기 사용 금지**. 동기 루프에서 콜백을 호출하고, 페이지 쪽은 `with st.status(...)` 블록 안에서 Service를 호출한다.
  - [x] 콜백 시그니처: `on_progress(stage: str, ratio: float)` — `stage ∈ {"preprocessing","split","train:<algo>","score","save","completed"}`

### 4.3a 아티팩트 저장 순서 (트랜잭션 내 일관성)

학습이 완료된 모델은 **DB id 확보 → 파일 저장 → path 업데이트** 순서로 처리한다. 반드시 단일 트랜잭션 내에서 수행하되, 파일 저장은 트랜잭션 외부 I/O 이므로 실패 시 보상 로직이 필요하다.

```
with session_scope() as session:
    job = training_repository.insert(session, ...)             # 1) TrainingJob id 확보
    for scored in scored_models:
        model = model_repository.insert(session, job.id, ...)  # 2) Model row (path 비워둠)
        session.flush()                                         # 3) model.id 확보
        try:
            artifacts.save_model_bundle(
                storage_dir / str(model.id), scored.estimator, preprocessor, schema, scored.metrics
            )                                                   # 4) 파일 저장
            model_repository.set_paths(session, model.id, ...) # 5) path 업데이트
        except StorageError:
            logger.exception("artifact.save_failed", extra={"model_id": model.id})
            # 파일 저장 실패 시: DB 롤백되도록 예외 재발생
            raise
    session.commit()
```

- [x] `model_repository.set_paths(session, model_id, model_path, preprocessor_path, schema_path, metrics_path)` 추가 *(실제 이름: `update_paths` — `feature_schema_json` 까지 같이 갱신하므로 조금 더 포괄적인 명칭으로 구현)*
- [x] 부분 실패 시 보상: `session_scope` 롤백 + 이미 쓰여진 파일은 `try/finally` 에서 정리 (best effort) *(구현: `services/training_service._persist_and_save` + `_cleanup_model_dirs`)*
- [x] `is_best` 는 모든 모델 저장 완료 후 한 번에 결정 → `model_repository.mark_best(session, job_id, best_model_id)` *(단일 트랜잭션 내에서 전체 저장 완료 후 best_entity 결정 → mark_best 호출)*

### 4.4 Model Service (`services/model_service.py`) — FR-073~075

- [x] `save_model(model_id) -> ModelDTO` (is_best 외 수동 저장 지원)
- [x] `list_models(project_id) -> list[ModelDTO]`
- [x] `get_model_detail(model_id) -> ModelDetailDTO` (metrics_summary, feature_schema 포함)
- [x] `delete_model(model_id) -> None` (파일+레코드)

### 4.5 Prediction Service (`services/prediction_service.py`) — FR-080~085

- [x] `predict_single(model_id, payload: dict) -> PredictionResultDTO`
  - [x] 스키마 검증 → 전처리 → 추정기 → 결과 반환
- [x] `predict_batch(model_id, file_path: Path) -> PredictionResultDTO`
  - [x] 누락 컬럼 차단(§10.4), 추가 컬럼 경고 후 무시
  - [x] 결과 CSV를 `storage/predictions/<job_id>.csv`에 저장
- [x] PredictionJob 레코드 기록

### 4.6 Service 테스트 (`tests/services/`)

- [x] `conftest.py` — 임시 sqlite + 임시 storage_dir + seed 시스템 유저 (§4.1 완료 시점)
- [x] `test_project_service.py` — CRUD + cascade 거부 (23건)
- [x] `test_dataset_service.py` — 업로드 해피/실패·프로파일·프리뷰·리스트·삭제 (22건)
- [x] `test_training_service.py` — 샘플 CSV로 분류·회귀 각 1회 성공 (11건)
- [x] `test_model_service.py` — list/detail/save(promote)/delete + cleanup (10건)
- [x] `test_prediction_service.py` — 단건/파일, 누락 컬럼 차단, 보상 로직 (11건)

**수용 기준**
- `pytest tests/services -q` 전부 통과
- Service 함수가 Streamlit 없이 import/실행 가능 (`python -c "from services.training_service import run_training"`)
- 반환 타입 모두 DTO (ORM 노출 없음) — 코드 리뷰에서 확인

---

## 단계 5. UI 스켈레톤

*페이지 별 구현 전, 공통 셸을 먼저 고정.*

### 5.1 App 진입점 (`app.py`)

- [x] `st.set_page_config(page_title="AutoML", layout="wide")` *(공용 `pages/components/layout.configure_page` 경유 — 각 페이지 재사용)*
- [x] 사이드바: 현재 프로젝트 표시, 페이지 네비 안내 *(`render_sidebar` — DB 뱃지, 프로젝트 카드, 선택 해제 버튼 포함)*
- [x] 초기화: DB 존재 체크 → 없으면 안내, flash 렌더 *(`utils/db_utils.is_db_initialized` + 본문/사이드바 이중 안내, `pages/components/toast.render_flashes` 로 플래시 소비)*
- [x] 홈 본문(화면 1): 서비스 소개 / 최근 프로젝트 3개 / 시작 버튼 *(`project_service.list_projects` 상위 3개 카드, "선택하기" 클릭 → `SessionKey.CURRENT_PROJECT_ID` 반영, stale id 자동 정리)*

### 5.2 공용 컴포넌트

- [x] `pages/components/toast.py` (단계 1에서 이미 작성)
- [x] `pages/components/layout.py` — `configure_page` / `render_sidebar` / `render_page_header` (§5.1 과 함께 선행 구현)
- [-] `pages/components/project_picker.py` — 사이드바에서 프로젝트 선택 위젯 *(§6.1 에서 현재 필요성 낮다고 판단되어 보류. 본문 카드 + 홈 "선택하기" 버튼이 동일 역할을 수행 중)*
- [x] `pages/components/data_preview.py` — df.head + 프로파일 테이블 *(§6.2 에서 구현. `render_preview(rows, caption, height)` + `render_profile(profile)` 2-API, 행/컬럼 메트릭 + 결측/고유 비율 테이블)*
- [-] `pages/components/metric_cards.py` — 성능 비교 카드/표 렌더러 *(분리 보류 — §6.4 `pages/04_results.py` 에서 `st.columns` + `st.metric` + `pd.DataFrame` 로 인라인 구현 중. 재사용처가 2 페이지 이상 되면 그때 추출)*
- [-] `pages/components/plots.py` — confusion matrix, scatter, metric bar *(분리 보류 — §6.4 결과 페이지와 §6.7 admin 페이지에서 `plotly.express` 로 인라인 구현. 축/라벨 규약이 달라 공통 래퍼가 오히려 옵션 폭발을 유발해서 MVP 에서는 페이지에 둔다)*

### 5.3 세션 가드

- [x] `require_project()` / `require_dataset()` 헬퍼 (`utils/session_utils` 또는 components 내 위치) *(`utils/session_utils.py:70,79` — `SessionKey.CURRENT_PROJECT_ID` / `CURRENT_DATASET_ID` 유효성 확인 후 int 반환)*
- [x] 미설정 시 `st.info` + `st.stop` 일관 적용 *(두 헬퍼 모두 미설정 시 `st.warning(Msg.*)` + `st.stop()` 호출. `pages/03_training.py`·`06_prediction.py` 등 전 페이지에서 시그니처 통일 호출)*

**수용 기준**
- `streamlit run app.py` → 홈 화면 정상 렌더, 사이드바에 프로젝트 선택 UI
- 공용 컴포넌트만 import해서 다른 페이지에서 재사용 가능

---

## 단계 6. 페이지별 기능 구현

*각 페이지는 단계 3/4의 Service만 호출한다. UI 로직 외에 비즈니스 로직을 두지 않는다.*

### 6.1 `pages/01_projects.py` — 프로젝트 관리 (화면 2, FR-020~024)

- [x] 프로젝트 목록 표 (선택/수정/삭제 버튼) *(행 별 `st.container(border=True)` + 3버튼 컬럼 구성, 현재 선택 항목은 ★ 마커 + 선택 버튼 비활성화)*
- [x] 생성 폼 (name, description) *(`st.expander` 안의 `st.form` — 생성 성공 시 자동 선택 → 사이드바 즉시 반영)*
- [x] 선택 시 `current_project_id` 세팅 *(`_select_project` 헬퍼가 success flash + session 상태 업데이트 + rerun 일괄 처리)*
- [x] 삭제 시 cascade 옵션 확인 다이얼로그 *(세션 상태 기반 인라인 컨펌 블록 + cascade 체크박스, `st.dialog` 는 AppTest 안정성 이슈로 회피 결정 기록)*
- [x] 수용: 생성 → 목록 갱신 → 선택 → 사이드바에 반영 *(AppTest 시나리오 `test_create_project_success_updates_list_and_selection` + `test_select_button_sets_current_project` 로 검증)*

### 6.2 `pages/02_dataset_upload.py` — 데이터 업로드 (화면 3, FR-030~035)

- [x] 파일 업로더 (csv/xlsx) *(`st.form` + `clear_on_submit=True` + `ALLOWED_EXTENSIONS` 화이트리스트. 파일 미선택 시 warning flash)*
- [x] 업로드 후 자동 미리보기(data_preview) + 컬럼 프로파일 *(성공 시 `SessionKey.CURRENT_DATASET_ID` 세팅 → 탭 2개(`샘플 데이터` / `컬럼 프로파일`) 자동 렌더. `pages/components/data_preview.py` 의 `render_preview` / `render_profile` 로 분리 — §6.3 학습 페이지에서도 재사용)*
- [x] 잘못된 파일 → error flash *(Service 의 `ValidationError`/`StorageError` 를 `flash("error")` + `st.rerun()` 로 즉시 렌더)*
- [x] 데이터셋 목록 + 삭제 버튼 *(행별 `st.container(border=True)` + `[선택][삭제]` 컬럼. 세션 플래그 기반 인라인 컨펌 블록으로 cascade 삭제 확인)*
- [x] 수용: 5만 행 CSV 미리보기 10초 이내 (NFR-003) *(로컬 측정 upload 0.15s + preview 0.11s + profile 0.001s = **0.26s** < 10s)*

### 6.3 `pages/03_training.py` — 학습 설정/실행 (화면 4·5, FR-040~066)

- [x] 문제 유형 라디오 (분류/회귀) *(`st.radio(horizontal=True)` + `task_type` 변경 시 metric 옵션 즉시 갱신)*
- [x] 타깃 컬럼 select *(`st.selectbox`, 데이터셋 컬럼 목록)*
- [x] 제외 컬럼 멀티 select (+ `suggest_excluded` 힌트) *(`services.dataset_service.suggest_excluded_columns(threshold=0.95)` 를 기본값으로 주입. target 컬럼은 후보에서 제외)*
- [x] 테스트 비율 slider (기본값 settings.DEFAULT_TEST_SIZE) *(0.05~0.5 step 0.05)*
- [x] 기준 지표 select (task_type별 옵션) *(`CLASSIFICATION_METRICS` / `REGRESSION_METRICS` 튜플 + `METRIC_DIRECTIONS` 방향 설명 help)*
- [x] 학습명 text input *(`max_chars=100`, 공백만 → None 처리)*
- [x] 실행 버튼 → `st.status` 로 진행률 표시 *(`training_service.run_training(on_progress=...)` 동기 콜백 → `st.progress` + caption 단계 로그. 실패 시 `status.update(state="error")`)*
- [x] 완료 후 `last_training_job_id` 세팅 + 결과 페이지로 이동 *(`SessionKey.LAST_TRAINING_JOB_ID` 갱신 → 요약 카드(3 metric) + 베스트 배지 + "결과 비교 페이지로 이동" CTA. §6.4 미구현 환경에서는 `switch_page` 실패 시 info flash 로 폴백)*
- [x] 수용: 샘플 CSV 분류 1회, 회귀 1회 성공 → 결과 페이지로 전환 *(tests/ui/test_training_page.py 의 `@pytest.mark.slow` 해피패스 2건이 실제 `run_training` 실행 후 `LAST_TRAINING_JOB_ID` 가 세팅되는지 검증)*

### 6.4 `pages/04_results.py` — 결과 비교 (화면 6, FR-070~073)

- [x] 성능 비교표 (metric_cards) — **규격 고정**:

  | 컬럼 | 타입 | 비고 |
  |------|------|------|
  | `algorithm` | str | `registry.py` spec name |
  | `status` | `"ok" \| "failed"` | 실패 모델도 테이블에 1행으로 유지 (FR-066) |
  | 기준 지표값 | float | 사용자가 선택한 `metric_key` 컬럼 강조 |
  | 보조 지표 2~3 | float | task_type 별 나머지 지표 |
  | `train_time_ms` | int | 학습 시간 |
  | `is_best` | bool | 베스트 1행에 배지 |
  | `error` | str | status="failed" 일 때만 |

  - 정렬: 기준 지표의 방향(`metric_direction`) 기준 내림/오름차순 *(`_sort_rows`: success 행은 direction 에 맞춰 정렬, failed 행은 맨 아래)*
  - 실패 행 표시: 지표 컬럼은 `None` (Streamlit 이 자동 `—` 처럼 빈칸 렌더), 에러 컬럼에 원문 노출
  - 렌더: `st.dataframe` + `column_config` *(is_best → `★ best` 텍스트 배지, 메트릭은 `NumberColumn(format="%.4f")`)*
- [x] 베스트 모델 강조(배지/색상) *(`★ best` 배지 컬럼 + 요약 영역의 `st.success` 카드 `↑/↓` 방향 표기)*
- [x] 분류: 혼동행렬 heatmap / 회귀: 예측 vs 실제 scatter (plots) *(학습 시점에 `build_plot_data` 결과를 `<model_dir>/plot_data.json` 으로 영속화 → `model_service.get_model_plot_data` 로 로드 → plotly imshow / scatter 렌더, plotly 실패 시 dataframe 폴백)*
- [x] "이 모델 저장" / "다른 모델 저장" 버튼 → model_service.save_model *(selectbox 로 성공 모델 목록 노출, 현재 베스트는 버튼 disabled + "베스트로 고정됨" 라벨, 다른 모델 선택 → `model_service.save_model` 이 is_best 승격 + MODEL_SAVED audit)*
- [x] 수용: 3개 이상 모델 비교 표시(실패 1건 포함해도 나머지 정상), 저장 후 모델 관리로 이동 가능 *(AppTest 4건으로 검증: 비교표 렌더, 플롯 셀렉터 존재, save 클릭 후 `is_best` 이동, CTA 버튼 → `st.switch_page` 폴백. §6.5 미구현 시 info flash)*

### 6.5 `pages/05_models.py` — 모델 관리 (화면 7, FR-074, FR-075)

- [x] 프로젝트별 저장 모델 목록 (알고리즘/생성일/metric) *(`model_service.list_models` → `st.container(border=True)` 4컬럼 행: 이름·주 지표·상세/삭제 버튼. 실패 모델도 목록에 유지, 베스트는 `★` prefix + 배지)*
- [x] 모델 상세(입력 스키마, metrics_summary) *(상세 버튼 토글 → `get_model_detail` 로 `FeatureSchemaDTO`·`metrics_summary` 전개. 수치/범주 컬럼 목록 + 타깃 강조, 범주 값은 `st.expander` 로 지연 노출)*
- [x] "예측하러 가기" 버튼 → 06 page로 이동 + 모델 선택 상태 전달 *(`SessionKey.CURRENT_MODEL_ID` 세팅 후 `st.switch_page("pages/06_prediction.py")`. §6.6 미구현 환경에서는 info flash 폴백으로 세션 값 유지)*
- [x] 삭제 버튼 (확인 다이얼로그) *(§6.1 과 동일하게 `st.dialog` 대신 세션 플래그 기반 인라인 확인 블록. 확정 시 `model_service.delete_model` 이 DB + `<models_dir>/<id>/` 정리, 취소 시 상태 초기화)*
- [x] 필터: 저장된 모델만 보기 토글 *(`is_best=True` 만 노출. 기본 off — 학습 직후 비교를 지원하면서도 정리된 뷰를 원할 때 on)*
- [x] 수용: AppTest 9건 (DB/프로젝트 가드, 빈 목록 안내, 목록 + ★ 배지, best-only 필터, 상세 펼침, 예측 이동 `CURRENT_MODEL_ID` 세팅, 삭제 확정/취소)

**진행 로그 (2026-04-20)**

- `utils/session_utils.py` 에 `SessionKey.CURRENT_MODEL_ID` 추가 — §6.5 → §6.6 간 모델 선택 상태 전달
- `utils/messages.py` 에 `Msg.MODEL_DELETED` / `Msg.MODEL_REQUIRED` 추가
- `pages/05_models.py` 생성: DB/프로젝트 가드 → 필터 토글 → 모델 카드 리스트 → 상세/삭제 인라인 패널. `model_service.list_models` / `get_model_detail` / `delete_model` / `save_model` (기존 서비스 재사용)
- `tests/ui/test_models_page.py` 9건 추가 (@pytest.mark.slow 6건 — 실제 `run_training` 으로 아티팩트 시드)
- `pytest` 265 passed (기존 256 + 신규 9), `ruff check` / `black` 통과, Streamlit smoke 6 페이지 (`/`, `/01~/05`) 모두 200

### 6.6 `pages/06_prediction.py` — 예측 (화면 8, FR-080~085)

- [x] 모델 선택 selectbox *(학습 성공 모델만 노출, 기본 선택 우선순위: `SessionKey.CURRENT_MODEL_ID` → `is_best` → 첫 행. 선택 즉시 `CURRENT_MODEL_ID` 동기화하여 §6.5 연동 CTA 흐름 유지)*
- [x] 탭: 단건 입력 / 파일 예측 *(`st.tabs(["단건 입력", "파일 예측"])`)*
- [x] 단건: `feature_schema.json` 기반 입력 폼 자동 생성 (FR-082) *(`st.form` 안 2열 그리드. `prediction_service.predict_single` 호출 후 예측값 + 분류일 때 상위 3개 확률 metric 카드 + 상세 expander)*
  - [x] numeric → number_input, categorical → selectbox(카테고리 목록) *(카테고리 목록이 비어있는 경우에만 `text_input` 폴백)*
- [x] 파일 예측: 업로드 → 결과 표 + CSV/XLSX 다운로드 버튼 *(`st.file_uploader(type=[csv,xlsx])` → `<predictions_dir>/_inputs/<project_id>/<uuid>.<ext>` 로 임시 저장 후 `predict_batch` → `st.dataframe` 미리보기 + `st.download_button` 로 원본 결과 CSV 다운로드)*
- [x] 누락 컬럼 시 사용자 친화 메시지 + 차단 (§10.4) *(Service 의 `PredictionInputError` → `flash("error", ...)` → `render_flashes` 에서 `st.error`)*
- [x] 수용: AppTest 8건 (DB/프로젝트/모델 가드, 모델 picker 기본값, 단건 해피패스 + 결과 캐시, Service error surface, 배치 결과 렌더 + 다운로드 버튼 노출, 업로드 없을 때 실행 버튼 disabled)

**진행 로그 (2026-04-20)**

- `pages/06_prediction.py` 생성: 가드 → 모델 picker → 상세 요약 → 탭(단건/파일) → 결과 캐시/다운로드
- 단건: `st.form` 안에서 스키마 기반 2열 그리드 + `number_input`/`selectbox` 자동 생성 → `prediction_service.predict_single` 호출 → 예측값 metric + 분류 확률 상위 3개
- 배치: `st.file_uploader` → `<predictions_dir>/_inputs/<project_id>/<uuid>.<ext>` 저장 (예측 재현성 보존) → `predict_batch` → 미리보기 `st.dataframe` + `st.download_button`
- 서비스·세션 재사용만으로 완결 (서비스 코드 변경 없음) — `predict_single`/`predict_batch` 는 §4.5 에서 이미 완성
- `tests/ui/test_prediction_page.py` 8건 추가 (2건 `@pytest.mark.slow` — 실제 `run_training` 아티팩트 시드). 나머지 6건은 `list_models`/`get_model_detail`/`predict_single` 패치로 경량 검증
- AppTest 호환 메모: `st.form_submit_button` 은 `at.button` 에 `FormSubmitter:<form_key>-<label>` 키로 노출되므로 이 키로 접근. `st.download_button` 은 `at.get("download_button")` 으로만 보이므로 길이로 존재성 검증
- `pytest` 273 passed (기존 265 + 신규 8), `ruff check` / `black` 통과, Streamlit smoke 7 페이지 (`/`, `/01~/06`) 모두 200

### 6.7 `pages/07_admin.py` — 이력/관리자 (화면 9·10, FR-090~093)

- [x] 학습 이력 테이블 (필터: 프로젝트/상태/기간) *(`admin_service.list_training_history` — Project/Model/BestModel 까지 조인 집계해 `project_name`/성공·실패 수/베스트 알고리즘·점수/소요시간(ms) 한 번에 반환. `st.dataframe(column_config=...)` 로 `best_score %.4f`, `소요(ms) %d` 포맷)*
- [x] 예측 이력 테이블 *(`admin_service.list_prediction_history` — PredictionJob → Model → TrainingJob → Project 역조인. input_type(form/file)·status·결과파일 경로 표시)*
- [x] 통계 카드: 프로젝트 수, 데이터셋 수, 모델 수, 실패 건수 *(`admin_service.get_stats` → `AdminStatsDTO` 5개 + 실패 2개 → `st.metric` 7개. 실패는 `delta_color="inverse"` 로 시각적 강조)*
- [x] 최근 실패 로그 요약 (audit_repository) *(`admin_service.list_recent_failures` — `AuditLog.action_type LIKE '%_failed'` 만 필터. action_type/target/detail 전개)*
- [x] 공통 필터: 프로젝트(전체/개별) · 상태(전체/completed/failed/running/pending) · 기간(7/30/90일/전체) *(세션 상태로 유지되어 탭 전환에도 보존)*
- [x] 수용: AppTest 6건 (DB 가드, 빈 DB 통계+안내, 필터 위젯 존재, 학습 해피패스 목록, 실패 감사 로그 렌더, 프로젝트 필터 선택 유지)

**진행 로그 (2026-04-20)**

- `services/admin_service.py` 신규 — 교차 도메인 집계를 이 모듈에 집약. `session_scope` 내부에서 필요한 서브쿼리(Model 집계/베스트)까지 SQL 로 처리해 N+1 회피
- `services/dto.py` 에 `AdminStatsDTO`, `TrainingHistoryRowDTO`, `PredictionHistoryRowDTO`, `AuditLogEntryDTO` 추가 (UI 는 ORM 을 여전히 보지 않음)
- `tests/services/test_admin_service.py` 9건 추가 (1건 `@pytest.mark.slow` 실제 `run_training` 수반 + 빈 DB/필터 조합/실패 필터/입력 검증 등 경량 8건). FK 제약 회피를 위해 `Dataset` 을 실제 insert 한 뒤 `TrainingJob` 을 참조하는 헬퍼 `_insert_dataset` 추가
- `pages/07_admin.py` 신규 — DB 가드 → 통계 카드 → 공통 필터 → 3탭(학습/예측/실패). 프로젝트 가드는 두지 않아 전역 조회 가능
- `pages/components/layout.py` 의 `NAV_ITEMS` 에 `"이력/관리자"` 추가 — 사이드바 네비 일관성
- `tests/ui/test_admin_page.py` 6건 추가 (1건 `@pytest.mark.slow` — 실제 학습 이력 테이블 검증)
- `pytest` 288 passed (기존 273 + 신규 15), `ruff check` / `black` 통과, Streamlit smoke 8 페이지 (`/`, `/01~/07`) 모두 200

### 6.8 인앱 매뉴얼 (FR-100 신설, 2026-04-21)

릴리즈 이후 사용자 온보딩 / 트러블슈팅 경험을 개선하기 위해 "앱 안에서 바로 보는
매뉴얼" 기능을 추가한다. 요구사항 문서 원본에는 없었으나, 운영 중 빈도가 높은
"어디서 뭘 하는가 / 에러가 났는데 어디를 봐야 하는가" 질문에 직접 응답하기 위해
FR-100 으로 편입.

**설계 결정**

- **배치**: 하이브리드. (a) 각 페이지 상단 `❓ 이 페이지 도움말` expander + (b) 전체 매뉴얼 허브 페이지(`pages/00_manual.py`, 사이드바 네비 "매뉴얼").
- **콘텐츠 저장**: `docs/manual/*.md` 단일 출처(SSOT). 코드가 아닌 텍스트만 수정해 빠른 반영 가능.
- **깊이**: 페이지 목적 · 주요 버튼/옵션 · 주의사항 · 연결 파일/FR · 자주 겪는 오류 (깊이2 수준).
- **대상**: 👤 사용자 섹션 + 🛠 개발자 섹션을 같은 MD 안에 공존. 기능별 접근 시 동일 문서에서 맥락 연결.

**체크리스트**

- [x] `pages/components/help.py` — `render_help(key)` expander 렌더 + `render_manual_section(key, anchor=)` + `list_manual_keys()` + `load_manual(key)` (mtime 기반 캐시, 트래버설 차단)
- [x] `docs/manual/` — `00_overview`, `01_projects` ~ `07_admin`, `troubleshooting`, 내부 `_hub_intro` 총 10 개 MD (깊이2 초안 완결)
- [x] `pages/00_manual.py` — 허브 페이지. 검색창(부분일치/대소문자무시) 으로 섹션 필터, TOC 2 컬럼 레이아웃. DB 없이도 열람 가능.
- [x] `pages/components/layout.py` — `NAV_ITEMS` 에 "매뉴얼" 엔트리 + 사이드바 `📘 매뉴얼 열기` 버튼 → `st.switch_page`.
- [x] `app.py` 홈 CTA — "문서 보기" → "📘 매뉴얼 보기" 로 변경, 허브 페이지로 `switch_page`.
- [x] 기능 페이지 01~07 상단에 `render_help("<page_key>")` 한 줄씩 삽입.
- [x] `tests/ui/test_help.py` — 9건 (로더 단위, 경로 트래버설 방어, mtime 캐시 무효화, 허브 렌더, 검색 필터, 기능 페이지 expander 노출).
- [x] `tests/docs/test_manual_files.py` — 19건 (필수 키 존재 / 최소 길이 / 사용자·개발자 섹션 보장 / libomp·seed 트러블슈팅 포함 / 미등록 파일 알람).

**수용 기준**

- 사이드바에서 "매뉴얼" 로 진입 → 10 개 섹션이 TOC + 본문 모두 렌더된다.
- 기능 페이지 어디서나 `❓ 이 페이지 도움말` 박스가 보이고, 내용이 해당 페이지 용도에 부합한다.
- 검색어 "libomp" 입력 시 트러블슈팅 섹션만 남는다.
- 매뉴얼 MD 를 수정하면 Streamlit 재실행 없이(동일 세션 next-rerun) 변경 내용이 반영된다.

**진행 로그 (2026-04-21)**

- 브랜치: `feature/in-app-manual` — 매뉴얼 기능만 격리.
- MD 콘텐츠는 AutoML_Streamlit_MVP.md (FR 매핑) · ARCHITECTURE.md (파일 경로) · README.md (트러블슈팅) 를 재구성해 사용자/개발자 양쪽 모두에게 한 화면으로 제공.
- `pytest tests/docs/ tests/ui/test_help.py` → 28 passed. 기능 페이지 전체 회귀는 하단 §8 체크.
- 향후 §8.x (배포) 시 정적 사이트 빌드가 필요하면 동일 MD 를 MkDocs/Docusaurus 로 재활용 가능하도록 파일 구조를 설계 (코드 의존 없음).

**수용 기준 (단계 6 전체)**
- `AutoML_Streamlit_MVP.md` §13.1의 모든 필수 수용 조건 ✅
- 분류/회귀 샘플 E2E 시나리오 각 1회 성공
- §6.8 인앱 매뉴얼 FR-100 ✅ (2026-04-21)

---

## 단계 7. 운영 / 품질 다지기

### 7.1 로그·감사 점검

- [x] 모든 Service 주요 분기에 `log_event` 호출 확인 *(services/*.py 전 도메인 15건: project 3 · dataset 3 · training 4 · model 2 · prediction 3. 각 성공·실패 브랜치에 `utils/events.py::Event` 상수로 통일)*
- [x] AuditLog 엔트리 확인 (프로젝트 생성 / 업로드 / 학습 / 저장 / 예측) *(`audit_repository.write(...)` 호출 15건 — log_event 와 1:1 매핑. `action_type` 은 `<domain>.<verb>[_failed]` 규약을 따름)*
- [x] `storage/logs/app.log` 롤링 동작 확인 *(`utils.log_utils._initialize` = `RotatingFileHandler(maxBytes=5MB, backupCount=3)`. 회귀 `tests/utils/test_log_utils.py` 4건 — 네임스페이스/idempotent, `| k=v` 포매터, handler 설정, 작은 maxBytes 로 실제 롤링 강제 → 백업 ≥1 / ≤ backupCount 검증)*

**진행 로그 (2026-04-20)**

- **Service 주요 분기 감사 매핑**
  | 도메인 | 성공 이벤트 | 실패/변형 이벤트 | 파일 위치 |
  |-|-|-|-|
  | project | `project.created`/`updated`/`deleted` | — | `services/project_service.py:96,166,208` |
  | dataset | `dataset.uploaded`/`deleted` | `dataset.upload_failed` | `services/dataset_service.py:104,160,277` |
  | training | `training.started`/`completed` | `training.failed`, 개별 `training.model_failed` | `services/training_service.py:166,286,314,383` |
  | model | `model.saved`/`deleted` | `artifact.save_failed` (training_service 내부) | `services/model_service.py:130,171` |
  | prediction | `prediction.started`/`completed` | `prediction.failed` | `services/prediction_service.py:195,235,267` |
- 감사 로그와 구조화 로그는 **같은 `Event.*` 상수**를 공유 — UI/운영 관점에서 `action_type` 필터와 stdout/log 파일 검색이 동일한 키로 연결된다 (`utils/events.py`).
- 로그 포매터는 `_KVFormatter` 가 `| k=v` 접미사 형태로 `extra=` 필드를 노출. `automl` 루트는 `propagate=False` — `caplog` 대신 실제 `RotatingFileHandler` 가 기록한 `app.log` 내용을 읽어 회귀 검증했다.
- 신규 테스트: `tests/utils/test_log_utils.py` (4 tests)
  - `test_get_logger_idempotent_and_namespaced` — `get_logger` 가 같은 이름에 같은 로거를 리턴하고 핸들러는 file + console 2개 고정
  - `test_log_event_format_includes_extra_kv` — `project_id=7 size_bytes=1024` 같은 k=v 직렬화를 파일 출력물에서 확인
  - `test_rotating_file_handler_is_configured` — `maxBytes=5MB · backupCount=3 · name=app.log` 3중 확인
  - `test_rotating_file_handler_actually_rotates` — maxBytes 를 256 바이트로 강제 교체 후 백업 파일(`app.log.1`, `app.log.2`) 생성/회수 확인
- `pytest -q`: 296 → **300 passed** (+4)

### 7.2 성능 점검 (NFR-003)

- [x] 5만 행 CSV 업로드·미리보기 10초 이내 *(`scripts/perf_bench.py --rows 50000` 실측 — upload 0.10s + profile 0.00s + preview 0.07s = **총 0.16s**, 목표 대비 60x 이상 여유)*
- [x] 저장 모델 단건 예측 3초 이내 *(logistic_regression 학습 후 `predict_single` — **cold 0.02s / warm median 0.02s**, 목표 대비 150x 이상 여유)*
- [x] `@st.cache_data` / `@st.cache_resource` 적용 확인 *(감사 결과: **현재 pages/ 에 미적용**. NFR-003 기준이 실측으로 충분히 달성되어 MVP 단계에서는 생략. 데이터 미리보기/프로파일은 DB schema_json 영속화로, 모델 로드는 joblib I/O 자체가 20ms 수준이라 Streamlit rerun 비용 대비 이득이 작음. follow-up 으로 기록)*

**진행 로그 (2026-04-20)**

- 신규 스크립트: `scripts/perf_bench.py` — 격리된 `STORAGE_DIR`/`DATABASE_URL` 에 5만 행 CSV(숫자 8 + 범주 4 + 타깃) 업로드, 프로파일, 3회 preview median, 800행 classification 학습 1건, predict_single 5회(cold + warm) 를 재현성 있게 측정. 대시 `make bench` 로 실행.
- 실측 요약 (macOS, Python 3.11.14, 2회 돌린 중앙값):
  | 단계 | 실측 | 목표 | 비율 |
  |-|-|-|-|
  | `upload_dataset` (50k 행) | 0.10–0.17s | 10s | ≤ 1.7% |
  | `get_dataset_profile` | 0.002s | 2s | ≤ 0.1% |
  | `preview_dataset` (n=50, median of 3) | 0.07s | 1s | ≤ 7% |
  | **5만 행 업로드 총합** | **0.16–0.24s** | **10s** | **≤ 2.4%** |
  | `train_one_algo` (logistic, 800 rows) | 0.59–0.63s | 참고치 15s | ≤ 4.2% |
  | `predict_single` (cold) | 0.02s | 3s | ≤ 0.8% |
  | `predict_single` (warm median) | 0.02s | 3s | ≤ 0.7% |
- **`@st.cache_data` / `@st.cache_resource` 적용 현황 감사**
  - `rg '@st\.(cache_data|cache_resource)' pages/ services/ ml/` → 0건. 설계 규약(`.cursor/rules/streamlit-ui.mdc`, `SKILL.md`) 은 데이터 미리보기/프로파일/모델 로드에 캐시를 권장하지만, 현재는 DB/파일 영속화로 비싼 연산이 이미 한 번만 수행됨 — 캐시 레이어는 중복.
  - 벤치가 NFR 목표를 2 자릿수 여유로 통과해 MVP 범위에서는 **캐시 미적용이 현실적 최적**. `pages/` 재진입/리렌더 비용이 실제 성능 병목이 될 경우를 대비해 follow-up(예: 대용량/공용 환경) 으로 분리.
- Makefile: `bench` 타깃 신설(`make bench`). `scripts/perf_bench.py --rows N --skip-predict` 옵션 지원.
- 신규 파일에 `ruff check` · `black` · `mypy` 모두 0 에러.

### 7.3 실패 경로 QA (NFR-004)

- [x] 빈 파일 / 깨진 CSV / 중복 컬럼 → 안내 메시지, 앱 유지 *(`tests/qa/test_failure_paths.py::test_dataset_upload_rejects_bad_inputs` 3 파라미터(빈 파일 / 빈 헤더 / 바이너리 쓰레기) + `test_validate_columns_rejects_duplicates_directly` 로 `validate_columns` 중복 감지까지 회귀. Service 는 `ValidationError` 를 던지고 파일/감사 로그 정합성 검증)*
- [x] 학습 중 단일 알고리즘 실패 → 나머지 정상, 결과표에 `failed` 표시 *(`test_training_single_algo_failure_keeps_job_completed` (@slow) — `qa_always_fail` AlgoSpec 주입 → `completed` 잡 + `failed/success` 혼재 + best 는 성공 알고리즘에서만)*
- [x] 예측 입력 누락 컬럼 → 차단 + 한국어 안내 *(`test_prediction_missing_column_service_blocks_and_audits` (@slow) — `PredictionInputError` + `prediction.failed` 감사 로그. `test_prediction_missing_column_ui_surfaces_korean_error` — 서비스 패치로 단건 폼 제출 시 `st.error` 에 "누락" 노출 + `at.exception` 은 비어 있음)*

**진행 로그 (2026-04-20)**

- `tests/qa/` 디렉터리 신설 — NFR-004 회귀를 한 파일(`test_failure_paths.py`, 8건)에 묶음. Service 직접 호출 + `streamlit.testing.v1.AppTest` 를 한 파일 안에서 섞기 위해 `tests/services/conftest.py` · `tests/ui/conftest.py` 와 동형의 autouse 엔진 override + 시스템 사용자 시드 + `STORAGE_DIR` 격리 fixture 복제
- pandas 2.x 는 CSV 헤더 중복을 자동 리네이밍(`a,a,b → a, a.1, b`)해 **파일 경로로는 `DUPLICATED_COLUMNS` 재현이 불가**. 회귀는 (a) 실제 업로드로 `FILE_EMPTY`/`HEADER_MISSING`/`FILE_PARSE_FAILED` 를 재현, (b) `validate_columns` 단위로 `DUPLICATED_COLUMNS` 를 별도 회귀로 묶는 2단 구조 채택
- 파일 업로드 실패 시나리오마다 `<storage>/datasets/<project_id>/` 디렉터리가 비어 있는지(롤백)와 감사 로그에 `*_failed` 만 남고 `*_uploaded` 이벤트는 섞이지 않는지까지 검증
- `pytest -q` 296 passed (기존 288 + 신규 8), `ruff check .` / `black --check .` 통과

### 7.4 문서 업데이트

- [x] `README.md` "빠른 시작" 검증 (새 클론에서 재현) *(6단계로 재구성: venv → pip → .env → `init_db --drop --seed` → `generate_samples` → `make run`. 각 스텝을 실제 실행으로 재현 검증. 페이지 구성표/폴더 구조/테스트 섹션(slow 분리: 296→277 fast)/트러블슈팅 추가)*
- [x] `ARCHITECTURE.md` 구현과 달라진 부분 반영 *(문서 버전 0.1→0.2. §3 폴더 트리 완전 재동기화: `06_predict.py`→`06_prediction.py`, `components/layout.py`, `services/dto.py`·`admin_service.py`, `ml/profiling.py`, `utils/{events,messages,db_utils}.py`, `tests/{ui,utils,qa}/`, `samples/`, `Makefile` 추가. §4.2 예측 플로우를 `predict_single`/`predict_batch` 실제 API 로 교체. §4.3 세션 키 테이블에 `current_model_id` 추가. §6.4 아티팩트 4파일(추가 `preprocessor.joblib`) · 롤백 계약 명시. §7 로그/감사 규약(`<domain>.<verb>[_failed]` 접미사) 명시)*
- [x] 샘플 데이터셋(`samples/classification.csv`, `samples/regression.csv`) 커밋 *(`.gitignore` 에 `samples/` 예외 없음 — 현재 저장소가 git 초기화 전이므로 "커밋" 은 저장소 초기화 시점에 반영. 재현성은 `scripts/generate_samples.py` (sklearn iris/diabetes) + `make samples` 로 보장. 151행 / 443행 확인)*

**진행 로그 (2026-04-20)**

- `README.md` 재작성 — 빠른 시작 6단계를 실제 명령으로 검증(`rm -f db/app.db && init_db --drop --seed && generate_samples && sync_streamlit_config && python -c "import app"` 정상 종료). 페이지 구성표(7 페이지 × FR 매핑)와 트러블슈팅 표 추가. 테스트 실행 가이드에 `@pytest.mark.slow` 분리 안내(fast 277건 / 전체 296건)
- `ARCHITECTURE.md` 0.2 로 승격 — 문서 상단 변경 요약을 명시하고, 0.1 단계에서 비어있던 `dto.py`, `admin_service.py`, `profiling.py`, `messages.py`, `events.py`, `db_utils.py`, `tests/qa/` 를 §3 트리와 본문에 일관 반영. 예측 데이터 흐름을 실제 구현(`predict_single`/`predict_batch`, `ModelBundle`, `validate_prediction_input`) 으로 고쳐 쓰고, 세션 키에 `current_model_id` 추가
- `samples/classification.csv`(iris, 150행 + 헤더 = 151행, 타깃 `species`) / `samples/regression.csv`(diabetes, 442행 + 헤더 = 443행, 타깃 `progression`) 재생성 확인. `.gitignore` 는 `samples/` 를 제외하지 않으므로 향후 git 초기화 시 그대로 tracked
- 기존 `Makefile` 의 `smoke` 타깃이 참조하는 `scripts/smoke_train.py` 는 미구현 — README 는 대신 `pytest -q -m "not slow"` 을 스모크 수단으로 안내 (후속 정비 항목)
- 코드 변경 없음 → `ruff check .` 통과, `pytest` 는 직전 296 passed 상태 유지

### 7.5 CI/품질 게이트

- [x] `pytest --cov` 커버리지 60% 이상 (ml/services)
- [x] `ruff check .` 0 에러
- [x] `mypy` 0 에러 (서드파티 `ignore_missing_imports=True`)
- [x] `scripts/init_db.py --drop` 재현성

---

## 단계 8. (선택) 3단계 확장

> **릴리즈 상태 고정 (2026-04-20)** — MVP 를 `v0.1.0` 으로 릴리즈 대상 고정. §8 이하 모든 항목은 **후속 마일스톤**으로 보류하며, 본 MVP 범위는 §0–§7 로 확정한다. 본 섹션의 `[ ]` 는 **계획 상 열어둔 백로그**로, `make plan-check` 는 `in-progress=0` 를 계속 유지한다.

### 8.1 인증 모드 (FR-010~012)

- [ ] `services/auth_service.py` + `User` 테이블 활성화
- [ ] 로그인/로그아웃 페이지
- [ ] `settings.AUTH_MODE="basic"` 스위치로 분기
- [ ] 비밀번호 bcrypt 해시 (passlib)
- [ ] **마이그레이션 (§2.2a 에서 이관)**
  - [ ] 기존 `owner_user_id IS NULL` Project 를 시스템 사용자(또는 지정 계정)로 귀속하는 1회성 마이그레이션 스크립트
  - [ ] `User.password_hash` 컬럼 NOT NULL 전환 (기존 시스템 사용자에는 sentinel hash 부여)
  - [ ] `AuditLog.user_id` 를 `SYSTEM_USER_ID(0)` 에서 실제 로그인 사용자 id 로 대체하는 경로 검증

### 8.2 PostgreSQL 전환

- [ ] `requirements.txt`에서 `psycopg[binary]` 활성화
- [ ] `.env.DATABASE_URL` 교체 후 `init_db.py` 동작 확인
- [ ] 주요 쿼리 성능 점검

### 8.3 비동기 학습 (권장안 B)

- [ ] `training_service.run_training` 을 Celery/RQ 태스크로 래핑
- [ ] Streamlit은 job status 폴링으로 전환

---

## 단계 9. 전처리 고도화 (FR-055~058)

> **릴리즈 타깃**: `v0.2.0` (MVP 이후 첫 기능 이터레이션).
> **범위 고정**: L1 (전략 선택) + L2 (datetime/bool/고카디널리티) + L5 (class_weight + SMOTE) + L6 (피처 변환 미리보기).
> **제외 (후속 이터레이션)**: Target encoding(L3), PolynomialFeatures/상호작용(L3), 상관/분산 기반 피처 선택(L4), 전처리 프리셋 저장(L6 심화). §9.11 에 후속 항목으로 등록.
>
> **하위호환 원칙**: `PreprocessingConfig` 의 모든 필드에 현재 동작(수치: median→standard / 범주: most_frequent→onehot / 불균형: none / 파생: off) 과 동일한 기본값을 부여한다. §3~§7 의 기존 테스트 300건은 코드 변경 후에도 무수정 통과해야 한다.

### 9.0 설계 결정 (고정)

- [ ] `PreprocessingConfig` 는 `ml/schemas.py` 에 `@dataclass(frozen=True, slots=True)` 로 추가. `TrainingConfig.preprocessing: PreprocessingConfig | None = None` 으로 주입 경로 일원화.
- [ ] 전처리 파이프라인은 **여전히 단일 `sklearn.Pipeline` / `ColumnTransformer`** 구조를 유지 (`.cursor/rules/ml-engine.mdc` §전처리). `imbalanced-learn` 의 SMOTE 는 파이프라인 밖 또는 `imblearn.pipeline.Pipeline` 로 분리해 평가 누수 방지.
- [ ] 신규 모듈:
  - `ml/type_inference.py` — datetime / bool / 고카디널리티 감지 (순수 pandas)
  - `ml/feature_engineering.py` — datetime 분해, bool 정규화 (순수 pandas / sklearn transformer)
  - `ml/balancing.py` — SMOTE / class_weight 라우팅 헬퍼 (`imblearn` 가드 포함, 미설치면 `none` 강제)
- [ ] 아티팩트 확장: `<model_dir>/preprocessing_config.json` 추가 (4 → 5 파일 레이아웃). `preprocessor.joblib` 과 **같은 트랜잭션**에서 저장되어야 하며, 로드 시 한쪽만 존재하면 `StorageError`.
- [ ] UI: `pages/03_training.py` 안 `st.expander("고급 전처리 (선택)", expanded=False)`. 모든 위젯은 기본값 = 하위호환 값.
- [ ] 의존성 추가: `imbalanced-learn>=0.12` (extras 가 아닌 메인 requirements). 미설치 환경에서는 SMOTE 옵션이 disabled 로 표시되고 `none/class_weight` 는 정상 동작해야 한다.

### 9.1 스키마 (`ml/schemas.py`) — FR-055

- [x] 전략 Literal 정의:
  - `ImputeNumericStrategy = Literal["median","mean","most_frequent","constant_zero","drop_rows"]`
  - `ImputeCategoricalStrategy = Literal["most_frequent","constant_missing"]`
  - `ScaleStrategy = Literal["standard","minmax","robust","none"]`
  - `OutlierStrategy = Literal["none","iqr_clip","winsorize"]`
  - `CategoricalEncoding = Literal["onehot","ordinal","frequency"]`
  - `ImbalanceStrategy = Literal["none","class_weight","smote"]`
  - 추가: `DatetimePart = Literal["year","month","day","weekday","hour","is_weekend"]`
- [x] `@dataclass(frozen=True, slots=True) class PreprocessingConfig`:
  - [x] 필드: `numeric_impute`, `numeric_scale`, `outlier`, `outlier_iqr_k: float = 1.5`, `winsorize_p: float = 0.01`, `categorical_impute`, `categorical_encoding`, `highcard_threshold: int = 50`, `highcard_auto_downgrade: bool = True`, `datetime_decompose: bool = False`, `datetime_parts: tuple[DatetimePart, ...] = ()`, `bool_as_numeric: bool = True`, `imbalance: ImbalanceStrategy = "none"`, `smote_k_neighbors: int = 5`
  - [x] `__post_init__`: 값 범위 검증 (iqr_k>0, 0<p<0.5, k_neighbors≥1, highcard_threshold≥2), `datetime_decompose=True` 에서 parts 비어 있으면 ValueError. 회귀+SMOTE 크로스 검증은 `TrainingConfig.__post_init__` 으로 이관 (task 미의존 원칙).
  - [x] `to_dict()` / `from_dict()` — 아티팩트 직렬화 (누락 키 기본값 복원, 미지정 키 조용히 무시)
  - [x] `is_default` 속성 + `summary()` 헬퍼 — 기본값 조합 여부 (run_log 기록 분기용)
- [x] `TrainingConfig` 에 `preprocessing: PreprocessingConfig | None = None` 필드 추가. `None` 이면 내부에서 `PreprocessingConfig()` 기본 인스턴스 사용. 회귀 + `imbalance="smote"` 조합은 `ValueError` 로 거부.
- [x] `FeatureSchema` 확장:
  - [x] `datetime: tuple[str, ...] = ()` (원본 datetime 컬럼 이름)
  - [x] `derived: tuple[DerivedFeature, ...] = ()` (신규 `DerivedFeature(name, source, kind)` dataclass, frozen/slots)
  - [x] `to_dict/from_dict` 확장, 기존 직렬화와 하위호환 (누락 필드는 빈 tuple로 복원)

### 9.2 타입 추론 강화 (`ml/type_inference.py`) — FR-056

- [x] `detect_datetime_columns(df) -> list[str]` — `is_datetime64_any_dtype` + object 컬럼은 `pd.to_datetime(..., errors="coerce")` 성공률 ≥95% (`DATETIME_PARSE_SUCCESS_RATIO`) 기준. 수치 문자열(“1”,”42”) 위주 object 컬럼은 to_datetime unit 오인 방지 차원에서 제외.
- [x] `detect_bool_columns(df) -> list[str]` — `is_bool_dtype` / int 고유값 ⊆ {0,1} / object 고유값(lower-stripped) ⊆ {"true","false","t","f","yes","no","y","n","0","1"} 토큰 집합. NaN 이 섞인 int 0/1 은 float 로 승격되어 제외되는 경계도 테스트로 고정.
- [x] `detect_highcard_categorical(df, cols, *, nunique_threshold=50, unique_ratio_threshold=0.3) -> list[str]` — nunique 축 **또는** unique_ratio 축(행수≥2) 어느 하나라도 임계 초과 시 선정. 존재하지 않는 컬럼은 조용히 스킵.
- [x] `skew_report(df, num_cols, *, abs_skew_threshold=1.0) -> dict[str, float]` — L3 후속 작업용 디폴트 off 유틸. 비수치/상수/NaN-skew 컬럼은 스킵, 소수 6자리 반올림. 기본 off — 호출 측에서 원할 때만 사용.
- [x] 단위 테스트 `tests/ml/test_type_inference.py` **19건** (DATETIME/BOOL/HIGHCARD/SKEW 4축):
  - [x] datetime 자동 감지 (datetime64 / ISO 문자열 / 부분 실패 95% 경계 / 수치 문자열 배제 / 전체 NaN 스킵 / 행=0 datetime64 dtype 보존)
  - [x] bool 감지 (네이티브 bool / int 0-1 / NaN 섞인 float 제외 / Y-N·yes-no·true-false 토큰 / 혼합 토큰 거부 / 전체 NaN 스킵)
  - [x] 고카디널리티 감지 (nunique 축 / unique_ratio 축 / 존재X 컬럼 스킵 / 빈 cols iter)
  - [x] skew_report (우측 치우침 / 상수+비수치 스킵 / 대칭 분포 미포함 / 반올림 6자리)

### 9.3 전처리 파이프라인 확장 (`ml/preprocess.py`) — FR-055, FR-056

- [x] 기존 `build_preprocessor(num_cols, cat_cols)` 시그니처 유지 + 오버로드 추가: `build_preprocessor(num_cols, cat_cols, *, config: PreprocessingConfig | None = None, df_sample: pd.DataFrame | None = None, datetime_cols=(), bool_cols=())`
  - [x] `config=None` 이면 기존 동작 = 테스트 회귀 0 (`_build_preprocessor_default` 로 분리, 기존 15건 무수정 통과)
  - [x] 내부 빌더 팩토리: `_build_numeric_pipeline(config)`, `_build_categorical_pipeline_for_encoding(config, encoding)` (+ `_make_cat_transformers` 그룹핑 헬퍼), `_build_datetime_pipeline(config)` (§9.4 의 `DatetimeDecomposer` 미구현 시 `NotImplementedError`), `_build_bool_passthrough()` (sklearn `"passthrough"` 반환)
- [x] 이상치 처리: `IQRClipper(k)` / `Winsorizer(p)` 를 `sklearn.base.BaseEstimator` + `TransformerMixin` 로 직접 구현. `fit` 시 분위수 저장, `transform` 시 `np.clip`. NaN 은 통과(후단 imputer 가 처리). sklearn `clone` 자동 지원 (별도 `__sklearn_clone__` 불필요 — `__init__` 하이퍼만 저장)
- [x] 고카디널리티 자동 라우팅: `plan_categorical_routing(df_sample, cat_cols, config)` 헬퍼로 분리. `categorical_encoding="onehot"` + `highcard_auto_downgrade=True` 이면 `nunique > highcard_threshold` 컬럼을 `frequency` 로 강등, `PreprocessingRouteReport.auto_downgraded` 에 기록. 반환 ColumnTransformer 의 `_route_report_` 속성으로 UI 노출 가능.
- [x] `split_feature_types` 확장:
  - [x] `split_feature_types_v2(df, target, excluded) -> (num, cat, dt, bool)` 4-tuple 신설 (native dtype 기준: `is_datetime64_any_dtype` / `is_bool_dtype`). 기존 `split_feature_types` 는 무수정 유지.
- [x] `build_feature_schema` 확장: `datetime_cols` / `bool_cols` / `config` / `route_report` kwargs 추가. `_enumerate_derived_features` 가 encoding(onehot→컬럼별 카테고리, ordinal/frequency→단일) / datetime_parts / bool_as_numeric 규칙에 따라 `DerivedFeature` 리스트를 산출. `config=None` 기본 경로는 `derived=()`, `datetime=()` 로 기존 동작 유지.
- [x] `FrequencyEncoder` (§9.3 부가): sklearn `BaseEstimator+TransformerMixin` 으로 직접 구현, 학습 빈도 비율(0~1) 매핑, unseen 범주 → 0.0.
- [x] 범주형 파이프라인에 `FunctionTransformer(_coerce_to_object)` 선행 (bool→object 승격) — `bool_as_numeric=False` 경로에서 SimpleImputer 가 bool dtype 을 거부하지 않게.
- [x] 테스트: `tests/ml/test_preprocess.py` 에 §9.3 전용 클래스 6개(`TestSplitFeatureTypesV2`, `TestIQRClipper`, `TestWinsorizer`, `TestFrequencyEncoder`, `TestBuildPreprocessorConfig`, `TestPlanCategoricalRouting`, `TestBuildFeatureSchemaExtended`) / 신규 26건 추가 — **총 41/41 통과, 기존 15건 무수정 통과**, `ml/preprocess.py` 커버리지 97%.

### 9.4 피처 공학 (`ml/feature_engineering.py`) — FR-056

- [x] `DatetimeDecomposer(parts: tuple[str, ...])` transformer (`ml/feature_engineering.py`, sklearn `BaseEstimator + TransformerMixin`)
  - [x] `fit` 은 입력 컬럼명만 `feature_names_in_` / `n_features_in_` 로 기억, `transform` 은 각 입력 datetime 컬럼에 대해 요청된 part 수만큼 float 열을 생성 (지원: `year`, `month`, `day`, `weekday`, `hour`, `is_weekend`; pandas weekday 0=Mon, is_weekend={5,6}). 출력 순서는 입력 컬럼 우선 (`col_part1, col_part2, ...`). `pd.to_datetime(errors="coerce")` 로 문자열도 파싱.
  - [x] NaT → NaN 으로 전파, 후단 `SimpleImputer(strategy="median")` 가 대치 (§9.3 `_build_datetime_pipeline` 에서 이미 결합). `get_feature_names_out(input_features)` 도 제공 → `<col>_<part>` 이름 생성.
  - [x] `parts=()` / unknown part → `ValueError` (Literal 외 방어). sklearn `clone` 자동 지원.
- [x] `BoolToNumeric` transformer — bool dtype(→ float) / 수치형(0·1 유지, 그 외 NaN) / object 토큰(`true/t/yes/y/1` → 1.0, `false/f/no/n/0` → 0.0, 소문자·strip 비교, 미지정 토큰 → NaN). `true_tokens` / `false_tokens` 겹침 시 `ValueError`. `get_feature_names_out` 제공.
- [x] (후속 이월) `LogTransformer`, `YeoJohnsonTransformer` 는 §9.11 L931 에 이미 기록 — 이번 스프린트 구현하지 않음.
- [x] 단위 테스트 `tests/ml/test_feature_engineering.py` **26건** (Datetime 13 / Bool 13): year·month·day·weekday·is_weekend·hour 추출 / NaT→NaN / 문자열 coerce 파싱 / 다중 컬럼 출력 순서 / `get_feature_names_out` / 잘못된 part·empty part → `ValueError` / sklearn `clone` 보존 / SimpleImputer 후단 결합 end-to-end. Bool 측: 네이티브 bool / int {0,1} / int 도메인 밖 → NaN / `yes/no/Yes/ NO `·대소문자·공백 / `true/false/T/F` / unknown 토큰 → NaN / NaN 전파 / 다중 컬럼 / 커스텀 토큰 오버라이드 / 토큰 겹침 rejection / sklearn `clone` / get_feature_names_out / 파이프라인 imputer 결합.
- [x] §9.3 통합: `_build_datetime_pipeline` 의 `NotImplementedError` 가드가 자동 해제됨. `tests/ml/test_preprocess.py` 에 `test_datetime_decompose_true_integrates_with_feature_engineering` 추가 — `datetime_decompose=True` 경로가 end-to-end 로 동작 (num 1 + decompose 2 parts = 3 컬럼, NaN 0). 기존 `test_datetime_decompose_true_guarded_without_feature_engineering` 는 구현 존재 분기로 자동 전환되어 계속 통과.

### 9.5 불균형 대응 (`ml/balancing.py`) — FR-057

- [x] `apply_imbalance_strategy(estimator, X_train, y_train, config, *, task_type="classification") -> (estimator, X_train, y_train)` — `ml/balancing.py` 신설
  - [x] `"none"` → passthrough (입력 객체 그대로 반환)
  - [x] `"class_weight"` → `_apply_class_weight` 가 `estimator.set_params(class_weight="balanced")` 시도. sklearn 이 미지원 시 `ValueError`/`TypeError` → `logger.warning("class_weight 를 지원하지 않는...")` 후 passthrough.
  - [x] `"smote"` → `_apply_smote` 가 `imblearn.over_sampling.SMOTE(k_neighbors=config.smote_k_neighbors, random_state=settings.RANDOM_SEED).fit_resample(X_train, y_train)`. 완료 후 `n_before/n_after/k_neighbors` info 로그.
  - [x] **test 세트에는 호출 금지** — docstring 에 명시, §9.6 호출자(`trainers.train_all`) 가 train_split 이후에만 전달하도록 책임 이전.
- [x] `imblearn` 미설치 가드: 모듈 import 시 try/except 로 `SMOTE_AVAILABLE: bool` 플래그 세팅. `smote` 요청인데 `False` → `MLTrainingError("imbalanced-learn 패키지가 필요합니다...")`. 회귀 + SMOTE 조합도 defense-in-depth 로 `MLTrainingError("회귀(regression) 작업에는 SMOTE 를 적용할 수 없습니다.")` (primary guard 는 `TrainingConfig.__post_init__`).
- [x] `requirements.txt` 에 `imbalanced-learn>=0.12,<1.0  # §9.5 SMOTE (FR-057)` 추가, 실측 설치 버전 `0.14.1` + `sklearn-compat-0.1.5`.
- [x] 단위 테스트 `tests/ml/test_balancing.py` **8건** (≥6 초과 달성): TestNoneStrategy(1) / TestClassWeightStrategy(2: 지원 estimator=LogisticRegression / 미지원 KNN → logger.warning 가로채 검증, automl 로거 `propagate=False` 특성 고려) / TestSmoteStrategy(4: 소수 클래스 리밸런스 검증(`counts[0]==counts[1]`, rows 증가) / `smote_k_neighbors` + `settings.RANDOM_SEED` 가 sampler 생성자에 전달되는지 fake SMOTE 로 확인 / 회귀 task_type → `MLTrainingError("회귀")` / monkeypatch `SMOTE_AVAILABLE=False` → `MLTrainingError("imbalanced-learn")`) / TestUnknownStrategy(1: frozen 우회 후 `"bogus"` 전략 → `ValueError`). `ml/balancing.py` 커버리지 **95%** (41 stmts / 2 miss — 미설치 경로).

### 9.6 학습 파이프라인 통합 (`ml/trainers.py`, `ml/evaluators.py`)

- [x] `train_all(specs, X_train, y_train, *, preprocess_cfg: PreprocessingConfig | None = None, balancer=None)` 시그니처 확장
  - [x] balancer 는 §9.5 의 `apply_imbalance_strategy` 호출 결과를 받아 fit 직전에 적용
  - [x] 기존 호출자(테스트 포함) 는 인자 생략 시 현재 동작 유지
- [x] 아티팩트 저장 (§3.7 / §4.3a 와 결합):
  - [x] `save_model_bundle(...)` 에 `preprocessing_config: PreprocessingConfig | None = None` 인자 추가
  - [x] 존재 시 `preprocessing_config.json` 를 번들 디렉터리에 저장. `load_model_bundle` 은 파일 없을 경우 `PreprocessingConfig()` 기본값으로 복원 (구 모델 하위호환).
- [x] `ModelBundle` dataclass 에 `preprocessing: PreprocessingConfig` 필드 추가 (기본값으로 역호환)

### 9.7 Service 계층 통합 (`services/training_service.py`)

- [x] `run_training(config: TrainingConfig, on_progress=None)` 내부에서:
  - [x] `config.preprocessing` 를 ml 레이어로 forward
  - [x] `run_log` 에 `"preprocessing: <summary>"` 1행 append (`is_default` 이면 `"preprocessing: default"`, 아니면 변경된 축만 나열)
  - [x] 진행 콜백 stage 에 `"feature_engineering"` 추가 (`preprocessing` → `feature_engineering` → `split` → `balance` → `train:<algo>` → ...)
- [x] `services/dto.py`:
  - [x] `PreprocessingConfigDTO` + `from_config(PreprocessingConfig)` / `to_config()`
  - [x] `FeaturePreviewDTO` (§9.9 UI 소비용) — `n_cols_in`, `n_cols_out`, `derived: tuple[tuple[str,str,str], ...]` (source, name, kind), `encoding_summary: dict[str, str]`, `auto_downgraded: tuple[str, ...]`
- [x] 신규 서비스 함수: `services/training_service.preview_preprocessing(dataset_id, config: TrainingConfig) -> FeaturePreviewDTO`
  - [x] 실제 fit 없이 메타데이터 기반 추정 (onehot 차원은 `df[cat_col].nunique()` + 고카디널리티 라우팅 반영)
  - [x] 5만 행 기준 < 2초 (NFR-003)

### 9.8 아티팩트 / 감사 / 하위호환

- [x] `<model_dir>/preprocessing_config.json` 저장/로드 왕복 테스트 1건
- [x] 기존 모델 (v0.1.0 에서 학습된 아티팩트) 로드 시 `preprocessing_config.json` 없음 → 기본값으로 복원 + `log_event("model.legacy_preprocessing_loaded")` 1회 경고
- [x] `AuditLog` action_type 에 `training.preprocessing_customized` 추가 (`is_default=False` 인 학습 시점에 1회) → `utils/events.py::Event` 상수

### 9.9 UI — `pages/03_training.py` (FR-055~058)

- [x] `st.expander("고급 전처리 (선택)", expanded=False)` 내부 위젯 배치:
  - [x] **결측/스케일/이상치**: 3 selectbox + `iqr_k` / `winsorize_p` number_input (조건부 노출)
  - [x] **범주**: 결측·인코딩 selectbox + "고카디널리티 자동 감지" 체크박스(기본 on) + 임계 숫자 입력
  - [x] **고급 타입**: datetime 자동 감지 목록 표시 + 분해 체크박스(off 디폴트) + 분해할 파트 multiselect
  - [x] **불균형 대응** (분류일 때만): `none` / `class_weight` / `smote` 라디오 + `smote_k_neighbors` number_input (조건부), imblearn 미설치 시 `smote` 옵션 disabled
- [x] **피처 변환 미리보기 카드 (FR-058)**:
  - [x] expander 하단에 `st.button("미리보기")` → `training_service.preview_preprocessing(...)` 호출
  - [x] 결과: `st.metric("원본 열", n_cols_in)` / `st.metric("변환 후 열", n_cols_out)`
  - [x] 테이블: `source | kind | derived_name` 목록
  - [x] 자동 다운그레이드 컬럼은 `st.info` 로 1회 고지
- [x] 설정 위젯 변경은 즉시 rerun 되며, `PreprocessingConfig.is_default` 가 아닌 경우 "⚙️ 커스텀 전처리" 뱃지 노출
- [x] UI 테스트 `tests/ui/test_training_page_advanced.py` ≥ 8건:
  - [x] 기본 진입 시 expander 접혀 있음, 모든 값 = 하위호환
  - [x] 스케일 `robust` 선택 후 실행 → TrainingJob.run_log 에 `"scale=robust"` 포함 (@slow)
  - [x] 회귀 + SMOTE 선택 → 검증 에러 flash
  - [x] imblearn 미설치 모킹 → SMOTE 옵션 disabled
  - [x] 미리보기 버튼 → `FeaturePreviewDTO` 렌더 (n_cols_out > n_cols_in 케이스)

### 9.10 문서 · 품질 · 계획

- [x] `ARCHITECTURE.md` §6 (ML 엔진) 갱신 — 전처리 레이어 구성도 + 아티팩트 5파일 레이아웃 반영
- [x] `.cursor/rules/ml-engine.mdc` — `PreprocessingConfig` 주입 규약 1~2 문단 추가
- [x] `README.md` "빠른 시작" 하단에 "고급 전처리" 항목 1단락
- [x] 새 파일 범위 `ruff check` / `mypy` / `black` 0 에러
- [x] `pytest --cov=ml --cov=services` 커버리지 `fail_under=60` 유지 (신규 코드는 80% 이상 목표)
- [x] `make ci` 통과
- [x] `scripts/init_db.py --drop` 재현성 영향 없음 확인 (스키마 변경 없음)
- [x] 계획서 §9 체크박스 + 진행 로그 append + `make plan-check` in-progress=0

### 9.11 후속 범위 (이번 이터레이션 제외, 기록만)

추적 유지를 위해 **구현하지 않지만 잊지 않는다**. 우선순위 제안 순서대로 나열.

- [ ] **Target encoding (L3)** — 회귀 및 이진 분류용. out-of-fold KFold 로 누수 방지 구현 필요. `ml/feature_engineering.py::OutOfFoldTargetEncoder`
- [ ] **수치 변환 (L3)** — `log1p` / `sqrt` / `yeo-johnson` + skew 자동 감지 기반 추천. `skew_report` (§9.2) 는 이번에 미리 배치해 둠.
- [ ] **PolynomialFeatures / 상호작용 (L3)** — `degree=2, interaction_only=True` + `VarianceThreshold`
- [ ] **피처 선택 (L4)** — 분산 필터(`VarianceThreshold`), 상관 필터(|r|>0.95), 후방 중요도 기반 drop
- [ ] **프리셋 저장/공유 (L6 심화)** — `Project.preprocessing_preset_json` 컬럼 추가, "내 전처리 프리셋" 불러오기 UI

---

**수용 기준 (단계 9 전체)**
- `AutoML_Streamlit_MVP.md` FR-055 ~ FR-058 이 전부 UI 에서 실행 가능
- 기본값으로 학습 시 §3~§7 기존 테스트 300건 무수정 통과
- 커스텀 설정으로 학습 시 아티팩트에 `preprocessing_config.json` 이 생성되고 재로드 · 예측까지 정상
- 회귀 + SMOTE 조합은 명확한 에러 메시지로 차단
- imblearn 미설치 환경에서 앱이 기동되며 SMOTE 만 disabled 로 노출

---

## 단계 10. 알고리즘 레지스트리 확장 (FR-067 ~ FR-069)

**범위 확정 (2026-04-23, 사용자 결정)**
1. Tier 1 sklearn 내장 모델 6종을 `ml/registry.py` 에 추가. Tier 2 (SVM/NB/MLP) 는 미포함 → §10.x+ 후속.
2. 알고리즘 다중 선택(체크박스/multiselect) UI 도입. 전체 선택은 `algorithms=None` 으로 정규화해 v0.2.0 과 byte/audit 동치 유지.
3. CatBoost 를 세 번째 optional backend 로 추가. **requirements 분리**: `requirements-optional.txt` 파일 신설에 `catboost>=1.2,<2.0` 기재 (CI 는 기본 requirements 만 설치, 로컬 opt-in).
4. §11 하이퍼파라미터 튜닝은 후속 스프린트지만 **스키마 슬롯만 이번에 선반영**: `AlgoSpec.param_grid` 필드 + `TuningConfig` dataclass + `TrainingConfig.tuning` 필드.

> **포함**: Tier 1 6종 (HistGBM/ExtraTrees/GradientBoosting/KNN/ElasticNet/DecisionTreeRegressor) + CatBoost optional + 후보 선택 UI + §11 스키마 슬롯.
> **제외 (후속 이터레이션)**: Tier 2 (SVM/NB/MLP), 실제 튜닝 실행(ml/tuners.py), UI 튜닝 토글, 본격 param_grid 확장. §10.9 에 후속 항목으로 등록.

### 10.0 설계 결정 (고정)

- [x] 레지스트리 확장은 **기존 `AlgoSpec` + optional backend 패턴을 그대로 유지** — 신규 태스크 함수/모듈 추가 금지(`ml/registry.py` 단일 파일 안에서 해결).
- [x] Tier 1 6종 이름 (task 내 unique):
  - 분류(+3): `hist_gradient_boosting`, `extra_trees`, `gradient_boosting`, + 공통 `kneighbors`
  - 회귀(+5): `hist_gradient_boosting`, `extra_trees`, `gradient_boosting`, `decision_tree`, `elastic_net`, + 공통 `kneighbors`
- [x] CatBoost optional backend 이름: `catboost` (분류/회귀 양쪽).
- [x] `AlgoSpec` 에 신규 필드 2개: `is_optional_backend: bool = False`, `param_grid: Mapping[str, tuple[Any, ...]] | None = None`.
- [x] `TrainingConfig.algorithms: tuple[str, ...] | None = None` (None=전체, 빈 튜플=ValueError, 중복=ValueError).
- [x] `TrainingConfig.tuning: TuningConfig | None = None` — **§10 에서는 스키마만**, `method != "none"` 이면 service 가 downgrade + run_log 경고.
- [x] `param_grid` 는 Tier 1 **6종 전부** 에 2~4개 축의 작은 기본 grid 를 이번에 등록(사용자 결정 B). §11 진입 시 bulk 수정 없이 튜너만 붙이면 됨.
- [x] KNN 은 포함하되 스케일 민감 경고를 `docs/manual/03_training.md` "알고리즘 선택" 소절에 명시(사용자 결정).
- [x] 하위호환: UI 가 전체 선택 시 `algorithms=None` 으로 내려보냄 → 감사/아티팩트 경로가 v0.2.0 과 byte 동치. 기존 `TrainingJob` 레코드/모델 번들은 전부 호환.

### 10.1 레지스트리 확장 (`ml/registry.py`) — FR-068, FR-069

- [x] sklearn 내장 6종 factory 추가 (random_state/n_jobs 기존 컨벤션 준수).
  - `HistGradientBoostingClassifier/Regressor(max_iter=300, learning_rate=0.1, early_stopping="auto")`
  - `ExtraTreesClassifier/Regressor(n_estimators=300, n_jobs=-1)`
  - `GradientBoostingClassifier/Regressor(n_estimators=200)`
  - `KNeighborsClassifier/Regressor(n_neighbors=5, weights="distance", n_jobs=-1)`
  - `ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)`
  - `DecisionTreeRegressor()` (분류 쪽과 비대칭 해소)
- [x] `_try_register_catboost()` 추가 — 기존 `_try_register_xgboost` / `_try_register_lightgbm` 와 동일 패턴. `CatBoostClassifier/Regressor(iterations=300, verbose=0, random_seed=_RANDOM_STATE, allow_writing_files=False)`.
- [x] `optional_backends_status()` 가 `["xgboost", "lightgbm", "catboost"]` 순서로 3건 반환.
- [x] `requirements-optional.txt` 신설: `catboost>=1.2,<2.0` 1줄. `requirements.txt` 는 변경 없음.
- [x] `README.md` 품질 도구 섹션에 "optional backends 설치: `pip install -r requirements-optional.txt`" 안내 추가.

### 10.2 `AlgoSpec` 메타데이터 확장 — FR-067, FR-068

- [x] `AlgoSpec` 에 `is_optional_backend: bool = False` 추가. xgboost/lightgbm/catboost 등록 시 True.
- [x] `AlgoSpec` 에 `param_grid: Mapping[str, tuple[Any, ...]] | None = None` 추가. Tier 1 6종 전부에 기본 grid 등록:
  - HistGBM: `{"learning_rate": (0.05, 0.1, 0.2), "max_iter": (200, 400)}`
  - ExtraTrees: `{"n_estimators": (200, 500), "max_depth": (None, 10, 20)}`
  - GradientBoosting: `{"n_estimators": (100, 200), "learning_rate": (0.05, 0.1)}`
  - KNN: `{"n_neighbors": (3, 5, 10), "weights": ("uniform", "distance")}`
  - ElasticNet: `{"alpha": (0.1, 0.5, 1.0), "l1_ratio": (0.2, 0.5, 0.8)}`
  - DecisionTreeRegressor: `{"max_depth": (None, 5, 10), "min_samples_split": (2, 5)}`
- [x] **불변식 테스트**: task 내 `name` 중복 금지 (`tests/ml/test_registry.py::test_algo_names_unique_per_task`).

### 10.3 `TrainingConfig` 확장 + service 필터링 — FR-067

- [x] `ml/schemas.py` 에 `TuningConfig` dataclass 신설 (frozen, slots). 필드: `method: Literal["none","grid","halving"]="none"`, `cv_folds: int = 3`, `max_iter: int | None = None`, `timeout_sec: int | None = None`.
- [x] `TrainingConfig` 에 `algorithms: tuple[str, ...] | None = None`, `tuning: TuningConfig | None = None` 추가.
- [x] `TrainingConfig.__post_init__`:
  - `algorithms` 가 빈 튜플 → `ValueError("최소 1개 알고리즘을 선택해야 합니다.")`
  - `algorithms` 에 중복 → `ValueError`
  - `tuning is not None and tuning.method != "none"` → 허용(service 층에서 downgrade).
- [x] `services/training_service.py::run_training` 에 specs 필터링 로직 추가 (`_apply_algorithm_filter`). 미등록 이름 포함 시 `ValidationError`.
- [x] `utils/events.py` 에 `TRAINING_ALGORITHMS_FILTERED = "training.algorithms_filtered"` + `TRAINING_TUNING_DOWNGRADED = "training.tuning_downgraded"` 추가. 선택이 전체 미만일 때만 `audit_repository.write` + `log_event` 각 1회.
- [x] `config.tuning` 이 `method != "none"` 이면 `_emit_tuning_downgrade` 로 run_log 에 `tuning=downgraded_v010` append + 실제로는 튜닝 미실행(§11 까지 stub).

### 10.4 서비스 API · DTO — FR-067

- [x] `services/dto.py` 에 `AlgorithmInfoDTO(name, task_type, default_metric, is_optional_backend, available, unavailable_reason)` + `OptionalBackendInfoDTO(name, available, reason)` 추가.
- [x] `services/training_service.py::list_algorithms(task_type) -> list[AlgorithmInfoDTO]` 신설.
  - 등록된 specs + `optional_backends_status()` 를 조합. skip 된 backend 도 `available=False` 로 리스트에 포함(UI 표시용).
- [x] `services/training_service.py::list_optional_backends() -> list[OptionalBackendInfoDTO]` 신설.
- [x] UI 가 `ml.registry` 를 직접 import 하지 않도록 경계 유지(레이어 규칙 준수).

### 10.5 UI — `pages/03_training.py` (FR-067)

- [x] 기존 "고급 전처리 (선택)" expander 위에 "🧪 알고리즘 선택 (선택)" expander 1개 추가.
- [x] 위젯: `st.multiselect("학습 후보", options=available names, default=all available names, key=ALGO_SELECTED_KEY)`.
- [x] 아래 caption 2줄:
  - Optional backend 상태 표 (xgboost/lightgbm/catboost 각각 ✅/⚠️ + reason).
  - 선택 개수 요약. 전체 != 선택 이면 "⚙️ 커스텀 후보 적용됨" 뱃지.
- [x] task 전환 시 상대 task 에 없는 이름은 자동 제거(세션 state stale 방지).
- [x] 제출 시: 선택 == 전체 → `algorithms=None`, 아니면 `tuple(sorted(selected))`.
- [x] `utils/messages.py` 에 `ALGORITHM_SELECT_TITLE`, `ALGORITHM_CUSTOM_BADGE`, `ALGORITHM_BACKEND_UNAVAILABLE` 상수 3개 추가.

### 10.6 테스트 — FR-067 ~ FR-069

- [x] `tests/ml/test_registry.py` +10건: Tier 1 6종 name 존재, `optional_backends_status` 3건, 각 factory smoke(get_params), param_grid 존재 검증, `test_algo_names_unique_per_task`, CatBoost 미설치 시 status.reason 에 "pip install" 포함.
- [x] `tests/ml/test_schemas.py` +3건: `TrainingConfig(algorithms=())` ValueError, 중복 algorithms ValueError, `TuningConfig` 기본값 / `method="grid"` 수용.
- [x] `tests/services/test_training_service.py` +5건: `algorithms=None` 기존동작 회귀 없음 / `algorithms=("random_forest",)` trained_models 1건 / `algorithms=("unknown",)` ValidationError / 필터 적용 시 `training.algorithms_filtered` AuditLog 1건 / 전체 선택 시 0건.
- [x] `tests/services/test_training_service.py` +2건: `list_algorithms` / `list_optional_backends` DTO 반환 구조.
- [x] `tests/ui/test_training_page_algorithms.py` (신규) +5건: 기본 전체 선택 / task 전환 시 stale 제거 / 1개 해제 후 run 시 run_log 에 `algorithms=` 포함 / catboost unavailable monkeypatch 시 caption 문구 / 전체 선택 = v0.2.0 byte 동치 smoke.
- [x] 전체 `make ci` passed, coverage 93% 이상 유지, fail_under=60 충족. (2026-04-24: 508 passed, coverage 93.83%)

### 10.7 문서 · 품질 · 계획 마감

- [x] `AutoML_Streamlit_MVP.md`:
  - FR-062 본문 업데이트 (Tier 1 포함 리스트)
  - **FR-067 알고리즘 후보 선택 UI** 신규 추가
  - **FR-068 Tier 1 모델 확장** 신규 추가
  - **FR-069 CatBoost optional backend** 신규 추가
- [x] `ARCHITECTURE.md` §6.1 레지스트리 — `AlgoSpec` 필드(`is_optional_backend`, `param_grid`, task-unique 불변식) 설명 + optional backend 3종 표.
- [x] `.cursor/rules/ml-engine.mdc` — "AlgoSpec 확장 규약" 섹션 신규 (네이밍 규칙, param_grid 타입, task 당 unique, optional backend 가드 패턴).
- [x] `docs/manual/03_training.md`:
  - "할 수 있는 일" 에 알고리즘 선택 bullet 1줄
  - "알고리즘 선택 (선택)" 소절 신규 — Tier 1 6종 한 줄 설명 표 + KNN 스케일 민감 경고 + optional backend(XGB/LGBM/CatBoost) 설치 안내
  - "자주 겪는 오류" 에 "등록되지 않은 알고리즘" / "최소 1개 선택 필수" 2줄 추가
- [x] `README.md` — optional backend 설치 안내에 `pip install -r requirements-optional.txt` 로 CatBoost 설치 1줄 추가.
- [x] `IMPLEMENTATION_PLAN.md`:
  - §10 모든 체크박스 `[x]` 처리
  - 변경 이력 0.5 행 append
  - 진행 로그 §10.1 ~ §10.7 append
- [ ] `make plan-check` → `OK: in-progress=0` 확인 후 `main` 으로 ff-merge + push.

### 10.8 수용 기준 (단계 10 전체)

- `AutoML_Streamlit_MVP.md` FR-067 ~ FR-069 가 전부 UI 에서 실행 가능.
- 기본 학습(전체 선택) 은 v0.2.0 과 byte/audit 동치 (artifacts/bundle/AuditLog 변경 없음).
- 사용자가 1개 이상 알고리즘을 체크해제한 경우 `training.algorithms_filtered` AuditLog 가 정확히 1회 기록.
- CatBoost 미설치 환경에서도 앱이 기동되며 UI 에 "설치 필요" reason 이 노출되고 선택 목록에서 자동 제외.
- §11 하이퍼파라미터 튜닝을 건드리지 않고도 `TuningConfig` · `AlgoSpec.param_grid` 스키마가 존재.

### 10.9 후속 범위 (이번 이터레이션 제외, 기록만)

- [ ] **§11 하이퍼파라미터 튜닝 본체** — `ml/tuners.py::run_tuning`, service 통합, UI 튜닝 토글, best_params 노출, `pages/04_results.py` 표시
- [ ] **Tier 2 알고리즘** — SVC/SVR (n_samples 가드 + LinearSVC 대체), GaussianNB, MLP(스케일 경고)
- [ ] **`algorithms_excluded` 자동 다운시프트** — 데이터 크기/차원 기반 자동 제외 규칙 (SVM 대용량 skip 등)
- [ ] **CatBoost lazy import** — 첫 import 가 느린 문제(3–5s) 측정 후 필요 시 factory 수준 lazy 전환

---

## 부록 A. FR → 파일 매핑 치트시트

| FR | 주요 파일 | 단계 |
|----|-----------|------|
| FR-001~003 앱 공통 | `app.py`, `utils/session_utils.py`, `utils/messages.py`, `utils/events.py`, `pages/components/toast.py` | 1, 5 |
| FR-010~012 인증 | `services/auth_service.py` (User 정책: 2.2a 참조) | 2, 8 |
| FR-020~024 프로젝트 | `services/project_service.py`, `services/dto.py`, `pages/01_projects.py` | 3, 4, 6 |
| FR-030~035 데이터셋 | `services/dataset_service.py`, `services/dto.py`, `utils/file_utils.py`, `pages/02_dataset_upload.py` | 1, 3, 4, 6 |
| FR-040~045 학습 설정 | `ml/schemas.py`, `services/training_service.py`, `pages/03_training.py` | 3, 4, 6 |
| FR-050~054 전처리 (기본) | `ml/preprocess.py`, `ml/profiling.py` | 3 |
| FR-055~058 고급 전처리 | `ml/schemas.py` (PreprocessingConfig), `ml/preprocess.py`, `ml/type_inference.py`, `ml/feature_engineering.py`, `ml/balancing.py`, `services/training_service.py` (preview_preprocessing), `pages/03_training.py` (고급 전처리 expander) | 9 |
| FR-060~066 학습 | `ml/trainers.py`, `ml/evaluators.py`, `ml/registry.py`, `services/training_service.py` (아티팩트 저장 순서: 4.3a) | 3, 4 |
| FR-067~069 알고리즘 확장 | `ml/registry.py` (Tier 1 + CatBoost optional), `ml/schemas.py` (algorithms/tuning), `services/training_service.py` (list_algorithms/필터링), `services/dto.py` (AlgorithmInfoDTO), `pages/03_training.py` (알고리즘 선택 expander), `requirements-optional.txt` | 10 |
| FR-070~075 결과/모델 | `ml/artifacts.py`, `services/model_service.py`, `services/dto.py`, `pages/04_results.py`, `pages/05_models.py` | 3, 4, 6 |
| FR-080~085 예측 | `services/prediction_service.py`, `pages/06_predict.py` | 4, 6 |
| FR-090~093 이력/관리자 | `repositories/audit_repository.py`, `services/admin_service.py`, `pages/07_admin.py` | 2, 4, 6 |
| 공통 운영 | `Makefile`, `.pre-commit-config.yaml`, `.streamlit/config.toml`, `scripts/sync_streamlit_config.py` | 0 |

---

## 부록 B. 단계 간 의존성 그래프

```
[0 부트스트랩]
      │
      ▼
[1 유틸/인프라] ──────────────┐
      │                      │
      ▼                      ▼
[2 DB/Repository]      [3 ML 엔진]
      │                      │
      └──────────┬───────────┘
                 ▼
          [4 Service]
                 │
                 ▼
          [5 UI 스켈레톤]
                 │
                 ▼
          [6 페이지 구현]
                 │
                 ▼
          [7 품질 QA]
                 │
                 ▼
          [8 (선택) 확장]
```

---

## 리스크 / 이슈 레지스터

*구현 중 막히거나 결정 보류된 항목을 `[!]`로 표시한 뒤 여기에 상세를 남긴다.*

| ID | 날짜 | 단계 | 요약 | 영향 | 대응 | 상태 |
|----|------|------|------|------|------|------|
| R-001 | 2026-04-20 | 3.2 | macOS 에서 `libomp` 누락 시 xgboost/lightgbm import 실패 → 조용히 후보 제외 | 학습 페이지에서 부스팅 모델이 선택지에 안 보이는데 원인이 불투명 | ① `ml/registry.py` 에 `optional_backends_status()` + 구조화 로그 추가 ② `make doctor` 에 상태 노출 ③ 학습 페이지 상단 `st.info` 로 사유 안내 ④ README 트러블슈팅 / 스택 표 갱신 ⑤ `brew install libomp` 실행은 환경별 수동 (README 기재) | **mitigated** |

**템플릿**
```
| R-XXX | 2026-04-18 | 3.5 | 짧은 요약 | 영향 | 대응 | open / mitigated / closed |
```

### 알려진 리스크 (사전 식별)

- **XGBoost/LightGBM macOS 설치** _(R-001, mitigated)_: macOS 에서 `libomp` 누락 시 import 가 실패해 `registry.py` 에서 자동 skip. 스킵된 사유는 `make doctor` / 학습 페이지 상단 `st.info` / 구조화 로그(`registry.optional_backend_skipped`) 세 경로로 가시화된다. 복구: `brew install libomp` 후 앱 재시작.
- **대용량 CSV 성능**: NFR-003 10초 목표. 초과 시 `@st.cache_data` 및 `pd.read_csv(..., usecols=...)` 범위를 조정.
- **Streamlit 재실행 모델**: 세션 상태 의존이 과해지면 디버깅이 어려워진다. 단계 5에서 세션 키 총량을 4개(`SessionKey`)로 고정.
- **SQLite 동시성**: MVP는 단일 사용자. 향후 Postgres 전환 시 트랜잭션 경계 재검토.

---

## 변경 이력

*계획서 자체의 변경 로그. 스코프 변경/단계 추가/항목 이동 시 반드시 기록.*

| 날짜 | 버전 | 변경 내용 |
|------|------|-----------|
| 2026-04-17 | 0.1 | 초안 작성 (단계 0~8, 부록 A/B) |
| 2026-04-17 | 0.2 | 계획서 유지 규칙 / Definition of Done / 샘플 데이터(0.3) / 리스크 레지스터 추가 |
| 2026-04-17 | 0.3 | **added**: 0.2a(네비 방식 고정), 0.2b(config.toml 동기화), 1.7(messages/events 카탈로그), 2.2a(AUTH 정책), 3.1a/b(DTO 분리), 4.3a(아티팩트 저장 순서), Makefile/pre-commit 훅, 6.4 비교표 규격 |
| 2026-04-23 | 0.4 | **added**: §9 전처리 고도화 (FR-055~058) 계획 수립. L1(전략 선택)+L2(datetime/bool/고카디널리티)+L5(class_weight+SMOTE)+L6(피처 변환 미리보기) 범위 확정. 후속 범위(§9.11)에 Target encoding / PolynomialFeatures / 피처 선택 / 프리셋 저장 기록. `AutoML_Streamlit_MVP.md` §6.6 에 FR-055~058, §3.1 포함범위에 '사용자 제어 전처리' 반영. 부록 A 에 FR-055~058 매핑 추가. |
| 2026-04-23 | 0.5 | **added**: §10 알고리즘 레지스트리 확장 (FR-067~069) 계획 수립. Tier 1 sklearn 내장 6종(HistGBM/ExtraTrees/GradientBoosting/KNN/ElasticNet/DecisionTreeRegressor) + CatBoost optional backend + 알고리즘 다중 선택 UI 범위 확정. §11 하이퍼파라미터 튜닝은 스키마(`TuningConfig` + `AlgoSpec.param_grid`) 만 이번에 선반영, 실제 튜너는 후속. CatBoost 의존은 `requirements-optional.txt` 별도 파일로 분리(사용자 결정 B). `param_grid` 는 Tier 1 6종 전부에 기본 grid 등록(사용자 결정 B). 후속 범위(§10.9): Tier 2(SVM/NB/MLP), 튜닝 본체, CatBoost lazy import. 부록 A 에 FR-067~069 매핑 추가. |

---

## 진행 로그

*각 단계 시작/완료 시점을 이 아래에 append. 포맷: `YYYY-MM-DD | 단계 | 상태 | 메모`*
*상태: `started` / `completed` / `blocked` / `resumed` / `note`*

- 2026-04-17 | 단계 0 | started | 부트스트랩 착수 (0.1~0.3 일괄 진행)
- 2026-04-17 | 단계 0.1 | completed | 패키지 스캐폴드(__init__.py 9개) + storage/db/.gitkeep
- 2026-04-17 | 단계 0.2 | completed | app.py/Makefile/pre-commit/dev_run.sh + check_plan_staged.sh
- 2026-04-17 | 단계 0.2a | completed | `pages/` 자동 멀티페이지로 고정, st.switch_page만 사용
- 2026-04-17 | 단계 0.2b | completed | config.toml + sync_streamlit_config.py (350↔200 라운드트립 확인)
- 2026-04-17 | 단계 0.3 | completed | iris/diabetes → samples/*.csv, conftest fixture 노출
- 2026-04-17 | 단계 0 | note | venv(python3.11) 생성, 최소 의존 설치 상태. lint/mypy/pytest 풀 설치는 단계 1 진입 시 `make install` 로 수행
- 2026-04-17 | 단계 0 | note | streamlit run app.py → HTTP 200 확인 (port 8599, headless)
- 2026-04-17 | 단계 0 | completed | 부트스트랩 완료, 단계 1 진입 준비
- 2026-04-17 | 단계 1 | started | 기본 유틸/인프라 일괄 진행 (1.1~1.8), 의존 전체 설치 포함
- 2026-04-17 | 단계 1 | completed | config/settings.py, utils/{errors,log_utils,file_utils,session_utils,messages,events}.py, pages/components/toast.py 구현. tests/utils/* 24건 통과, ruff/black 클린, storage/logs/app.log 기록 확인. 0.2의 `[~]` 항목도 함께 정리.
- 2026-04-17 | 단계 2.1~2.2a | started | ORM base(session_scope) + 엔터티 7종 + AUTH_MODE=none 정책 반영
- 2026-04-17 | 단계 2.1 | completed | repositories/base.py: engine 경로 정규화, SessionLocal, DeclarativeBase, session_scope, TimestampMixin
- 2026-04-17 | 단계 2.2 | completed | repositories/models.py: User/Project/Dataset/TrainingJob/Model/PredictionJob/AuditLog (7 tables). 상위 삭제 cascade 정책 적용
- 2026-04-17 | 단계 2.2a | completed | Project.owner_user_id/AuditLog.user_id nullable, SYSTEM_USER_* 상수 선언 (시드/감사 쓰기는 §2.3/§2.4 에서)
- 2026-04-17 | 단계 2.5 | partial | conftest(sqlite_engine/session_factory/db_session) + test_session_scope.py 6건 통과
- 2026-04-17 | 단계 2.3~2.5 | started | Repository 6종 + init_db 스크립트 + 통합 테스트로 단계 2 종결 진입
- 2026-04-17 | 단계 2.3 | completed | Repository 6종 구현 (project/dataset/training/model/prediction/audit). 함수형, 세션 주입식. 상태/input_type 화이트리스트, §4.3a 보상 로직 지원.
- 2026-04-17 | 단계 2.4 | completed | scripts/init_db.py (--drop/--seed). 시스템 유저 + 샘플 프로젝트 멱등 upsert. 구조화 로그 연동.
- 2026-04-17 | 단계 2.5 | completed | test_project/training_repository 20건 추가. 총 repository 테스트 26건 통과. SQLite PRAGMA foreign_keys=ON 자동 활성화.
- 2026-04-17 | 단계 2 | completed | 전체 pytest 50건 통과, ruff/black 클린, init_db smoke (drop→seed→재실행 멱등) 확인
- 2026-04-17 | 단계 3.1~3.3 | started | ML/Service DTO 분리 + 알고리즘 레지스트리 + 데이터 프로파일링 (전처리는 다음 세션)
- 2026-04-17 | 단계 3.1a | completed | ml/schemas.py 5개 dataclass (frozen/slots). TrainingConfig 자체 검증 포함. FeatureSchema to_dict/from_dict 왕복.
- 2026-04-17 | 단계 3.1b | completed | services/dto.py 10종 DTO + from_orm. tests/conftest.py 로 db_session 전역화.
- 2026-04-17 | 단계 3.2 | completed | ml/registry.py: sklearn 기반 7종 + optional XGB/LGBM. 런타임 로드 실패도 skip 처리.
- 2026-04-17 | 단계 3.3 | completed | ml/profiling.py: profile_dataframe/suggest_excluded. missing·unique 비율 포함.
- 2026-04-17 | 단계 3.1~3.3 | partial | tests/ml/{registry,profiling} + tests/services/test_dto 총 23건. 전체 pytest 73건 통과. ml/ 에 streamlit/sqlalchemy import 0건. 다음: §3.4 전처리
- 2026-04-17 | 단계 3.4~3.6 | done | preprocess/trainers/evaluators + tests/ml/{preprocess,trainers,evaluators,pipeline_e2e} 총 37건. 전체 pytest 110건 통과. iris/diabetes 샘플 CSV 로 end-to-end(전처리→학습→평가→best) 실행 확인. `ml/` streamlit/sqlalchemy import 여전히 0건. 다음: §3.7 artifacts (+ §4.3 과 결합)
- 2026-04-17 | 세션 종료 | checkpoint | 단계 3.6 까지 종결. 110 tests green / ruff·black clean. 다음 세션은 §3.7 부터 (아래 "다음 세션 재개 가이드" 참고)
- 2026-04-20 | 단계 4.1 | started | 루트 C 진입: Project Service 부터 쌓기 시작
- 2026-04-20 | 단계 4.1 | completed | services/project_service.py (CRUD + cascade 가드 + 감사 로그) 구현. project_repository 에 count_datasets/count_models/count_training_jobs + exists_by_name(exclude_project_id) 추가. tests/services/conftest.py (SessionLocal 모킹 + seeded_system_user + tmp_storage) / test_project_service.py 23건 추가. 전체 pytest 133건 green, ruff/black clean.
- 2026-04-20 | 단계 4.2 | started | Dataset Service: 업로드·프로파일·프리뷰·목록·삭제
- 2026-04-20 | 단계 4.2 | completed | services/dataset_service.py 구현. 업로드 파이프라인(project 존재 → ext/size → save → read_tabular → profile → DB insert) + 실패 보상(파일 unlink + dataset.upload_failed 감사). schema_json ↔ DatasetProfileDTO 왕복 헬퍼. preview_dataset 은 df.to_json 경유로 NaN→None 정규화 및 MAX_ROWS 클램프. delete_dataset 은 커밋 이후 best-effort unlink. tests/services/test_dataset_service.py 22건 추가 (해피/실패/프로파일 재구성/프리뷰 클램프/cascade 가드). 전체 pytest 155건 green, ruff/black clean.
- 2026-04-20 | 단계 3.7 + 4.3 | started | Artifacts + Training Service 결합 진행 (§4.3a 보상 설계를 한 번에 검증)
- 2026-04-20 | 단계 3.7 | completed | ml/artifacts.py: save/load_model_bundle(4파일 레이아웃) + ModelBundle(frozen/slots) + validate_prediction_input(누락→ValueError, 추가 drop, 수치 coerce, 범주 str). tests/ml/test_artifacts.py 10건 (저장 파일 검증 / 로드 왕복 / 누락 파일 / 입력 검증 4종). ml/ 은 여전히 utils/errors 비의존(의도) — Service 가 PredictionInputError 로 변환.
- 2026-04-20 | 단계 4.3 | completed | services/training_service.py: run_training / get_training_result / list_training_jobs. 수명주기(pending→running→completed|failed) 와 run_log 스탬프, on_progress(stage, ratio) 동기 콜백. §4.3a 보상 구현: bulk_insert → flush → save_model_bundle → update_paths → mark_best → status 전이를 **단일 트랜잭션**으로 수행, 실패 시 session_scope 롤백 + saved_dirs rmtree + 별도 트랜잭션으로 job=failed 확정. metric_summary_json 에 {status, metrics, error, train_time_ms} 구조로 저장해 실패 행도 DTO 로 복원 가능. tests/services/test_training_service.py 11건 (분류/회귀 해피, 파일 누락 StorageError, 타깃 누락 ValidationError, metric_key 검증, 부분 실패 recorded, 전체 실패 → job=failed, 아티팩트 저장 실패 시 DB 롤백 + 파일 정리 + MLTrainingError, list/get). 전체 pytest 176건 green, ruff/black clean.
- 2026-04-20 | 단계 4.4 + 4.5 | started | Model Service(사후 관리) + Prediction Service 를 결합 진행 (save_model → predict 경로 연결을 한번에 검증)
- 2026-04-20 | 단계 4.4 | completed | services/model_service.py: list_models / get_model_detail(FeatureSchemaDTO + metrics_summary 재구성) / save_model(수동 pin, model_repository.mark_best 위임, 아티팩트 존재 요건) / delete_model(DB 삭제 → `<models_dir>/<id>/` + PredictionJob.result_path rmtree·unlink best-effort) / find_best_model(조회 헬퍼). tests/services/test_model_service.py 10건 (list 포함 관계, detail 스키마/지표 복원, pin 전환 + audit manual=True 기록, 아티팩트 없는 모델 차단, 삭제 시 파일·레코드 동시 정리). ORM 노출 없음 / Streamlit import 0건.
- 2026-04-20 | 단계 4.5 | completed | services/prediction_service.py: predict_single(form dict → 1-row df) / predict_batch(read_tabular → 경고 수집 → validate → predict → `<predictions_dir>/<pj_id>.csv`) 모두 PredictionJob pending→running→completed|failed 전이를 기록하고 감사 로그(Event.PREDICTION_STARTED/COMPLETED/FAILED) 를 남긴다. §10.4 규칙 반영: 누락 컬럼 PredictionInputError 로 차단, 추가 컬럼과 unseen 범주는 warnings 에 누적. 분류 시 `predict_proba` 성공하면 `prob_<class>` 컬럼 자동 부착. 실패 경로는 AppError/일반 예외 모두 `_finalize_failure` 로 묶여 PredictionJob.status='failed' + audit 고정. ValueError → PredictionInputError 변환으로 ml 레이어의 순수성 유지. tests/services/test_prediction_service.py 11건 (iris 단건+proba 합≈1, diabetes 단건, 누락 컬럼 차단, 빈 payload, unknown model, 배치 CSV 저장·복원, extra/unseen 경고, 파일 미존재 ValidationError, to_csv 실패 StorageError 보상(PredictionJob=failed + audit), 모델 path=None 시 NotFoundError). 전체 pytest **197건 green**, ruff/black clean.
- 2026-04-20 | 단계 4.3a | completed | §4.3a 3개 체크박스 마감 — ``model_repository.update_paths`` (plan 네이밍은 ``set_paths`` 였으나 ``feature_schema_json`` 까지 함께 갱신해서 포괄적 명칭으로 구현), ``services/training_service._persist_and_save`` + ``_cleanup_model_dirs`` 로 session_scope 롤백 + 디스크 rmtree 정리, ``mark_best`` 는 모든 저장 완료 후 단일 트랜잭션 내 한 번 호출. 모든 로직은 §4.3 완료 시점에 이미 구현되어 있었고 본 세션에서 체크박스만 정합화.
- 2026-04-20 | 단계 4.6 | completed | Service 레이어 end-to-end 왕복 테스트 1건 추가: tests/services/test_service_e2e.py. 시나리오 = 프로젝트 생성 → 데이터셋 업로드 → 학습(진행 콜백 6단계/ratio 단조 증가/최종 1.0 검증) → 모델 상세 조회 → 수동 pin(is_best 전환 & 단일성 검증) → 단건 예측(proba 합≈1) → 배치 예측(CSV 저장 경로/행수 일치) → 모델 삭제(디렉터리/PredictionJob cascade/결과 CSV best-effort 정리) → 프로젝트 cascade 가드(ValidationError) → 데이터셋 삭제 → 프로젝트 삭제 → 감사 로그에 핵심 이벤트 9종(project.created/dataset.uploaded/training.completed/model.saved/prediction.started/prediction.completed/model.deleted/dataset.deleted/project.deleted) 전부 기록 확인. 전체 pytest **198건 green**, ruff/black clean, `make plan-check` in-progress=0. 단계 4 서비스 계층 전체 종결.
- 2026-04-20 | 단계 5.1 | started | UI 스켈레톤 진입: App 진입점 + 사이드바/헤더 공용 컴포넌트 + DB 초기화 체크
- 2026-04-20 | 단계 6.1 | started | 프로젝트 관리 페이지 착수 (생성/목록/선택/수정/삭제 + 사이드바 연동)
- 2026-04-20 | 단계 6.2 | started | 데이터 업로드 페이지 착수 (업로드 폼/프리뷰/프로파일/목록/삭제 + `data_preview` 컴포넌트 분리)
- 2026-04-20 | 단계 6.4 | started | 결과 비교 페이지 착수 (비교표 + 플롯 + save_model 연동)
- 2026-04-20 | 단계 6.4 | completed | `pages/04_results.py` 신설 + 플롯 데이터 영속화 파이프라인 추가. 구성: **학습 잡 피커**(`list_training_jobs` 최신순 → `LAST_TRAINING_JOB_ID` 기본 선택, 변경 시 즉시 동기화) / **요약 카드**(`st.metric` 3종 + 베스트 `st.success` 메시지에 `↑/↓` 방향 표기) / **성능 비교표**(`_sort_rows` 로 기준 지표 방향대로 정렬, 실패 행은 맨 아래 & 지표 None, `st.dataframe` + `column_config` 로 `★ best` 배지 & metric `format="%.4f"`) / **플롯 섹션**(성공 모델 셀렉터, 기본값은 베스트 인덱스, `model_service.get_model_plot_data(model_id)` 로 load → `kind=confusion_matrix` → plotly `imshow` blues heatmap + `text_auto`, `kind=regression_scatter` → plotly scatter + y=x 대각선. plotly 예외 시 `st.dataframe` 폴백) / **저장 액션**(성공 모델 셀렉터 + `[이 모델 저장]` 버튼, 현재 베스트는 disabled + 라벨 "베스트로 고정됨", 다른 모델 선택 후 클릭 → `model_service.save_model` → is_best 승격 + MODEL_SAVED audit + flash success) / **CTA**(`모델 관리로 이동`/`다시 학습하기` — `st.switch_page` 실패 시 info flash 폴백). **플롯 데이터 영속화**(§3.7/§4.3 확장): `ml/evaluators.build_plot_data(trained, X_test, y_test, task_type)` 추가 — 성공 모델별로 `{kind, ...}` dict 반환(회귀는 2000포인트 균등 샘플링 상한), `training_service._persist_and_save` 가 bundle 저장과 같은 트랜잭션에서 `<model_dir>/plot_data.json` 을 write (쓰기 실패는 `suppress(OSError, TypeError, ValueError)` 로 보조 파일 취급). `services/model_service.get_model_plot_data(model_id) -> dict|None` 로 안전 로드(파일 없음/깨진 JSON → `None`). DTO 확장: `ModelComparisonRowDTO.model_id` + `TrainingResultDTO.task_type` 추가 — UI 가 save_model/plot 조회에 바로 쓸 수 있도록. 메시지 추가: `Msg.TRAINING_RESULT_REQUIRED`, `Msg.MODEL_SAVED`, `Msg.MODEL_SAVE_REQUIRES_SUCCESS`. tests: `tests/ml/test_evaluators.py` +4건(build_plot_data 분류/회귀/실패 스킵/잘못된 task), `tests/services/test_model_service.py` +3건(plot_data.json 존재·로드, 파일 제거 시 None, 존재하지 않는 모델 None), `tests/ui/test_results_page.py` 7건(DB 가드 / 프로젝트 가드 / 학습결과 가드 / [slow] 분류 렌더(카드·success·플롯 셀렉터·저장 셀렉터) / [slow] 회귀 scatter 준비 / [slow] 저장 버튼 → is_best 이동 검증 / [slow] 정렬 렌더). 전체 pytest **256건 green**, ruff/black clean, streamlit smoke (`/`, `/01_projects`, `/02_dataset_upload`, `/03_training`, `/04_results`) 모두 HTTP 200, `make plan-check` in-progress=0.
- 2026-04-20 | 단계 6.3 | started | 학습 설정/실행 페이지 착수 (`st.status` 진행률 + `run_training` 콜백 연동)
- 2026-04-20 | 단계 6.3 | completed | `pages/03_training.py` 신설 + `services/dataset_service.suggest_excluded_columns` 추가 + `ml.profiling.ID_UNIQUE_RATIO_THRESHOLD` public 승격. 페이지 구성: **데이터 확인 섹션**(`render_profile` 재사용, `st.expander` 기본 collapsed) / **데이터셋 피커**(`selectbox` 라벨 `"[id] filename — rows × cols"`, 선택 즉시 `SessionKey.CURRENT_DATASET_ID` 동기화) / **설정 폼**(task_type 라디오 → metric 옵션이 분류/회귀 튜플로 자동 전환, target selectbox, excluded multiselect 는 `suggest_excluded_columns(threshold=0.95)` 를 기본값으로 주입하되 target 컬럼은 후보에서 배제, test_size slider(0.05~0.5), metric selectbox 에 `METRIC_DIRECTIONS` 기반 ↑/↓ help, job_name text_input — 공백만이면 None) / **실행**(폼 제출 → `TrainingConfig` 조립 → `st.status("학습 진행 중...", expanded=True)` 안에서 `training_service.run_training(on_progress=...)` 호출, 콜백은 `progress_bar.progress(ratio, text=stage)` + 최근 6단계 caption 로그) / **완료 처리**(성공 → `status.update(state="complete")` + `SessionKey.LAST_TRAINING_JOB_ID` 세팅 + flash success + 3-metric 카드 + 베스트 배지 + 실패건 expander, 실패 → `status.update(state="error")` + flash error. 결과 페이지(§6.4) CTA 는 `st.switch_page` 시도 후 `StreamlitAPIException` 시 info flash 로 폴백). **세션 stale 보정**: `CURRENT_DATASET_ID` 가 프로젝트의 현재 datasets 리스트에 없으면 렌더 전에 제거. 결정 기록: `st.form` 대신 **개별 위젯 + 하단 실행 버튼** — task_type 변경 시 metric 옵션이 즉시 갱신돼야 하기 때문. Streamlit 위젯 상태 보존 특성상 target 을 런타임에 바꿔도 excluded 기본값은 초회 렌더 시점 고정 (UX 문서화, 테스트도 이에 맞춰 초회 렌더만 검증). tests: `tests/services/test_dataset_service.py` +3건(식별자 힌트 / 임계값 튜닝 / 없는 데이터셋 NotFoundError), `tests/ui/test_training_page.py` 9건 (DB 가드 / 프로젝트 가드 / 데이터셋 없음 가드 / 분류 폼 기본값 / 회귀 전환 후 metric 옵션 / 식별자 컬럼 기본 제외 / [slow] 분류 happy path(`run_training` 실행 후 `LAST_TRAINING_JOB_ID` 세팅 & 요약 메트릭 렌더) / [slow] 회귀 happy path / monkeypatch 로 Service 예외 주입 시 error flash 노출 + LAST_TRAINING_JOB_ID 미세팅). 전체 pytest **242건 green**, ruff/black clean, streamlit smoke (`/`, `/01_projects`, `/02_dataset_upload`, `/03_training`) 모두 HTTP 200, `make plan-check` in-progress=0.
- 2026-04-20 | 단계 6.2 | completed | `pages/02_dataset_upload.py` + `pages/components/data_preview.py` 신설. 구성: **업로드 폼**(`st.form(clear_on_submit=True)` + `ALLOWED_EXTENSIONS`, 빈 파일 제출 → `warning` flash / ValidationError → error flash / 성공 시 `SessionKey.CURRENT_DATASET_ID` 자동 세팅) / **프리뷰 섹션**(`st.tabs([샘플 데이터, 컬럼 프로파일])` — 샘플 탭은 `render_preview` 로 최대 50행 테이블 + `caption` 표시, 프로파일 탭은 `render_profile` 로 행/컬럼 메트릭 2개 + 컬럼별 통계 테이블(`결측`, `결측비율`, `고유값`, `고유비율`)) / **목록**(행별 `st.container(border=True)` + `[선택][삭제]` 컬럼, 현재 선택 데이터셋은 ★ 마커 + select disabled) / **삭제 확인 플로우**(세션 플래그 `datasets_delete_target_id` 기반 인라인 컨펌 블록, cascade 경고 고정, 현재 선택 대상 삭제 시 `SessionKey.CURRENT_DATASET_ID` 자동 해제) / **stale 세션 정리**(목록에 없는 current_dataset_id 는 렌더 전에 제거). 결정: `st.dialog` 대신 §6.1 과 동일 패턴(세션 플래그 + 인라인)로 AppTest 친화 유지. 에러 플래시 UX 정합: 실패 경로 모두 `flash("error") + st.rerun()` 로 같은 사이클 안에서 `render_flashes` 가 소비. tests: `tests/ui/test_dataset_upload_page.py` 10건 (DB 가드 / 프로젝트 미선택 가드 / 목록 렌더 + 사이드바·서브헤더 노출 / 기본 select 미설정 검증 / 탭2+메트릭2 렌더 / 선택 버튼 상호작용 / 빈 파일 제출 warning / 삭제 성공 → DB·파일 제거 / 현재 선택 삭제 시 세션 해제 / 삭제 취소 / stale current_dataset_id 자동 제거). NFR-003 실측: 5만 행 12컬럼 CSV 기준 upload=0.15s / preview(50)=0.110s / profile=0.001s → **end-to-end 0.26s (목표 10s)**. 전체 pytest **230 green**, ruff/black clean, streamlit smoke (`/`, `/01_projects`, `/02_dataset_upload`) 모두 HTTP 200.
- 2026-04-20 | 단계 6.1 | completed | `pages/01_projects.py` 신설. 구성: **생성 폼**(`st.expander` + `st.form`, 성공 시 자동 선택 + 사이드바 즉시 반영) / **목록**(행 별 `st.container(border=True)` + `[선택][수정][삭제]` 3버튼 컬럼, 현재 프로젝트는 ★ 마커 + 선택 버튼 disabled) / **수정 플로우**(세션 `projects_edit_target_id` 기반 인라인 `st.form` 노출, 저장 시 `project_service.update_project`) / **삭제 플로우**(세션 `projects_delete_target_id` 기반 인라인 컨펌 블록 + cascade 체크박스 — 기본값은 children 유무에 따라 결정, cascade=False 로 ValidationError 발생 시 컨펌 블록 유지 후 flash error 로 사용자 재시도 안내, 현재 선택 프로젝트 삭제 시 `SessionKey.CURRENT_PROJECT_ID` 자동 해제). 결정 기록: `st.dialog` 대신 세션 상태 기반 인라인 컨테이너 사용 — `streamlit.testing.v1.AppTest` 환경에서 dialog 내부 위젯 조작이 버전별로 불안정한 반면, 세션 플래그 + 인라인 렌더는 버튼 한 번으로 같은 UX 를 제공하고 테스트가 견고함. 에러 플래시 UX 정합화: 생성/수정/삭제 실패 시 `flash("error", ...)` 직후 `st.rerun()` 호출로 같은 사이클 안에서 `render_flashes` 가 소비하게 처리. tests: `tests/ui/test_projects_page.py` 12건 (DB 미초기화 가드 / 빈 목록 안내 / 생성 해피 + 자동 선택 + 사이드바 success / 이름 공백 ValidationError flash / 이름 중복 flash / 선택 버튼 상호작용 / 현재 프로젝트 row 마커 + select disabled / 수정 저장 + 취소 / 삭제 해피 / cascade 거부 → 1차 실패 + 2차 성공 왕복 / 현재 프로젝트 삭제 시 session 해제). 전체 pytest **220건 green**, ruff/black clean, `streamlit run app.py --server.headless true --server.port 8599` 로 `/` + `/01_projects` 각각 HTTP 200 smoke 확인.
- 2026-04-23 | 단계 9 | note | §9 전처리 고도화 (FR-055~058) 스펙 수립. 범위: L1 전략 선택 / L2 datetime·bool·고카디널리티 / L5 class_weight+SMOTE / L6 피처 변환 미리보기. 후속 범위(§9.11): Target encoding / Polynomial / 피처 선택 / 프리셋 저장. `AutoML_Streamlit_MVP.md` §6.6 FR-055~058 추가, §3.1 포함범위에 '사용자 제어 전처리' 반영. `IMPLEMENTATION_PLAN.md` §9 (9.0~9.11) 블록 · 부록 A · 변경 이력 0.4 동기화. 구현은 별도 브랜치 (`feature/preprocessing-v2` 예정) 에서 착수 예정.
- 2026-04-23 | 단계 9.1 | completed | `feature/preprocessing-v2` 브랜치 분기. `ml/schemas.py` 확장: 전략 Literal 6종 + `DatetimePart` / `DerivedFeature`(frozen/slots, kind=free-form + 권장값 docstring) / `PreprocessingConfig`(전 필드 기본값 = 현 MVP 동작, `__post_init__` 값·조합 검증, `is_default`·`summary()`·`to_dict/from_dict`(누락 키 기본값 복원)). `TrainingConfig` 에 `preprocessing: PreprocessingConfig \| None = None` 추가 + 회귀+SMOTE 크로스 검증 이관. `FeatureSchema` 에 `datetime`/`derived` 필드 추가 + `to_dict/from_dict` 하위호환 확장. 라이브러리 import(`sklearn`/`imblearn`) 도입하지 않음 — transformer 주입은 §9.3/§9.5 로 이관. tests/ml/test_schemas.py 17건 신규 (defaults=MVP 동치 / is_default / summary / iqr_k·winsorize_p·smote_k·highcard_threshold 검증 / datetime_decompose+parts / to-from dict 기본·커스텀·누락 키 / TrainingConfig preprocessing kwarg+None / 회귀+SMOTE 거부 / FeatureSchema 왕복 with datetime/derived / legacy dict 호환 / DerivedFeature frozen). 전체 pytest **351 passed**, ruff/black/mypy clean, `make ci` OK (coverage 93.98%, fail_under=60 충족). `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.2 | completed | `ml/type_inference.py` 신설 — 순수 pandas 유틸 4종: `detect_datetime_columns`(datetime64 + object 컬럼 to_datetime 성공률 ≥95% / 수치 문자열 오인 방지용 numeric coerce ≥0.8 배제 / pandas 포맷 추정 UserWarning 억제) · `detect_bool_columns`(bool dtype / int{0,1} / object 토큰 소문자 비교 — `_BOOL_TOKENS` = {true,false,t,f,yes,no,y,n,0,1}, 혼합 토큰 거부) · `detect_highcard_categorical`(nunique 축 OR unique_ratio 축, 행수<2 일 때 ratio 축 비활성, 없는 컬럼 조용히 스킵) · `skew_report`(수치 아님/존재X/NaN-skew 스킵, threshold 기본 1.0, 6자리 반올림). `Iterable` 은 `TYPE_CHECKING` 블록으로 이동해 런타임 import 최소화. Streamlit/SQLAlchemy/sklearn/imblearn 의존 0건 유지 (ml/ 레이어 규약 준수). tests/ml/test_type_inference.py 19건 신규 — datetime(6) / bool(5) / highcard(4) / skew(4) 커버, 경계 시나리오 고정(95% parse 경계·numeric 문자열 배제·NaN 섞인 int 0-1 float 승격·행=0 datetime dtype 보존). 전체 pytest **370 passed** (기존 351 무수정 통과), `make ci` OK — `ml/type_inference.py` 커버리지 92%, 전체 93.83%, fail_under=60 충족. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.7 | completed | Service 계층 통합. **`services/dto.py` 확장**: `PreprocessingConfigDTO`(frozen/slots, 14 필드 평면화 — tuple→list, `from_config`/`to_config` 양방향 변환, `to_config` 는 ml 레이어 `PreprocessingConfig.__post_init__` 검증을 경유해 DTO 측 잘못된 조합은 `ValueError` 로 차단) + `FeaturePreviewDTO`(frozen/slots, `n_cols_in`/`n_cols_out`/`derived: tuple[tuple[str,str,str], ...]`/`encoding_summary: dict[str, str]`/`auto_downgraded: tuple[str, ...]`). **`services/training_service.py` 확장**: ① `_build_preprocessing` 에서 `config.preprocessing is None` 시 기존 legacy 경로(split_feature_types + build_preprocessor 인자 없이)를 유지해 기존 테스트 11건 무수정 통과 보장, 아니면 `split_feature_types_v2`(num/cat/datetime/bool 4-tuple) + `plan_categorical_routing`(auto_downgrade 반영) + `build_preprocessor(config=..., df_sample=..., datetime_cols=..., bool_cols=...)` + `build_feature_schema(datetime_cols=..., bool_cols=..., config=..., route_report=...)` 로 경로 분기. 반환값 마지막에 `PreprocessingRouteReport | None` 추가. ② `_make_balancer(pp_cfg, task_type) -> BalancerCallable | None` 헬퍼 — `imbalance="none"` 이면 None 반환(train_all 패스스루), 아니면 `apply_imbalance_strategy(estimator, X, y, pp_cfg, task_type)` 를 감싼 per-spec 클로저. **§9.5/§9.6 주의사항 준수**: balancer 호출자(run_training)가 `split_dataset` 이후의 `X_train/y_train` 만 train_all 에 전달하므로 테스트 세트 리샘플링이 원천 차단되는 흐름 docstring 명시. ③ `run_training` 본문: `_STAGE_RATIO` 에 `feature_engineering=0.07`·`balance=0.12` 키 추가. 전처리 완료 직후 `_append_log("preprocessing: {summary}")` (default 면 `"default"`, 아니면 `"numeric_scale=robust ..."` 처럼 변경 축만 나열). `feature_engineering` → `split` → `balance` 순으로 stage emit(balancer 실제 주입 여부와 무관하게 항상 emit 해서 UI 프로그레스 일관성 확보, balancer 가 not None 이면 `balance: strategy=<name>` 추가 로그). `train_all` 호출 시 `preprocess_cfg=pp_cfg_effective, balancer=balancer` 전달. ④ `_persist_and_save(preprocessing_config=...)` kwarg 추가 + 호출부에서 `config.preprocessing is not None and not pp_cfg_effective.is_default` 인 경우에만 `save_model_bundle(..., preprocessing_config=config.preprocessing)` 전달해 **기본 설정 학습 시에는 `preprocessing_config.json` 이 생성되지 않도록 하여 구 모델 바이트 동치 유지**(§9.8 하위호환 이월 동작). ⑤ 신규 `preview_preprocessing(dataset_id, config) -> FeaturePreviewDTO` — TrainingJob 생성 없이 읽기 전용, `read_tabular` 후 `split_feature_types_v2` + `plan_categorical_routing` + `build_feature_schema` 메타데이터만으로 `n_cols_in`/`n_cols_out`(= num passthrough + derived 수) / derived / encoding_summary / auto_downgraded 산출. `bool_as_numeric=False` 면 bool 컬럼을 effective_cat_cols 로 합류시켜 onehot 라우팅을 반영 (UI 기대와 일치). 예외: 데이터셋 부재 `NotFoundError` / 파일 부재 `StorageError` / 타깃 부재 `ValidationError`. **테스트 추가**: tests/services/test_training_service.py +14건 (TestPreprocessingForwarding 6: feature_engineering+balance stage 순서 검증 / default 시 run_log=`"preprocessing: default"` / custom 시 summary 축 나열 / custom 시 preprocessing_config.json 생성 / default 시 미생성 / 회귀+SMOTE TrainingConfig 레벨 거부, TestPreviewPreprocessing 8: iris 수치 전용 derived=∅·n_out=n_in / 고카디널리티 60 unique → auto_downgrade frequency / 저카디널리티 3 unique → onehot 확장 n_out=1+3 / 데이터셋 부재·타깃 부재·파일 부재 에러 3종 / bool_as_numeric=False 로 native bool→ onehot 2 파생 / preview 는 TrainingJob 미생성 / `dataclasses.replace(preprocessing=...)` 로 다른 필드 보존, TestBalancerIntegration 1: `imbalance=class_weight` end-to-end + run_log 에 `balance: strategy=class_weight`). tests/services/test_dto.py +8건 (TestPreprocessingConfigDTO 4: 기본값 / 14필드 왕복 보존 / FrozenInstanceError / `datetime_decompose=True+datetime_parts=[]` 조합을 DTO 는 허용하되 to_config 단계에서 ml 검증으로 차단, TestFeaturePreviewDTO 2: 최소 생성 / frozen). **품질 게이트**: 전체 pytest **466 passed** (이전 444 + 22), ruff/black/mypy 0 에러. `services/training_service.py` 커버리지 91% (258 stmts / 22 miss — 대부분 기존 실패 경로·롤백 분기), `services/dto.py` 99% (175 stmts / 1 miss — optional 경로), 전체 93.67%. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.6 | completed | `ml/trainers.py` + `ml/artifacts.py` §9.6 통합. **trainers 확장**: `train_all(specs, preprocessor, X_train, y_train, *, on_progress=None, preprocess_cfg=None, balancer=None)` 시그니처 확장(기존 위치 인자 순서 보존 → 회귀 0). `balancer: Callable[[estimator, X_train, y_train], tuple[estimator, X_train, y_train]]` 가 주어지면 각 spec 의 fresh estimator 에 대해 fit 직전 1회 호출되어 `(estimator, X_use, y_use)` 를 반환 → 해당 값으로 Pipeline 학습. balancer 호출 중 예외는 기존 `except Exception` 실패 격리 경로에 포함되어 해당 spec 만 `TrainedModel(status="failed")` 로 기록되고 다음 spec 은 진행. `preprocess_cfg: PreprocessingConfig | None` 은 본 구현에서 분기에 직접 쓰이지 않는 예약 파라미터(ruff `ARG001` 허용 주석) — 호출자(Service 레이어, §9.7)가 preprocessor/balancer 구성 책임을 갖고 train_all 은 orchestration 만 담당하도록 경계 유지. **§9.5 통합 주의사항 준수**: balancer 의 `X_train/y_train` 인자는 항상 `split_dataset` 이후의 train split 이어야 한다는 docstring 명시 — 테스트 세트 리샘플링 차단 책임은 호출자에 있음을 문서화. **artifacts 확장**: `ModelBundle` 에 `preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)` 필드 추가(`@dataclass(frozen=True, slots=True)` 유지, 기존 4필드 생성자 호출은 기본값 채움으로 하위호환). `save_model_bundle(..., preprocessing_config: PreprocessingConfig | None = None)` → 제공 시 `preprocessing_config.json` 생성 + `paths["preprocessing"]` 반환, 생략 시 파일 미생성으로 구 모델 디렉터리와 바이트 동치 유지. `load_model_bundle` 은 `preprocessing_config.json` 을 선택(optional) 로 처리: 존재 시 `PreprocessingConfig.from_dict`(누락 키 기본값 복원 로직 재사용), 부재 시 `PreprocessingConfig()` 로 fallback. `_REQUIRED_FILES` 에는 포함하지 않아 구 번들 로드가 `FileNotFoundError` 발생하지 않음. 신규 상수 `PREPROCESSING_FILENAME = "preprocessing_config.json"` + `__all__` 반영. tests/ml/test_trainers.py **+5건** (TestTrainAllBalancer: `preprocess_cfg` 단독 패스스루 / balancer 호출 횟수·인자 형태 검증(spec 당 1회, 입력 X/y row 수 일치) / balancer 가 리샘플 데이터 치환 시 fit 데이터가 치환본으로 변경되고 원본 X/y 는 불변 / balancer 예외가 해당 spec 만 failed 로 격리하고 다음 spec 은 성공 / 두 kwargs 모두 None 일 때 legacy 호출과 결과 동치). tests/ml/test_artifacts.py **+5건** (TestPreprocessingConfigPersistence: ModelBundle 기본 preprocessing = `PreprocessingConfig()` 보장 / save 시 config 생략 → 파일 없음(구 모델 동치) / save 시 config 제공 → JSON 생성 + 반환 paths["preprocessing"] / save→load 왕복으로 outlier=iqr/numeric_scale=robust/bool_as_numeric=False/imbalance=smote/smote_k_neighbors=3 등 모든 축 보존 + 로드 후 예측 가능 smoke / 구 번들(preprocessing 파일 부재) load 시 기본값 복원 + `_REQUIRED_FILES` 체크 무손상). 기존 테스트 무수정 통과(trainers 5건 / artifacts 10건). 전체 pytest **444 passed** (이전 433 + 신규 9 + 회귀 1건 수용), `make ci` OK — `ml/trainers.py` 커버리지 **96%** (54 stmts / 2 miss: `on_progress` 콜백 내부 `contextlib.suppress` 경로 단일 분기), `ml/artifacts.py` **97%** (70 stmts / 2 miss: legacy tuple 변환 드문 경로), 전체 93.71% (fail_under=60 충족). ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.5 | completed | `ml/balancing.py` 신설 + `requirements.txt` 에 `imbalanced-learn>=0.12,<1.0` 추가 (실측 설치 `imbalanced-learn==0.14.1`, 부수 `sklearn-compat==0.1.5`). 공개 API: `apply_imbalance_strategy(estimator, X_train, y_train, config, *, task_type="classification") -> (estimator, X_train, y_train)` + `SMOTE_AVAILABLE: bool` 플래그. 전략별 분기: **none** = passthrough(입력 그대로), **class_weight** = `estimator.set_params(class_weight="balanced")` 시도 후 `ValueError`/`TypeError` 캐치 → `logger.warning("class_weight 를 지원하지 않는 estimator 입니다. 적용을 건너뜁니다.")` + passthrough (KNN 같은 미지원 estimator 보호), **smote** = `_SMOTE(k_neighbors=config.smote_k_neighbors, random_state=settings.RANDOM_SEED).fit_resample(X_train, y_train)` 후 `logger.info("SMOTE 리샘플링 완료", extra={n_before, n_after, k_neighbors})`. **가드**: ① 회귀 + SMOTE → `MLTrainingError("회귀(regression) 작업에는 SMOTE 를 적용할 수 없습니다.")` (primary guard 는 `TrainingConfig.__post_init__`, 여기는 defense-in-depth), ② imblearn 미설치(`SMOTE_AVAILABLE=False`) + SMOTE → `MLTrainingError("SMOTE 를 사용하려면 imbalanced-learn 패키지가 필요합니다. ...")`, ③ 알 수 없는 전략 → `ValueError`. Streamlit/SQLAlchemy 의존 0, `PreprocessingConfig` 는 TYPE_CHECKING 블록으로 import 하여 런타임 최소화. tests/ml/test_balancing.py **8건** — TestNoneStrategy(1) / TestClassWeightStrategy(2: LogisticRegression 적용 / KNN 미지원 시 automl 로거가 `propagate=False` 이므로 `monkeypatch.setattr(balancing.logger, "warning", _capture)` 로 직접 가로채 검증 — caplog 로 안 잡히는 특성 기록) / TestSmoteStrategy(4: 20/200 불균형 → `counts[0]==counts[1]` 리밸런스 & 총 row 증가 / 가짜 `_FakeSMOTE` 로 `k_neighbors=7 & settings.RANDOM_SEED=42` 전달 확인 / 회귀 task_type → `MLTrainingError("회귀")` / `monkeypatch(SMOTE_AVAILABLE=False)` → `MLTrainingError("imbalanced-learn")`) / TestUnknownStrategy(1: `object.__setattr__` 로 frozen 우회 후 `"bogus"` → `ValueError`). **§9.6 통합 책임 이전 주의**: test 세트에는 호출 금지 — docstring 명시, trainers 호출자에서 train_split 이후에만 통과. 전체 pytest **433 passed** (이전 425 + 8), `make ci` OK — `ml/balancing.py` 커버리지 **95%** (41 stmts / 2 miss: 의존 부재 import 블록은 `pragma: no cover`), 전체 93.66%. ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.4 | completed | `ml/feature_engineering.py` 신설 — 순수 pandas/numpy/sklearn 기반 트랜스포머 2종. **`DatetimeDecomposer(parts)`**: sklearn `BaseEstimator+TransformerMixin`, `fit` 은 컬럼명만 기억(`feature_names_in_`/`n_features_in_`), `transform` 은 `pd.to_datetime(errors="coerce")` 후 `_PART_EXTRACTORS` 로 year/month/day/weekday/hour/is_weekend 생성(NaT→NaN 전파, 후단 SimpleImputer 가 처리), 출력 순서는 입력 컬럼 우선 `col_part1, col_part2, ..., col2_part1`, `get_feature_names_out` 로 `<col>_<part>` 이름 반환. `parts=()` / unknown part 는 `ValueError`. **`BoolToNumeric(true_tokens, false_tokens)`**: bool dtype → astype(float), 수치형은 0/1 유지 나머지 NaN(토큰 매핑 도메인 밖), object 는 `_normalize_series` 로 소문자·strip 비교 매핑(defaults: `{true,t,yes,y,1}` / `{false,f,no,n,0}`), 미지정 토큰/NaN→NaN(후단 imputer 가 대치). `true_tokens∩false_tokens≠∅` 시 `ValueError`. sklearn `clone` 은 `__init__` 하이퍼(parts/true_tokens/false_tokens)만 보존해 자동 호환 — 별도 `__sklearn_clone__` 불필요. **§9.3 통합**: `_build_datetime_pipeline` 의 `NotImplementedError` 가드가 자동 해제. `tests/ml/test_preprocess.py::test_datetime_decompose_true_integrates_with_feature_engineering` 추가 — `datetime_decompose=True` + `datetime_parts=(year,month)` 경로가 end-to-end 로 (3,3) shape 산출, NaN 0 검증. 기존 guarded 테스트는 구현 존재 분기로 자동 전환 통과. tests/ml/test_feature_engineering.py **26건** 신규 (Datetime 13: 기본 fit·year/month/day·weekday·is_weekend·hour·NaT 전파·문자열 coerce·다중 컬럼 순서·feature_names_out(fit/explicit)·invalid part·empty parts·clone·Pipeline+SimpleImputer end-to-end / Bool 13: native bool·int0/1·int 도메인 밖→NaN·yes/no·true/false·unknown→NaN·object NaN·다중 컬럼·커스텀 토큰·토큰 겹침 rejection·clone·feature_names_out·Pipeline+SimpleImputer). `§9.11` L931 의 `log1p/sqrt/yeo-johnson` 후속 이월 기록은 기존과 동일 유지 — 이번 스프린트 미구현. 전체 pytest **425 passed** (기존 398 + feature_engineering 26 + preprocess integration 1), `make ci` OK — `ml/feature_engineering.py` 커버리지 87%, `ml/preprocess.py` 97% 유지, 전체 93.63% (fail_under=60 충족). ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-23 | 단계 9.3 | completed | `ml/preprocess.py` 확장 — **하위호환 보존**(기존 `build_preprocessor(num, cat)` 는 `_build_preprocessor_default` 로 분리, `config=None` 기본 경로는 기존 15건 무수정 통과). **신규 API**: `build_preprocessor(..., *, config, df_sample, datetime_cols, bool_cols)` 오버로드 + `plan_categorical_routing(df_sample, cat_cols, config) -> PreprocessingRouteReport`(encoding_per_col + auto_downgraded 기록) + `split_feature_types_v2(df, target, excluded) -> (num, cat, dt, bool)` 4-tuple(native dtype 기준: is_datetime64_any_dtype / is_bool_dtype) + `build_feature_schema(..., datetime_cols, bool_cols, config, route_report)` 확장(`_enumerate_derived_features` 헬퍼로 onehot=컬럼별 카테고리 전개 / ordinal·frequency=단일 / datetime_parts / bool_as_numeric 규칙 산출). **트랜스포머**(sklearn `BaseEstimator + TransformerMixin` 으로 직접 구현 — 별도 `__sklearn_clone__` 없이 clone 자동 지원): `IQRClipper(k)` (Q1/Q3 ± k·IQR clip, NaN 통과) · `Winsorizer(p)` (p/1-p 분위수 clip) · `FrequencyEncoder` (학습 빈도 비율 0~1 매핑, unseen→0.0). **내부 팩토리**: `_build_numeric_pipeline`(impute → optional outlier → scaler, `numeric_impute=drop_rows` 는 방어용 most_frequent 로 fallback, `numeric_scale=none` 은 스케일 스텝 skip) · `_build_categorical_pipeline_for_encoding` (+ `_make_cat_transformers` 그룹핑 헬퍼, encoding 별로 sklearn 인스턴스 분할. `FunctionTransformer(_coerce_to_object)` 선행으로 bool → object 승격 — SimpleImputer 가 bool dtype 을 거부하는 이슈 해소) · `_build_datetime_pipeline` (§9.4 의 `DatetimeDecomposer` 미존재 시 `NotImplementedError("§9.4 ...")`) · `_build_bool_passthrough` (sklearn `"passthrough"` 반환; 네이티브 bool 은 자동 0/1). 고카디널리티 자동 라우팅은 `categorical_encoding="onehot"` + `highcard_auto_downgrade=True` + `nunique > highcard_threshold` 3조건에서 `frequency` 로 강등, ColumnTransformer 의 `_route_report_` 속성으로 부착(§9.9 UI 미리보기 소비). tests/ml/test_preprocess.py **+26건** (TestSplitFeatureTypesV2·TestIQRClipper·TestWinsorizer·TestFrequencyEncoder·TestBuildPreprocessorConfig·TestPlanCategoricalRouting·TestBuildFeatureSchemaExtended) — default=MVP 동치 shape / scale=none·robust·impute=constant_zero·iqr_clip / onehot·ordinal·frequency encoding / 고카디널리티 자동 강등 on/off / bool passthrough / bool_as_numeric=False 경로(FunctionTransformer 로 bool→object 후 onehot) / datetime_decompose=True 가드 / datetime_decompose=False 제외 / 라우팅 계산 (df_sample=None 무강등) / build_feature_schema 확장(기본=빈 derived, onehot 전개, datetime_parts, bool_numeric, route_report 기반 다운그레이드 반영). **전체 pytest 398 passed** (기존 370 + §9.3 신규 28 — 19/3/4/2 + 기존 compat 15 유지), `make ci` OK — `ml/preprocess.py` 커버리지 97%, 전체 94.05%, fail_under=60 충족. ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 재확인.
- 2026-04-24 | 단계 10.5~10.7 | completed | **§10 알고리즘 레지스트리 확장 구현 마감**. §10.5 UI (`pages/03_training.py`): `utils/messages.py` 에 `ALGORITHM_SELECT_TITLE` · `ALGORITHM_CUSTOM_BADGE` · `ALGORITHM_BACKEND_UNAVAILABLE` · `ALGORITHM_REQUIRE_AT_LEAST_ONE` 4개 상수 추가. `pages/03_training.py` 에서 `ml.registry.optional_backends_status` 직접 import 제거 → `training_service.list_optional_backends()` / `list_algorithms(task_type)` 경유로 전환(레이어 경계 강화). 신규 헬퍼 2종: `_normalize_selected_algorithms(available_names, task_type)` — task 전환 시 상대 task 에 없는 세션 값 자동 제거 / `_render_algorithm_selection_expander(task_type)` — "🧪 알고리즘 선택 (선택)" expander 렌더(multiselect + 커스텀 뱃지 + unavailable optional backend 캡션). 전체 선택 시 `algorithms=None` 으로 정규화해 v0.2.0 byte/audit 동치 유지. 0개 선택 시 `Msg.ALGORITHM_REQUIRE_AT_LEAST_ONE` flash + rerun 으로 `TrainingConfig.__post_init__` 진입 차단(UX 빠른 피드백). §10.6 테스트: `tests/ml/test_registry.py` +5건(기존 14→19건: optional 3종 포함 / task 당 name unique / is_optional_backend 플래그 / Tier1 6종 param_grid 존재 / Tier1 factory smoke / core specs param_grid 미보유) · `tests/ml/test_schemas.py` +9건(algorithms 기본 None / 빈 튜플·중복 rejection / roundtrip / TuningConfig 기본·grid 수용·invalid cv_folds·timeout rejection / TrainingConfig+tuning 슬롯) · `tests/services/test_training_service.py` +11건(TestAlgorithmFiltering 5: None=전체 동치 / 단일 선택 / unknown ValidationError / 다중 선택 / tuning downgrade 이벤트 + 학습은 계속 · TestAlgorithmDiscovery 6: list_algorithms 분류·회귀 / unknown task rejection / is_optional_backend 표시 / unavailable reason 포함 / list_optional_backends 3건) · `tests/ui/test_training_page_algorithms.py` **신규 5건**(기본 전체 선택 = 하위호환 / 페이지가 ml.registry 직접 import 하지 않음 / task 전환 시 stale 제거 / 부분 선택이 run_log + audit 에 반영 / unavailable optional backend caption 노출). `make ci` → **508 passed** · coverage **93.83%** · ruff/black/mypy 0 에러. §10.7 문서: `AutoML_Streamlit_MVP.md` 에 **FR-067 알고리즘 선택 UI** / **FR-068 확장 알고리즘 지원** / **FR-069 선택적 백엔드** 3개 FR 신규 추가 + FR-062 확장 목록 반영. `ARCHITECTURE.md` §6.1 — `AlgoSpec` 전체 필드 명세(is_optional_backend / param_grid) + UI 경유 규약(`list_algorithms` / `list_optional_backends`) + 하이퍼파라미터 튜닝 스키마 슬롯(§11) 서술. `.cursor/rules/ml-engine.mdc` 레지스트리 섹션 재작성 — 필수/선택 백엔드 분류, task 당 name unique 불변식, `_try_register_*` 가드 패턴, UI 경계 규칙, `TuningConfig` 다운그레이드 규칙 기재. `docs/manual/03_training.md` — "할 수 있는 일" + "기본 사용 순서" 에 알고리즘 선택 단계 추가, **"알고리즘 선택 (선택) — FR-067~069"** 소절 신규(Tier 1 6종 명세 + KNN 스케일 민감 경고 + CatBoost optional 경고 + 선택 백엔드 상태 캡션 설명) + "자주 겪는 오류" 에 "최소 1개 선택 필수" 1줄 추가 + 아티팩트/감사 설명에 `training.algorithms_filtered` 이벤트 + `TrainingConfig.algorithms` 필드 추가. `README.md` — `pages/03_training.py` FR 범위를 "FR-040~049, 055~058, 060~069" 로 확장. **후속 범위**(§10.9): Tier 2(SVM/NB/MLP), 튜닝 본체(`ml/tuners.py`), `algorithms_excluded` 자동 다운시프트, CatBoost lazy import.
- 2026-04-23 | 단계 10 | note | §10 알고리즘 레지스트리 확장 (FR-067~069) 스펙 수립. 범위: Tier 1 sklearn 내장 6종 (HistGBM / ExtraTrees / GradientBoosting / KNN / ElasticNet / DecisionTreeRegressor) + CatBoost optional backend + 알고리즘 다중 선택 UI(`TrainingConfig.algorithms`). §11 하이퍼파라미터 튜닝은 후속 스프린트지만 **스키마 슬롯만 이번에 선반영** — `AlgoSpec.param_grid` 필드 + `TuningConfig` dataclass + `TrainingConfig.tuning` 필드. 사용자 결정 3건: (1) CatBoost 의존은 `requirements-optional.txt` 분리(CI 는 미설치, 로컬 opt-in), (2) KNN 은 포함하되 스케일 민감 경고를 `docs/manual/03_training.md` 에 적시, (3) `param_grid` 는 Tier 1 6종 **전부** 에 2~4축 기본 grid 등록. 후속 범위(§10.9): Tier 2(SVM/NB/MLP), 튜닝 본체(`ml/tuners.py`), CatBoost lazy import, `algorithms_excluded` 자동 다운시프트. `AutoML_Streamlit_MVP.md` FR-067~069 추가는 §10.7 에서 일괄 수행. `IMPLEMENTATION_PLAN.md` §10 (10.0~10.9) 블록 · 부록 A · 변경 이력 0.5 동기화. 구현은 `feature/algorithms-v2` 브랜치에서 착수.
- 2026-04-23 | 단계 9.10 | addendum | in-app 매뉴얼(`docs/manual/`) 에도 §9 내용을 반영. **§9.10 원 스펙(README.md)** 외 누락이었던 매뉴얼 보강. (1) `docs/manual/03_training.md`: "할 수 있는 일" 에 고급 전처리 + 피처 변환 미리보기 2줄 추가, "기본 사용 순서" 5단계로 확장(3→고급 전처리, 4→미리보기 신설), **"고급 전처리 (선택) 옵션"** 소절 신규 — 10개 축(수치 결측/스케일/이상치·범주 결측/인코딩/고카디널리티 임계·datetime 분해·bool 수치 통과·불균형 전략/SMOTE k) 표 + 커스텀 뱃지/`AuditLog(training.preprocessing_customized)` 설명, **"피처 변환 미리보기 (FR-058)"** 소절 신규 — 메트릭 3개 + source/derived_name/kind 테이블 + auto_downgrade info 설명. "주의사항 & 팁" 에 SMOTE 분류 전용·robust/iqr_clip 권장 케이스 2줄 추가. "개발자 관점" 표 보정: 서비스에 `preview_preprocessing` 추가, ML 목록에 `feature_engineering.py` · `balancing.py` · `type_inference.py` 추가, 스키마에 `PreprocessingConfig` 추가, 저장 레이아웃 4→최대 5개(선택 `preprocessing_config.json`) + 바이트 동치 주석, 감사 행 신설 + FR 범위에 FR-055~058 추가. **"확장 포인트"** 확장 — 새 전처리 축/트랜스포머/불균형 전략 추가 레시피 3개 신설. "자주 겪는 오류" 표에 §9 신규 예외 4개 추가(회귀+SMOTE / imblearn 미설치 / datetime parts 빈 값 / outlier 파라미터 범위). (2) `docs/manual/00_overview.md`: "빠른 사용 순서" 3단계 하단에 "필요하면 고급 전처리 expander + 미리보기" 보조 bullet 1줄 추가 (`FR-055~058`). **검증**: `pytest tests/docs/ tests/ui/test_help.py -q` → 28 passed (매뉴얼 SSOT 파일 존재 / anchor / render_help smoke / 00_manual 검색 smoke 전부 통과). `make ci` → **478 passed** · coverage 93.70% 유지 · ruff/black/mypy 0 에러 · `make plan-check` → `OK: in-progress=0`.
- 2026-04-23 | 단계 9.10 | completed | 문서 · 품질 · 계획 마감. **`ARCHITECTURE.md` §6.2 확장**: 제목을 `ml/preprocess.py` 단독 → `ml/preprocess.py + ml/feature_engineering.py + ml/balancing.py` 3모듈 묶음으로 수정. 기본 정책은 `PreprocessingConfig()` 의 MVP 동치임을 명시. 전처리 데이터 흐름 ASCII 다이어그램(입력 DataFrame → split_feature_types_v2 → 4개 경로(numeric/categorical/datetime/bool) + 각 경로 옵션 축 · plan_categorical_routing 으로 고카디널리티 자동 강등 · ColumnTransformer 의 `_route_report_` 부착 · apply_imbalance_strategy(분류 전용) → Estimator.fit) 추가. `PreprocessingConfig` 불변 dataclass + `is_default` 시 아티팩트·감사 억제로 바이트 동치 유지 문단, SMOTE 의 1차/2차 가드(`TrainingConfig.__post_init__` + `apply_imbalance_strategy`) 설명 추가. **`ARCHITECTURE.md` §6.4 확장**: 파일 레이아웃을 4파일 → 5파일(필수 4: model/preprocessor/feature_schema/metrics · 선택 1: preprocessing_config.json) 로 업데이트. 파일 부재 시 `PreprocessingConfig()` fallback + `Event.MODEL_LEGACY_PREPROCESSING_LOADED` 1회 emit 규약 명시. 필수 4개 누락은 `FileNotFoundError`, `preprocessing_config.json` 만 예외 로 "선택 사항" 으로 규정. **`.cursor/rules/ml-engine.mdc` 확장**: "PreprocessingConfig 주입 규약 (§9.1~§9.9, FR-055~058)" 섹션 신설 — `build_preprocessor` 의 2가지 호출 형태(기본/커스텀) · config 불변 약속 · `PreprocessingConfig()` ≡ `config=None` 동치 · `is_default` 기반 아티팩트 저장 책임은 service 층이라는 경계 · SMOTE/class_weight 의 1차/2차 가드 · `plan_categorical_routing` + `_route_report_` 속성 규약 · `imblearn` 선택 의존/`SMOTE_AVAILABLE` 가드. 아티팩트 섹션도 5파일 + 레거시 fallback 규약 반영. **`README.md` 확장**: "첫 화면에서 해보기" 하위에 "고급 전처리 (선택) — §9" 단락 1개 추가 — 학습 페이지 expander 설명 + 제어 축 목록(결측/스케일/이상치/인코딩/고카디널리티 강등/datetime 분해/bool 수치 통과/분류 불균형) + 미리보기 카드 용도 + "기본값은 v0.1.0 과 바이트 동치(preprocessing_config.json · training.preprocessing_customized 미생성)" 명시. **검증**: (a) 코드/스키마 변경 0 — `git diff --stat main...HEAD -- repositories/ scripts/init_db.py` 빈 출력. (b) `DATABASE_URL="sqlite:///db/_smoke_9_10.db" .venv/bin/python scripts/init_db.py --drop --seed` 정상 수행 — `drop_all.completed` → `create_all.completed | tables=['audit_logs', 'datasets', 'models', 'prediction_jobs', 'projects', 'training_jobs', 'users']` 7테이블 + `seed.system_user.created` + `project.created id=1` + `OK` (임시 DB 는 테스트 후 삭제). (c) `make ci` → **478 passed** · coverage 94% (ml+services 모듈 종합) / 93.70% 전체 (fail_under=60 충족) / ruff + black + mypy 0 에러. 세부 커버리지: `ml/schemas.py` 97%, `ml/preprocess.py` 97%, `ml/artifacts.py` 97%, `ml/balancing.py` 95%, `ml/feature_engineering.py` 87%, `ml/type_inference.py` 92%, `services/training_service.py` 92% — §9 전 신규 코드는 목표치 80% 이상 충족. (d) `make plan-check` → `OK: in-progress=0` — §9.10 8개 체크박스 전부 `[x]`, §9 전체 섹션 9.1~9.10 마감. **§9 전체 스프린트 요약**: 7 섹션 마일스톤(9.1 schemas · 9.2 type_inference · 9.3 preprocess v2 · 9.4 feature_engineering · 9.5 balancing · 9.6 training_service pipeline · 9.7 service api · 9.8 artifacts/audit · 9.9 UI expander · 9.10 문서) 완료. 테스트 총 470 → 478 (+8, §9.9 UI), 기존 §9.8 검증 470 는 변함없이 유지. `§9.11 후속 범위(이터레이션 제외)` 는 계획서 유지 — Target encoding · log1p/yeojohnson · 프리셋 저장 · fit-time optuna 등은 추후 스프린트.
- 2026-04-23 | 단계 9.9 | completed | UI — `pages/03_training.py` 고급 전처리 expander + 미리보기 완료. **메시지**: `utils/messages.py` 에 `PREPROCESSING_ADVANCED_TITLE` / `PREPROCESSING_CUSTOM_BADGE` / `PREPROCESSING_PREVIEW_TITLE` / `PREPROCESSING_PREVIEW_HINT` / `PREPROCESSING_SMOTE_UNAVAILABLE` / `PREPROCESSING_SMOTE_CLASSIFICATION_ONLY` / `PREPROCESSING_PREVIEW_AUTO_DOWNGRADED` 7개 상수 추가 — 페이지 내 한글 리터럴 최소화(자유문구는 expander 내 `st.caption` 설명 텍스트 일부만 로컬 유지). **페이지 확장**: `from ml import balancing as ml_balancing` (모듈 참조로 두어 테스트의 `monkeypatch.setattr(bal_mod, "SMOTE_AVAILABLE", False)` 가 UI 에도 즉시 반영되게 함 — 과거 `from ... import SMOTE_AVAILABLE` 복사 패턴 문제 회피) + `ml.feature_engineering.DEFAULT_DATETIME_PARTS` + `ml.schemas.PreprocessingConfig` import. 위젯 key 17개 (`PP_NUM_IMPUTE_KEY` … `PP_PREVIEW_RESULT_KEY`) Final 상수화 + 옵션 튜플(`_NUM_IMPUTE_OPTIONS` / `_NUM_SCALE_OPTIONS` / `_OUTLIER_OPTIONS` / `_CAT_IMPUTE_OPTIONS` / `_CAT_ENCODING_OPTIONS` / `_IMBALANCE_OPTIONS_CLASSIFICATION` / `_IMBALANCE_OPTIONS_REGRESSION`) 선언. **섹션 헬퍼 5종 분리** (ruff C901 complexity 충족): `_render_numeric_section` (impute/scale/outlier 3-col selectbox + `iqr_clip` 선택 시 `outlier_iqr_k` number_input / `winsorize` 선택 시 `winsorize_p`) · `_render_categorical_section` (impute/encoding + `highcard_auto_downgrade` 체크박스(기본 True) + `highcard_threshold` 숫자) · `_render_advanced_types_section` (`datetime_decompose` 체크 on 시 `DEFAULT_DATETIME_PARTS` multiselect + `bool_as_numeric` 체크) · `_render_imbalance_section` (`_imbalance_options(task_type)` 결과로 옵션 축소 + 사유 caption — 회귀는 `("none",)` + "분류 작업에서만" 문구, 분류+imblearn 미설치는 smote 제외 + "imbalanced-learn 미설치" 문구; 세션 stale smote 값은 옵션 바깥이면 "none" 으로 정규화) · `_handle_preview_click` (`_collect_preprocessing_config` → `TrainingConfig(preprocessing=...)` 조립 → `training_service.preview_preprocessing(dataset_id, cfg)` 호출 → `AppError` / `ValueError` 는 `flash("error", ...)` + 결과 초기화). **메인 헬퍼**: `_render_advanced_preprocessing_expander(task_type, dataset_id)` — 현재 `PreprocessingConfig.is_default` 가 False 면 expander 라벨에 "⚙️ 커스텀 전처리 적용됨" 점 구분 suffix 추가(닫힌 상태에서도 커스텀 시각 단서), 섹션 5종 호출 후 divider → `PREPROCESSING_PREVIEW_HINT` caption + "미리보기" 버튼 → 저장된 `pp_preview_result` 가 있으면 `_render_preview_result` 렌더. `_render_preview_result` 는 `st.metric("원본 열", n_cols_in)` / `st.metric("변환 후 열", n_cols_out)` / `st.metric("파생 피처", len(derived))` 3개 + `auto_downgraded` 존재 시 `st.info` (컬럼명 ", " 조인) + `st.dataframe([{source, kind, derived_name}, ...], width="stretch", hide_index=True)` (파생 0 이면 caption 대체). **`_collect_preprocessing_config(task_type)`**: 세션 상태에서 14개 필드를 읽어 `PreprocessingConfig` 조립, 회귀+smote 잔존 시 강제 none 정규화, `ValueError` 발생 시 flash + None 반환(호출자는 실행을 건너뜀). **`_render_config_form` 통합**: job_name 아래에서 `_render_advanced_preprocessing_expander(task_type, dataset_id)` 호출 → `pp_cfg` 확보, `is_default=False` 면 expander 바깥에도 `st.caption(PREPROCESSING_CUSTOM_BADGE)` (테스트 captions 탐지 경로). 학습 실행 제출 시 `pp_cfg is None` 이면 `st.rerun` + 학습 건너뜀. `TrainingConfig(..., preprocessing=pp_cfg if not pp_cfg.is_default else None)` — 기본값은 **명시적으로 None 으로 떨어뜨려** 기존 하위호환 경로(§9.6: `preprocessing_config.json` 미생성 + §9.8: `training.preprocessing_customized` AuditLog 미기록)를 그대로 유지. **테스트**: `tests/ui/test_training_page_advanced.py` 8건 — `test_expander_defaults_are_backward_compatible` (위젯 초기 세션에서 14개 필드를 재조립 → `PreprocessingConfig.is_default==True`) / `test_regression_hides_smote_option` (회귀 전환 후 `pp_imbalance` radio options 튜플이 `("none",)` 로 축소) / `test_regression_forces_imbalance_to_none_even_if_session_had_smote` (세션에 과거 "smote" 가 남아있어도 회귀 렌더 후 "none" 으로 덮어쓰기) / `test_smote_option_hidden_when_imblearn_missing` (`monkeypatch.setattr(ml.balancing, "SMOTE_AVAILABLE", False)` → `pp_imbalance.options` 에 "smote" 부재 + `at.caption` 에 "imbalanced-learn" 포함) / `test_preview_button_renders_feature_preview` (혼합 CSV: num×2 + cat_low(3) + cat_high(nunique=50, unique_ratio=0.5 → suggest_excluded 제외) + bool + datetime → 미리보기 클릭 후 metric 3종 label + `pp_preview_result` 가 `FeaturePreviewDTO` + `n_cols_out > n_cols_in`) / `test_preview_auto_downgrade_shows_info` (임계 5로 낮춤 → cat_high 가 `auto_downgraded` 에 포함 + info 문자열에 "frequency" 존재) / `test_custom_preprocessing_badge_caption_appears` (`pp_numeric_scale="robust"` 세션 세팅 후 run → captions 에 "커스텀 전처리") / `@slow test_robust_scale_propagates_to_run_log` (robust 로 실제 학습 → `TrainingJob.run_log` 에 "numeric_scale=robust" 포함). **테스트 픽스처**: `_mixed_csv` 에서 `cat_high` 는 `f"v{i%50}"` 로 만들어 nunique=50 / unique_ratio=0.5 — `suggest_excluded_columns(ID_UNIQUE_RATIO_THRESHOLD=0.95)` 제안 대상 아님을 확인 후 auto_downgrade 테스트 케이스 성립. SafeSessionState 는 `.get()` 을 지원하지 않으므로 테스트에서 `"key" in at.session_state` + subscript 사용. **품질**: `_render_advanced_preprocessing_expander` 를 5개 섹션 헬퍼로 쪼개 C901 complexity=12 → <10 해소. ruff auto-fix 로 import 정렬(I001) 자동 교정. `make ci` **478 passed** (§9.8 470 → §9.9 +8), coverage 93.70% 유지(fail_under=60 충족), `services/training_service.py` 92% / `ml/balancing.py` 95% / `ml/preprocess.py` 97%, ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 유지.
- 2026-04-23 | 단계 9.8 | completed | 아티팩트 / 감사 / 하위호환 통합. **`utils/events.py` 확장**: `TRAINING_PREPROCESSING_CUSTOMIZED = "training.preprocessing_customized"` · `MODEL_LEGACY_PREPROCESSING_LOADED = "model.legacy_preprocessing_loaded"` 상수 2개 추가(AuditLog.action_type + 구조화 로그 공통). **`ml/artifacts.py` 확장**: `utils.log_utils.get_logger` + `log_event` + `utils.events.Event` import. `load_model_bundle` 의 `preprocessing_config.json` 부재 분기에서 `log_event(logger, Event.MODEL_LEGACY_PREPROCESSING_LOADED, model_dir=str(source_dir))` 1회 emit — 파일이 존재하면 호출되지 않음(정상 로드 경로에는 noise 미발생). `ml/` 레이어 규칙 준수: streamlit/sqlalchemy 비의존 유지(이미 `ml/balancing.py`, `ml/registry.py` 가 `utils.log_utils` 사용 중 — 허용 패턴). **`services/training_service.py` 확장**: `run_training` 의 preprocessing 요약 run_log append 직후 동일 `session_scope` 내에서 `pp_cfg_effective.is_default` 가 False 일 때만 `audit_repository.write(action_type=Event.TRAINING_PREPROCESSING_CUSTOMIZED, target_type="TrainingJob", target_id=job_id, detail={"summary": pp_cfg_effective.summary()})` + 동일 이벤트로 `log_event` 1회. **하위호환 보장**: 기본 config 학습은 ① `preprocessing_config.json` 미생성(§9.6) + ② `training.preprocessing_customized` AuditLog 미기록 — 두 축으로 v0.1.0 모델과 바이트·감사 동치 유지. `<model_dir>/preprocessing_config.json` 저장/로드 왕복 테스트는 §9.6 의 `TestPreprocessingConfigPersistence::test_load_roundtrips_preprocessing_config` 가 이미 커버 → §9.8 체크박스 1 충족으로 표식. **테스트**: `tests/ml/test_artifacts.py::TestLegacyPreprocessingLoggedOnce` 2건 신규 — legacy 번들 로드 시 `log_event` monkeypatch 으로 `MODEL_LEGACY_PREPROCESSING_LOADED` 정확히 1회 캡처 + `model_dir` extra 일치 / 모던 번들(preprocessing_config 있음) 로드 시 legacy 이벤트 0건 확인. `tests/services/test_training_service.py::TestPreprocessingForwarding` 에 2건 추가 — 커스텀 config 학습 후 `audit_repository.list_logs(action_type="training.preprocessing_customized")` 를 `target_id=job_id` 로 필터해 정확히 1건 + `detail_json["summary"]` 에 `numeric_scale=robust` / `imbalance=class_weight` 포함 검증 / 기본 config 학습 후 동일 이벤트 0건 확인. **전체 pytest 470 passed** (§9.7 466 → §9.8 +4), `make ci` OK — `ml/artifacts.py` 97%, `services/training_service.py` 92%, 전체 93.70%, fail_under=60 충족. ruff/black/mypy 0 에러. `make plan-check` → `OK: in-progress=0` 유지.
- 2026-04-20 | 단계 5.1 | completed | app.py 재작성 + `pages/components/layout.py` 신설 + `utils/db_utils.py` 추가. 구성: `configure_page`(set_page_config 래퍼, 페이지 재실행에 대비해 `StreamlitAPIException` suppress) / `render_sidebar`(DB 뱃지·현재 프로젝트 카드·선택 해제·네비 리스트) / `render_page_header`(title+caption+`render_flashes`). 홈 본문은 서비스 소개 → 최근 프로젝트 3장 카드(`project_service.list_projects[:3]`, "선택하기" 클릭 시 `SessionKey.CURRENT_PROJECT_ID` 갱신 + flash("success") + `st.rerun()`) → "프로젝트 페이지로 이동"/"문서 보기" CTA. DB 초기화 체크는 `utils.db_utils.is_db_initialized` 로 inspector 경유 7개 필수 테이블 존재 검사, 미초기화 시 본문+사이드바 양쪽에서 `init_db` 명령 안내. Stale project id 는 `get_project` `NotFoundError` 를 잡아 세션에서 조용히 제거. tests: `tests/utils/test_db_utils.py` 5건(전체·부분·공DB·깨진 엔진·REQUIRED_TABLES 회귀), `tests/ui/test_app_home.py` 5건 (Streamlit `AppTest` 기반: DB 미초기화 안내 / 프로젝트 없을 때 인트로 / 3개 카드 노출 + 선택 상호작용 / stale id 자동 제거 / 사이드바 현재 프로젝트 표시). **전체 pytest 208건 green**, ruff/black clean. `streamlit run app.py --server.headless true --server.port 8599` → `/_stcore/health=ok`, `/=HTTP 200` smoke 성공.

---

## 다음 세션 재개 가이드 (Resume)

### 현재 완료 상태 (2026-04-17 체크포인트)
- **완료**: 단계 0 (부트스트랩), 1 (유틸·세션), 2 (DB·Repository), 3.1~3.6 (ML 스키마/DTO/레지스트리/프로파일링/전처리/학습/평가)
- **검증**: `pytest -q` → **110 passed**, `ruff check .` clean, `black --check` clean
- **불변식**: `ml/` 아래 `streamlit`/`sqlalchemy` 실 import 0건 (docstring 언급만 있음)

### 즉시 실행 가능한 샘플 파이프라인
샘플 CSV 기반 end-to-end 학습/평가는 `tests/ml/test_pipeline_e2e.py` 참조.
순수 `ml/` 모듈만으로 아래 순서가 돌아감을 확인:
`prepare_xy → split_feature_types → build_feature_schema → build_preprocessor → split_dataset → train_all → score_models → select_best`

### 다음 세션 시작 전 준비 체크리스트
1. `source .venv/bin/activate` (Python 3.11)
2. `pytest -q` → 110 passed 기대. 깨지면 환경/의존성 먼저 복원.
3. `make plan-check` → `OK: in-progress=0` 확인.
4. `IMPLEMENTATION_PLAN.md` 의 §3.7 / §4.x 체크리스트를 참조점으로 삼는다.

### 다음 작업 후보 (우선순위 제안 순)
- **A. §3.7 `ml/artifacts.py` 단독** — `save_model_bundle` / `load_model_bundle` / `validate_prediction_input` + `test_artifacts.py` 왕복 테스트. 단계 3 종결.
- **B. §3.7 + §4.3 Training Service 결합** — 아티팩트 저장 순서(§4.3a 보상 로직: DB insert → 파일 저장 → 실패 시 model_repository.delete / update_paths) 와 함께 설계하면 저장 경로 불일치 리스크가 한 번에 해결됨. **추천.**
- **C. §4.1 Project Service / §4.2 Dataset Service 선행** — UI(§5~6) 진입 전 가장 작은 서비스 단위부터 쌓는 루트. §4.3 보다 먼저 하면 Training Service 가 의존할 Project/Dataset 조회가 준비됨.

가장 안전한 루트는 **C → B** 순서 (Project/Dataset Service → Artifacts+Training Service). 속도를 우선하면 **B** 로 직진.

### 참고 파일 포인터
- 규약/불변식: `.cursor/rules/ml-engine.mdc`, `docs/AutoML_Streamlit_MVP.md`
- ML 내부 계약: `ml/schemas.py` (TrainingConfig, FeatureSchema, ScoredModel)
- 서비스→UI 계약: `services/dto.py`
- Repository 사용법: `repositories/*.py` (전부 함수형, `session: Session` 주입식)
- DB 부트스트랩: `scripts/init_db.py --drop --seed` (개발용 멱등 리셋)

### 오픈 이슈/리스크
- R-001 (PLAN §리스크 표): XGBoost import 실패(macOS libomp) — 레지스트리에서 `Exception` catch 로 skip 처리됨. 설치된 환경에서는 자동으로 목록에 포함.
- stratify fallback 은 소규모 데이터셋 안정성 용도. UI 단에서 "클래스 분포 경고"를 띄울지 §4.3/§5 설계 시 결정 필요.
- `ml/evaluators.py` 의 roc_auc 는 `predict_proba` 부재/실패 시 조용히 누락됨. UI 표 렌더링 시 `"-"` 표시 규칙은 §6.3(결과 화면) 에서 합의 필요.

### 진행 로그 — §7.5 CI/품질 게이트 (2026-04-20)

**완료 항목**

1. **mypy 0 errors 달성** — 48 source files 통과.
   - `repositories/base.py::_enable_sqlite_foreign_keys`, `services/admin_service.py::_scalar_count`/`list_logs` 에 타입 주석 추가 (`dbapi_connection: Any`, `column: Any`, `**kwargs: Any`).
   - 더 이상 필요 없는 `# type: ignore[import-not-found]` (ml/registry.py xgboost/lightgbm 2건) · `# type: ignore[arg-type]` (services/training_service.py get_specs 1건) 제거 → `warn_unused_ignores` 경고 해소.
   - `pages/components/layout.py::configure_page` 의 `layout: str` 을 `Literal["centered", "wide"]` 로 좁혀 `st.set_page_config` 오버로드 일치.
   - `pages/components/data_preview.py::render_preview/render_profile` — `height: int | None` 을 내부 래퍼 `_render_df` 로 분기(None 일 때 인자 제거)해 `st.dataframe` 오버로드 만족.
   - `pages/05_models.py::render_bundle_details` — 바깥 루프의 `col: str` 과 겹치던 내부 `for col, (metric_key, value) in zip(...)` 을 `metric_col` 로 리네이밍 (shadow 제거).

2. **ruff check / black 통과** — 93 files, 0 errors.

3. **pytest --cov 커버리지 게이트 자동화**.
   - `pyproject.toml` 에 `[tool.coverage.run] source = ["ml", "services"]` + `[tool.coverage.report] fail_under = 60` 추가.
   - 실측: **전체 수트 296 passed, ml+services 커버리지 92.95%** (fast-only 로도 93%).
   - `exclude_lines` 에 `pragma: no cover`, `if TYPE_CHECKING:`, `if __name__ == "__main__":` 포함.

4. **`make ci` 통합 게이트 도입**.
   - `Makefile` 에 `test-fast`, `cov`, `ci` 타깃 신설. `ci = lint + cov` (ruff → mypy → pytest --cov 순차).
   - `smoke` 타깃이 가리키던 미존재 `scripts/smoke_train.py` 를 `pytest -q -m "not slow"` 로 교체 — §7.4 에서 기록한 follow-up 해소.
   - README "품질 도구" 섹션에 신규 타깃(`make ci`, `make cov`, `make test-fast`) 사용법 추가.

5. **`scripts/init_db.py --drop` 재현성 확인**.
   - 삭제 → `--drop --seed` 1회 → 동일 명령 2회차 → 플래그 없이 3회차 실행까지 모두 "OK" 반환 (멱등).
   - 결과물: `audit_logs`, `datasets`, `models`, `prediction_jobs`, `projects`, `training_jobs`, `users` 7개 테이블 + system user + sample project.

**검증**

```bash
$ make ci
ruff check .            # All checks passed!
mypy .                  # Success: no issues found in 48 source files
pytest --cov=ml --cov=services --cov-report=term-missing
...
Required test coverage of 60.0% reached. Total coverage: 92.95%
======================== 296 passed in 66.34s (0:01:06) ========================
[ci] quality gate OK (ruff + mypy + pytest --cov fail_under=60)
```

**팔로업 (차회 PR 후보)**

- GitHub Actions / GitLab CI 워크플로 파일 추가 — 현재는 로컬 `make ci` 만 보장.
- 커버리지 HTML 리포트(`--cov-report=html`) 생성 위치(`htmlcov/`) `.gitignore` 등록 여부 재검토.
- `tests/ui/` 의 `AppTest` 기반 테스트를 CI 에서 headless 로 안정 돌릴 수 있는지 별도 검증.

### 진행 로그 — §7 마무리 점검 (2026-04-20)

§8 진입 전 PLAN 전수 감사 수행. 체크박스와 실제 코드의 불일치를 해소하고 차기 이월/보류 사유를 명시했다.

**체크박스 보정 (PLAN-현실 불일치 해소, 3건)**

1. §3.8 L369 `test_artifacts.py` → [x] — `tests/ml/test_artifacts.py` 10건(4파일 생성 / roundtrip / missing dir·file 에러 / schema JSON / `validate_prediction_input` 5시나리오) 이 이미 존재.
2. §5.3 L499 `require_project()` / `require_dataset()` 헬퍼 → [x] — `utils/session_utils.py:70,79` 에 구현.
3. §5.3 L500 `st.info + st.stop` 일관 적용 → [x] — 위 헬퍼가 `st.warning(Msg.*)` + `st.stop()` 로 전 페이지에서 공통 호출.

**표식 규약 도입 ([-] 보류 / [→] 이관 / ☐ DoD 템플릿)**

- `[-]` (의도적 보류, scope 조정): §5.2 `project_picker.py`, `metric_cards.py`, `plots.py` — 대체 구현 위치 명시.
- `[→]` (다른 섹션으로 이관): §2.2a `AUTH_MODE=basic` 마이그레이션 3항목 — §8.1 에 본체 추가 후 원출처는 이관 표시.
- `[-]` (차기 이월, 선택): §1.7 ruff custom rule — `Msg` 강제는 rules + 리뷰로 충분, custom ast 탐지기 구현비용 대비 MVP 외.
- `☐` (DoD 템플릿, 반복 적용 기준): §루트 "Definition of Done (모든 항목 공통)" 7줄을 `[ ]` → `☐` 로 전환. `make plan-check` 카운트에서 제외됨을 명시.

**잔여 열린 항목 집계**

- `make plan-check` → **in-progress=0** (모든 `[~]` 해소 유지)
- `[ ]` 잔여 **13건** 모두 §8 하위 — MVP 확장 범위로 의도된 상태:
  | 섹션 | 건수 | 내용 |
  |-|-|-|
  | §8.1 인증 (FR-010~012) | 8 | auth_service · 로그인·로그아웃 · bcrypt · AUTH_MODE 전환 + §2.2a 에서 이관된 마이그레이션 3건 |
  | §8.2 PostgreSQL 전환 | 3 | psycopg 활성화 · DATABASE_URL 교체 · 쿼리 성능 점검 |
  | §8.3 비동기 학습 | 2 | Celery/RQ 래핑 · job status 폴링 |

**검증**

- `pytest -q` 300 passed · coverage 92.95% · ruff/black/mypy 0 에러 유지 (코드 변경 없음, PLAN 문서만 수정).
- `make plan-check`: `OK: in-progress=0`.
