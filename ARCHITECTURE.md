# AutoML Streamlit MVP — Architecture

**문서 버전:** 0.2 (구현 반영)
**대상 문서:** `AutoML_Streamlit_MVP.md` v0.2 (권장안 A)
**목적:** 바이브코딩 구현 전에 코드 레이어, 데이터 흐름, 모듈 책임, 확장 지점을 한눈에 고정한다.

> 0.1 → 0.2 차이: 실제 구현을 반영해 파일명(`06_prediction.py`), `components/` 구성(`layout.py`), 추가 서비스(`admin_service.py`), 세션 키(`current_model_id`), 아티팩트 `preprocessor.joblib`, 테스트 디렉터리(`tests/ui/`, `tests/qa/`, `tests/utils/`) 를 동기화.

---

## 1. 아키텍처 원칙

1. **단일 Python 레포, 레이어 분리**: UI(Streamlit) → Service → Repository → Storage/DB / ML Engine.
2. **Streamlit은 UI 계층에만 의존**: 학습·예측·DB 접근은 반드시 Service를 통해서만 호출한다. `st.*` 호출은 `app.py`/`pages/*`에만 존재.
3. **Service는 얇은 오케스트레이터**: 입력 검증 → Repository/ML 호출 → DTO 반환. 비즈니스 규칙은 Service에, 쿼리/영속성은 Repository에 둔다.
4. **ML 모듈은 순수 함수 지향**: `pandas.DataFrame` in, 모델/메트릭/아티팩트 out. Streamlit/DB 의존을 갖지 않는다.
5. **Fail loudly, log always**: 모든 Service 경계에서 `try/except`로 도메인 오류로 변환하고 `log_utils`로 기록한다.
6. **확장 지점 유지**: Service 인터페이스는 추후 FastAPI가 그대로 호출할 수 있도록 설계 (Streamlit 종속 타입 반환 금지).

---

## 2. 레이어 구조

```
┌──────────────────────────────────────────────────────────────┐
│  UI Layer  (Streamlit)                                       │
│  app.py, pages/01_projects.py ... pages/07_admin.py          │
│  - st.session_state 관리, 위젯 렌더링, 사용자 메시지 표시    │
│  - 입력 바인딩만 수행 → Service 호출                         │
└────────────────────────────┬─────────────────────────────────┘
                             │ (DTO / dataclass)
┌────────────────────────────▼─────────────────────────────────┐
│  Service Layer                                               │
│  services/{project,dataset,training,model,prediction}_service│
│  - 비즈니스 규칙, 입력 검증, 트랜잭션 경계                   │
│  - Repository + ML Engine 오케스트레이션                     │
└──────────┬─────────────────────────────────┬─────────────────┘
           │                                 │
┌──────────▼──────────────┐       ┌──────────▼────────────────┐
│  Repository Layer       │       │  ML Engine                │
│  repositories/*.py      │       │  ml/{preprocess,trainers, │
│  - SQLAlchemy/SQLite    │       │      evaluators,registry, │
│  - CRUD + 쿼리          │       │      schemas}.py          │
└──────────┬──────────────┘       └──────────┬────────────────┘
           │                                 │
┌──────────▼──────────────┐       ┌──────────▼────────────────┐
│  DB (SQLite → Postgres) │       │  Storage (local files)    │
│  db/app.db              │       │  storage/{datasets,models,│
│                         │       │          predictions}/    │
└─────────────────────────┘       └───────────────────────────┘
```

---

## 3. 폴더 구조 (확정)

```text
automl_app/
├─ app.py                    # 앱 진입점: 전역 세션/라우팅/사이드바
├─ pages/                    # Streamlit 멀티페이지 (UI 전용)
│  ├─ 01_projects.py
│  ├─ 02_dataset_upload.py
│  ├─ 03_training.py
│  ├─ 04_results.py
│  ├─ 05_models.py
│  ├─ 06_prediction.py
│  ├─ 07_admin.py
│  └─ components/            # 재사용 Streamlit 컴포넌트
│     ├─ layout.py           # 공통 사이드바/네비 (NAV_ITEMS 단일 소스)
│     ├─ toast.py            # 공통 알림 (FR-003)
│     └─ data_preview.py     # 데이터 미리보기/프로파일 위젯
├─ services/                 # 비즈니스 로직 (UI/DB 비의존 시그니처)
│  ├─ dto.py                 # UI 에 노출되는 모든 dataclass (ORM 누출 금지)
│  ├─ project_service.py
│  ├─ dataset_service.py
│  ├─ training_service.py
│  ├─ model_service.py
│  ├─ prediction_service.py
│  └─ admin_service.py       # 교차 도메인 집계 (이력/통계/실패)
├─ ml/                       # ML 엔진 (순수 함수 지향)
│  ├─ preprocess.py          # 결측치/인코딩/스케일링 파이프라인
│  ├─ trainers.py            # 알고리즘 등록/학습
│  ├─ evaluators.py          # 메트릭 계산/베스트 모델 선정
│  ├─ registry.py            # 알고리즘 카탈로그 (task_type→models)
│  ├─ artifacts.py           # joblib 저장/로드, 스키마 저장
│  ├─ profiling.py           # 데이터셋 프로파일(수치/범주/결측) 생성
│  └─ schemas.py             # FeatureSchema, TrainingConfig dataclass
├─ repositories/             # 영속성 (SQLAlchemy 2.x)
│  ├─ base.py                # engine / SessionLocal / session_scope()
│  ├─ models.py              # ORM 엔터티 (User, Project, Dataset, ...)
│  ├─ project_repository.py
│  ├─ dataset_repository.py
│  ├─ training_repository.py
│  ├─ model_repository.py
│  ├─ prediction_repository.py
│  └─ audit_repository.py
├─ utils/
│  ├─ file_utils.py          # 업로드 저장, 확장자 검증, read_tabular
│  ├─ session_utils.py       # st.session_state 헬퍼 (SessionKey Enum)
│  ├─ log_utils.py           # logging 초기화, 구조화 로깅
│  ├─ events.py              # 로깅/감사 이벤트명 상수 (Event)
│  ├─ messages.py            # 사용자 노출 한글 메시지 카탈로그 (Msg)
│  ├─ db_utils.py            # is_db_initialized / required_tables 점검
│  └─ errors.py              # AppError 계층 (Validation/NotFound/...)
├─ config/
│  ├─ settings.py            # pydantic-settings 기반 Settings
│  └─ logging.yaml           # 로깅 설정
├─ storage/                  # (gitignore) 로컬 파일 저장소
│  ├─ datasets/
│  ├─ models/
│  ├─ predictions/
│  └─ logs/
├─ db/
│  └─ app.db                 # (gitignore) SQLite 파일
├─ samples/                  # sklearn iris/diabetes 재현성 샘플 (scripts/generate_samples.py)
│  ├─ classification.csv
│  └─ regression.csv
├─ tests/
│  ├─ ml/                    # 전처리/학습/평가/아티팩트 순수 함수 테스트
│  ├─ services/              # DB + 파일 I/O 통합 테스트
│  ├─ ui/                    # streamlit.testing.v1.AppTest 헤드리스 UI
│  ├─ utils/                 # 유틸 단위 테스트
│  ├─ qa/                    # NFR-004 실패 경로 회귀 수트
│  └─ repositories/          # ORM/CRUD 통합 테스트
├─ scripts/
│  ├─ init_db.py             # 스키마 생성/시드 (--drop / --seed)
│  ├─ generate_samples.py    # samples/*.csv 재생성
│  ├─ sync_streamlit_config.py  # .env → .streamlit/config.toml 동기화
│  └─ dev_run.sh             # `streamlit run app.py`
├─ Makefile                  # install/samples/test/lint/fmt/run/plan-check
├─ requirements.txt
├─ pyproject.toml            # ruff/black/mypy/pytest 설정 (markers: slow, integration)
├─ .env.example
├─ .cursor/                  # rules/ + skills/ (에이전트 컨벤션)
└─ README.md
```

> MVP 문서(11.3) 폴더 구조를 기준으로 **repositories/models.py(ORM), services/dto.py, ml/artifacts.py · profiling.py, config/, tests/{ml,services,ui,utils,qa,repositories}/, utils/{errors,events,messages,db_utils}.py** 를 추가해 확장/테스트 가능성을 확보한다.

---

## 4. 핵심 데이터 흐름

### 4.1 학습 (FR-060 ~ FR-074)

```
[pages/03_training.py]
  │  타깃/제외컬럼/테스트비율/기준지표 입력
  ▼
training_service.run_training(training_config: TrainingConfig)
  │ 1) dataset_repository.get(dataset_id)  → file_path
  │ 2) ml.preprocess.build_pipeline(df, config)  → (X_train/X_test, preprocessor)
  │ 3) ml.trainers.train_all(task_type, X, y)   → List[TrainedModel]
  │ 4) ml.evaluators.score(models, X_test, y_test, metric_key)
  │ 5) training_repository.insert(job) / model_repository.bulk_insert(models)
  │ 6) ml.artifacts.save(best_model, preprocessor, schema) → storage/models/
  ▼
returns TrainingResultDTO (JSON-직렬화 가능)
```

### 4.2 예측 (FR-080 ~ FR-085)

```
[pages/06_prediction.py]
  │  단건 입력폼 or 파일 업로드
  ▼
prediction_service.predict_single(model_id, payload: dict)
prediction_service.predict_batch(model_id, file_path: Path)
  │ 1) model_repository.get(model_id)            → artifact_dir
  │ 2) ml.artifacts.load_model_bundle(dir)       → ModelBundle(estimator, preprocessor, schema, metrics)
  │ 3) validate_prediction_input(df, schema)     → 누락 차단(PredictionInputError) + 추가컬럼/unseen 경고
  │ 4) bundle.estimator.predict(df) (+ predict_proba 분류)
  │ 5) prediction_repository.insert(job)         → PredictionJob (form/file)
  │ 6) (batch) <predictions_dir>/<job_id>.csv 기록 → result_path
  ▼
returns PredictionResultDTO (rows: list[dict], result_path, warnings: list[str])
```

### 4.3 세션 상태 (FR-002)

`utils/session_utils.py::SessionKey` 에 정의된 키만 사용한다 (그 외 금지):

| Key | 타입 | 설명 |
|-----|------|------|
| `current_project_id` | `int \| None` | 사이드바/페이지 공통 컨텍스트 |
| `current_dataset_id` | `int \| None` | 업로드 직후 설정 |
| `last_training_job_id` | `int \| None` | 결과 화면 이동용 (§6.3 → §6.4) |
| `current_model_id` | `int \| None` | 모델 선택 상태 (§6.5 "예측하러 가기" → §6.6) |
| `flash` | `list[ToastMsg]` | 성공/실패/경고 큐 (FR-003) |

---

## 5. 도메인 모델 (ORM 매핑 개요)

요구사항 §9의 엔터티를 그대로 매핑한다. MVP는 `User` 미사용 모드도 허용 (FR-010).

- `Project(1) ─< Dataset(N)`
- `Dataset(1) ─< TrainingJob(N)`
- `TrainingJob(1) ─< Model(N)`  *(모델 비교 결과의 각 알고리즘이 1 row)*
- `Model(1) ─< PredictionJob(N)`
- `AuditLog` 은 모든 주요 액션에 대해 append-only

파일 경로는 DB에 상대경로로 저장, 루트는 `config.settings.STORAGE_DIR`.

---

## 6. ML 엔진 설계

### 6.1 알고리즘 레지스트리 (`ml/registry.py`)

```python
# 의사 구조
CLASSIFIERS = {
    "logistic_regression": LogisticRegressionSpec(...),
    "decision_tree":       DecisionTreeSpec(...),
    "random_forest":       RandomForestSpec(...),
    "xgboost":             XGBClassifierSpec(...),   # 선택
    "lightgbm":            LGBMClassifierSpec(...),  # 선택
}
REGRESSORS = { ... }  # linear, ridge, lasso, rf, xgb/lgbm
```

- 신규 알고리즘 추가는 Spec 1개 등록만으로 완료되어야 한다 (OCP).
- Spec은 `name, estimator_factory(), default_params, supports_task`를 노출한다.

### 6.2 전처리 (`ml/preprocess.py`)

- `sklearn.compose.ColumnTransformer` 기반 단일 `Pipeline` 생성.
- 정책 (FR-051~053):
  - 수치형: 결측 → median, 스케일링 → StandardScaler (옵션)
  - 범주형: 결측 → most_frequent, 인코딩 → OneHotEncoder(handle_unknown="ignore")
- 파이프라인 객체는 모델 아티팩트와 **반드시 함께 저장** (예측 시 동일 변환 보장, §10.4).

### 6.3 평가 (`ml/evaluators.py`)

- 지표 사전:
  - 분류: `accuracy, f1, roc_auc`
  - 회귀: `rmse, mae, r2`
- `select_best(models, metric_key)` 는 기준 지표와 방향(max/min)을 내장해 정렬.

### 6.4 아티팩트 (`ml/artifacts.py`)

저장 레이아웃:
```
storage/models/<model_id>/
  ├─ model.joblib           # fit 된 Pipeline (전처리+추정기)
  ├─ preprocessor.joblib    # Pipeline 에서 추출된 ColumnTransformer 복제본
  ├─ feature_schema.json    # 입력 컬럼/타입/카테고리값
  └─ metrics.json           # 지표 + 혼동행렬/산점도 등 시각화 데이터
```

- `feature_schema.json` 은 예측 시 입력 폼 자동 생성(FR-082)과 입력 검증(FR-083)의 단일 출처(Single Source of Truth).
- 저장·로드는 `save_model_bundle()` · `load_model_bundle()` 쌍을 통해서만 한다. 4개 파일 중 하나라도 누락되면 `FileNotFoundError` 로 실패한다.
- 부분 저장 실패 시 호출자(`training_service`) 가 디렉터리 단위 `shutil.rmtree` 로 롤백한다.

---

## 7. 오류/로그/관측

- **예외 계층** (`utils/errors.py`):
  - `AppError` ← `ValidationError`, `NotFoundError`, `StorageError`, `MLTrainingError`, `PredictionInputError`.
  - Service 는 ML/Repository 예외를 반드시 `AppError` 로 변환해 UI 에 노출한다.
- **메시지 카탈로그** (`utils/messages.py::Msg`):
  - 사용자 노출 한글 리터럴은 Service/UI 에서 직접 쓰지 않고 `Msg.*` 로 참조 (FR-003 일관성).
- **로깅** (`utils/log_utils.py` + `utils/events.py::Event`):
  - 구조화 키: `action`(= `Event` enum), `project_id` / `dataset_id` / `job_id` / `duration_ms` / `status`.
  - 핸들러는 `storage/logs/app.log` 롤링 파일.
- **감사 로그** (FR-092, NFR-007):
  - 프로젝트 생성, 업로드 성공/실패, 학습 start/end/fail, 모델 저장/삭제, 예측 수행/실패 시 `audit_repository.write()`.
  - `action_type` 접미사 규약: 성공은 `<domain>.<verb>`, 실패는 `<domain>.<verb>_failed`.

---

## 8. 보안/운영 (NFR-006, NFR-007)

- 업로드: 확장자 화이트리스트(`csv`, `xlsx`) + MIME 재검사 + 최대 크기 설정값.
- 인증 모드 스위치: `settings.AUTH_MODE = "none" | "basic"`.
- 비밀번호: `passlib[bcrypt]` 해시 저장.
- 파일 경로 주입 방지: 저장 파일명은 UUID로 재명명.

---

## 9. 구성/설정

`config/settings.py` (pydantic-settings):

| Key | 기본값 | 설명 |
|-----|--------|------|
| `APP_ENV` | `dev` | `dev`/`prod` |
| `DATABASE_URL` | `sqlite:///db/app.db` | Postgres 전환 시 덮어씀 |
| `STORAGE_DIR` | `./storage` | 파일 루트 |
| `MAX_UPLOAD_MB` | `200` | 업로드 상한 |
| `AUTH_MODE` | `none` | `none`/`basic` |
| `DEFAULT_TEST_SIZE` | `0.2` | 학습 분할 기본값 |
| `LOG_LEVEL` | `INFO` | 로깅 레벨 |

`.env` → `Settings()` 으로 주입, 모든 모듈은 `from config.settings import settings` 사용.

---

## 10. 확장 로드맵 (권장안 B 전환)

| 전환 항목 | 현재 (MVP) | 다음 단계 |
|-----------|------------|-----------|
| 실행 환경 | Streamlit 동기 | FastAPI 엔드포인트 + Celery 워커 (`training_service` 그대로 재사용) |
| 저장소 | SQLite | PostgreSQL (DATABASE_URL 교체만) |
| 파일 저장 | 로컬 `storage/` | S3 호환 (`ml/artifacts` 만 수정) |
| 인증 | basic | OAuth/SSO 어댑터 추가 |

Service 인터페이스가 Streamlit 비의존 DTO만 반환하도록 유지하면 위 전환들은 **UI 재작성 없이** 가능하다.

---

## 11. 구현 우선순위 (문서 §14 반영)

1. **1단계**: `config`, `repositories/base+models`, `project_service`, `dataset_service`, `ml/preprocess + trainers + registry + profiling`, `training_service`, `app.py`, `pages/01~03`.
2. **2단계**: `ml/evaluators + artifacts`, `model_service`, `prediction_service`, `pages/04~06`.
3. **3단계**: `audit_repository` 확장 + `admin_service`, `pages/07_admin`, 로그인, Postgres 스위치.

각 단계는 수용 기준(§13)에 대응하는 통합 테스트 1건 이상을 `tests/` 에 남긴다. NFR-004(실패 경로) 회귀는 `tests/qa/test_failure_paths.py` 에 묶여 있다.
