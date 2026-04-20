# AutoML Streamlit MVP

정형 데이터(CSV/XLSX)를 업로드해 분류/회귀 AutoML을 수행하고, 모델 비교 → 저장 → 단건/파일 예측까지 한 화면 흐름으로 처리하는 Python 중심 MVP.

**릴리즈**: `v0.1.0` (MVP 고정 · 2026-04-20) · 저장소: <https://github.com/ms-ishift/AUTOML_APP>

- 요구사항 원문: [`AutoML_Streamlit_MVP.md`](./AutoML_Streamlit_MVP.md)
- 아키텍처: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- 구현 계획(체크리스트): [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)

## 릴리즈 상태

- **범위**: `IMPLEMENTATION_PLAN.md` §0–§7 전부 완료. §8 (인증/PostgreSQL/비동기 학습) 은 후속 마일스톤으로 보류.
- **품질 게이트**: `ruff` · `mypy` · `pytest --cov` (fail_under=60, 실측 **92.95%**, 300 tests) 가 `make ci` 1 명령으로 모두 통과.
- **성능 NFR-003**: 5만 행 CSV 업로드·미리보기 ≤ 0.25s, 단건 예측 ≤ 0.02s (`make bench` 재현 가능).
- **회귀 QA NFR-004**: 빈 파일/깨진 CSV/중복 컬럼/학습 단일 실패/예측 누락 컬럼 모두 `tests/qa/` 에서 커버.

---

## 빠른 시작

> **전제**: Python 3.11+ · macOS / Linux / WSL2. 아래 명령은 **새 클론에서 그대로 재현**되도록 검증되어 있다.

```bash
# 1) 가상환경 + 의존성 (Makefile 이 .venv 를 자동 생성·사용)
make install
#   내부: python3.11 -m venv .venv → .venv/bin/pip install -r requirements.txt
#   활성화(source) 없이도 이후 모든 make 타깃이 .venv 바이너리로 실행된다.

# 2) 환경 변수
cp .env.example .env        # 필요 시 값 수정 (MAX_UPLOAD_MB 등)

# 3) DB 초기화
make samples                # 선택: sklearn 기반 샘플 CSV 2개 생성
./.venv/bin/python scripts/init_db.py --drop --seed
#  --drop : 기존 스키마 제거 후 재생성 (개발용)
#  --seed : 시스템 사용자 + 샘플 프로젝트 upsert

# 4) 앱 실행 (내부에서 .env 의 MAX_UPLOAD_MB 를 .streamlit/config.toml 에 동기화)
make run
```

기본 주소: <http://localhost:8501>

> **Tip** `make run` 이 실패하면 `make doctor` 로 어느 python/streamlit 이 잡혔는지 확인한다.
> 대부분 "`.venv` 미생성" 또는 "requirements 미설치" 가 원인이며 `make install` 로 해결된다.

`make` 가 없는 환경이라면 venv 를 수동 활성화한 뒤 아래로 대체 가능:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/sync_streamlit_config.py
streamlit run app.py
```

### 첫 화면에서 해보기

1. 사이드바 → **프로젝트 관리** 에서 새 프로젝트 생성
2. **데이터 업로드** 에서 `samples/classification.csv` 업로드
3. **학습** 에서 타깃 `species` · 지표 `accuracy` 로 실행
4. **결과 비교** 에서 베스트 모델 저장 (★)
5. **예측** 에서 단건/배치 예측 → CSV 다운로드

---

## 기술 스택

| 영역 | 선택 |
|------|------|
| UI | Streamlit (멀티페이지) |
| Data | pandas, openpyxl |
| ML | scikit-learn (LogReg/Tree/RF, Ridge/Lasso/RF) + (선택) XGBoost / LightGBM |
| 저장 | joblib, SQLAlchemy 2.x (SQLite → Postgres) |
| 시각화 | plotly (혼동행렬·회귀 산점도) |
| 설정 | pydantic-settings, python-dotenv |
| 품질 | ruff, black, mypy, pytest |

> XGBoost / LightGBM 은 `requirements.txt` 에 포함돼 있으나 **네이티브 런타임 의존**
> (macOS 는 `libomp`) 이 없으면 import 가 실패해 학습 후보에서 자동 제외된다. 현재 상태는
> `make doctor` 의 *optional backends* 섹션으로 확인한다. macOS 복구: `brew install libomp`.

---

## 레이어 구조 (요약)

```
UI (app.py, pages/*)
    │
    ▼
services/*  ──▶ repositories/*  ──▶ SQLite / Postgres
    │
    └──────▶ ml/*  ──▶ storage/models/<model_id>/
```

- UI는 **Service만** 호출한다.
- ML 모듈은 **순수 함수** (Streamlit/DB 의존 금지).
- Service 는 DTO (`services/dto.py`) 만 UI 에 넘긴다 (ORM 누출 금지).
- 세부: [`ARCHITECTURE.md`](./ARCHITECTURE.md) §2, §4 참조.

---

## 페이지 구성

| 경로 | 역할 | 주요 FR |
|------|------|---------|
| `app.py` | 홈 · DB 상태 · 전역 사이드바 | FR-001~003 |
| `pages/01_projects.py` | 프로젝트 생성/선택/수정/삭제 | FR-010~019 |
| `pages/02_dataset_upload.py` | 업로드 · 미리보기 · 프로파일 | FR-030~036 |
| `pages/03_training.py` | 타깃/지표/알고리즘 선택 · 학습 | FR-040~049, 060~062 |
| `pages/04_results.py` | 모델 비교표 · 성능 플롯 · 저장 | FR-063, 070~073 |
| `pages/05_models.py` | 저장 모델 목록 · 상세 · 삭제 | FR-074, 075 |
| `pages/06_prediction.py` | 단건/파일 예측 · 결과 다운로드 | FR-080~085 |
| `pages/07_admin.py` | 학습/예측 이력 · 통계 · 실패 감사 | FR-090~093 |

사이드바 네비는 `pages/components/layout.py::NAV_ITEMS` 가 단일 소스다.

---

## 폴더 구조 (발췌)

```text
automl_app/
├─ app.py                    # 홈 + 사이드바 + 가드
├─ pages/                    # Streamlit 멀티페이지
│  └─ components/            # layout / toast / data_preview
├─ services/                 # 비즈니스 로직 + DTO (dto.py)
│  ├─ project_service.py
│  ├─ dataset_service.py
│  ├─ training_service.py
│  ├─ model_service.py
│  ├─ prediction_service.py
│  └─ admin_service.py       # 교차 도메인 집계 (§6.7)
├─ ml/                       # 전처리/학습/평가/아티팩트 (순수 함수)
├─ repositories/             # SQLAlchemy CRUD + ORM 모델
├─ utils/                    # file / session / log / events / errors / messages
├─ config/                   # pydantic-settings
├─ storage/                  # datasets/ models/ predictions/ logs/  (gitignore)
├─ db/                       # SQLite 파일 (gitignore)
├─ samples/                  # sklearn iris/diabetes 기반 회귀용 샘플 (make samples)
├─ tests/                    # ml / services / ui / utils / qa / repositories
├─ scripts/                  # init_db / generate_samples / sync_streamlit_config
└─ .cursor/                  # rules/ + skills/ (에이전트 컨벤션)
```

---

## 테스트

전체 수트는 296 케이스(2026-04-20 기준)로, 실제 `scikit-learn` 학습을 도는 테스트 19건은 `@pytest.mark.slow` 로 분리되어 있다.

```bash
# 전체 실행 (학습 수트 포함, 약 40초)
pytest -q

# 빠른 수트만 (slow 제외, 약 25초)
pytest -q -m "not slow"

# 커버리지 (ml / services 중심)
pytest --cov=ml --cov=services
```

샘플 CSV 를 쓰는 fixture 는 `samples/*.csv` 가 없으면 skip 된다.
다음 중 하나로 생성해 두면 전체 수트가 돈다.

```bash
make samples                       # 또는
python scripts/generate_samples.py
```

---

## 개발 컨벤션

프로젝트 전반 규칙은 `.cursor/rules/` 에 정의되어 있다.

- `automl-project.mdc` — 레이어 경계, 금지 사항, 네이밍
- `python-style.mdc` — 타입힌트, dataclass, 예외/로깅
- `streamlit-ui.mdc` — 페이지 패턴, 세션 상태, 성능 캐시
- `service-layer.mdc` — Service/Repository 책임 분리
- `ml-engine.mdc` — 순수 함수, 파이프라인, 아티팩트

기능 구현 워크플로우는 `.cursor/skills/automl-feature-dev/SKILL.md` 를 따른다.

### 품질 도구

```bash
make lint           # ruff check + mypy (mypy 0 errors)
make fmt            # ruff --fix + black
make test           # pytest -q (296 tests, slow 포함)
make test-fast      # pytest -q -m "not slow" (~25s)
make cov            # pytest --cov=ml --cov=services (fail_under=60, 실측 93%)
make ci             # lint + cov 통합 게이트 (§7.5 CI gate)
make plan-check     # IMPLEMENTATION_PLAN 의 [~] 잔여 점검
```

`make ci` 는 배포 전 필수 게이트 — `ruff check` · `mypy` · `pytest --cov (≥ 60%)` 를 순차 실행한다.

개별 도구를 쓰려면:

```bash
ruff check .        # 린트
black .             # 포맷
mypy .              # 타입 체크
pytest              # 테스트
```

---

## 구현 단계

1. **1단계** — 프로젝트 생성, 업로드, 미리보기, 타깃 선택, 학습 실행
2. **2단계** — 모델 비교/저장, 단건·파일 예측, 결과 다운로드
3. **3단계** — 이력·관리자 화면, 로그인, Postgres 전환

세부 매핑과 수용 기준은 [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) 및 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §11 참조.

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| 업로드 후 "파일 크기 초과" 경고 | `.env` 의 `MAX_UPLOAD_MB` 수정 → `make run` (자동 동기화) |
| `DB가 아직 초기화되지 않았습니다` 배너 | `python scripts/init_db.py --drop --seed` 실행 |
| 테스트에서 `샘플 파일 없음` 으로 skip | `make samples` 또는 `python scripts/generate_samples.py` |
| 학습 시 `ValueError: 타깃이 충분히 구별되지 않습니다` | 데이터셋 유니크 값 / 행 수가 부족. `samples/classification.csv` 로 먼저 확인 |
| Streamlit 포트 충돌 | `streamlit run app.py --server.port 8502` |
| 학습 알고리즘 후보에 XGBoost/LightGBM 이 안 보임 | `make doctor` 로 사유 확인. macOS 에서 `libomp` 누락이면 `brew install libomp` 후 앱 재시작 |
| `XGBoostError: libxgboost.dylib could not be loaded` / `libomp.dylib` 관련 OSError | 위와 동일 — macOS OpenMP 런타임 누락. `brew install libomp` (Linux 는 `apt install libgomp1`) |

---

## 확장 (권장안 B 전환)

MVP 완료 후 필요에 따라:

- FastAPI 엔드포인트 + Celery 워커 (`training_service` / `prediction_service` 재사용)
- PostgreSQL (`DATABASE_URL` 교체만)
- S3 호환 오브젝트 스토리지 (`ml/artifacts.py` 만 교체)
- OAuth/SSO 인증 어댑터

Service 가 Streamlit 비의존 DTO 만 반환하는 제약 덕분에 UI 재작성 없이 전환 가능하다.
