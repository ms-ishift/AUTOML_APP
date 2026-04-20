---
name: automl-feature-dev
description: AutoML Streamlit MVP 프로젝트에 새 기능(FR)을 구현할 때의 워크플로우와 레이어별 체크리스트. 새 페이지, 서비스 함수, 알고리즘, 전처리 단계, 예측 기능 추가 등 요구사항 명세(AutoML_Streamlit_MVP.md)의 FR 번호를 구현할 때 사용.
---

# AutoML 기능 개발 워크플로우

이 프로젝트는 `AutoML_Streamlit_MVP.md`(요구사항), `ARCHITECTURE.md`(설계), `IMPLEMENTATION_PLAN.md`(진행 체크리스트), `.cursor/rules/*`(스타일)을 기준으로 구현한다. 새 기능을 추가할 때 아래 절차를 따른다.

## 0단계. 계획서 확인 및 상태 변경 (필수 시작점)

**모든 작업은 `IMPLEMENTATION_PLAN.md` 에서 시작한다.**

```
Task Progress:
- [ ] IMPLEMENTATION_PLAN.md 에서 해당 작업 항목 찾기
- [ ] 계획서에 없으면 → 먼저 항목 추가 (해당 단계 하위절) + "변경 이력" 기록
- [ ] 항목 상태를 [ ] → [~] 로 변경 (StrReplace)
- [ ] 진행 로그에 "YYYY-MM-DD | 단계 X.Y | started | 메모" append
- [ ] 의존 단계가 완료(`[x]`)인지 확인 (부록 B 의존성 그래프)
```

의존 단계가 미완이면 해당 작업부터 처리하거나, 병행 가능 여부를 계획서에 메모한다.

## 1단계. 요구사항 위치 확인

작업할 기능의 FR 번호를 `AutoML_Streamlit_MVP.md` §6에서 찾아 수용 기준을 읽는다. 그런 뒤 관련 화면을 §7에서, 처리 규칙을 §10에서 확인한다.

필수 입력:
- 어떤 FR을 충족시키는가? (예: `FR-060 학습 실행`)
- 어떤 화면에 연결되는가? (예: `화면 4. 학습 설정`)
- 어떤 처리 규칙이 적용되는가? (예: §10.2 학습 규칙)

## 2단계. 레이어 분해

`ARCHITECTURE.md` §2, §4의 데이터 흐름을 보고, 작업이 어느 레이어에 속하는지 결정한다.

```
Task Progress:
- [ ] UI 변경 필요? → pages/*.py
- [ ] 비즈니스 규칙/오케스트레이션? → services/*.py
- [ ] DB 스키마/쿼리? → repositories/*.py
- [ ] 전처리/학습/평가/저장? → ml/*.py
- [ ] 설정/환경 변수? → config/settings.py
```

한 작업이 여러 레이어에 걸치면 **아래에서 위로 구현**한다:
`repositories → ml → services → pages`

## 3단계. 레이어별 체크리스트

### Repository 추가/변경

- [ ] `repositories/models.py` ORM 엔터티 업데이트 (필드는 §9 데이터 요구사항과 일치)
- [ ] 함수 시그니처에 `session: Session` 를 첫 인자로
- [ ] 쿼리/CRUD만 작성, 비즈니스 규칙/로깅 금지
- [ ] 마이그레이션/스키마 재생성: `python scripts/init_db.py`

### ML 모듈 추가/변경

- [ ] `streamlit`, `sqlalchemy` import 하지 않음 (순수 함수)
- [ ] 신규 알고리즘이면 `ml/registry.py` 에 `AlgoSpec` 한 줄 추가
- [ ] 개별 모델 학습 실패가 전체 중단을 일으키지 않음 (`try/except`로 `status="failed"` 기록)
- [ ] 예측 재현성: `random_state=settings.RANDOM_SEED`
- [ ] 학습 결과는 `storage/models/<model_id>/{model,preprocessor,feature_schema,metrics}` 로 저장
- [ ] `tests/ml/test_*.py` 단위 테스트 추가

### Service 추가/변경

- [ ] 함수 docstring에 FR 번호 기재 (예: `"""FR-060: 학습 실행."""`)
- [ ] 입력 검증 → `ValidationError` (도메인 예외)
- [ ] `with session_scope() as session:` 블록 내에서 Repository 호출
- [ ] 반환은 DTO(dataclass). ORM/DataFrame 원본/`st.*` 타입 금지
- [ ] 진입/성공/실패 3지점에 `logger.info/exception` 구조화 로그
- [ ] 주요 액션이면 `audit_repository.write(...)` 호출
- [ ] `tests/services/test_*.py` 에 해피패스 + 실패 케이스 각 1개

### UI (pages/) 추가/변경

- [ ] 페이지 상단에서 필수 세션 상태 확인 → 없으면 `st.info(...)` + `st.stop()`
- [ ] 위젯 값 → Service 호출 → 결과 렌더링 순서 유지
- [ ] Service 호출은 `try/except AppError` 로 감싸고 `st.error` + `flash` 로 사용자 메시지 표시
- [ ] 데이터 미리보기는 `@st.cache_data`, 모델 로드는 `@st.cache_resource`
- [ ] 5만 행 이상은 `df.head(n)`로 표시 (NFR-003)
- [ ] `sqlalchemy`, `joblib`, `sklearn` import 없는지 확인

## 4단계. 수용 기준 검증

FR 수용 기준(§13)과 매핑되는 수동/자동 검증을 수행한다.

```
검증 체크:
- [ ] FR 수용 기준 문구가 실제 동작으로 확인되는가?
- [ ] 실패 경로(잘못된 입력/파일)에서 사용자 친화 메시지 + 앱 중단 없음
- [ ] 로그에 이벤트가 기록되는가? (storage/logs/app.log)
- [ ] 관련 테스트가 통과하는가? (`pytest tests/...`)
```

## 5단계. 계획서 갱신 및 마무리 (필수 종료점)

작업 코드가 동작한다고 끝나는 것이 아니다. **계획서를 갱신해야 작업이 완료된다.**

### Definition of Done 체크

```
- [ ] 단위 테스트 통과 (pytest <path>)
- [ ] ruff check <path> / mypy <path> 0 에러
- [ ] 레이어 경계 준수 (UI→Service만, ML 순수성 유지)
- [ ] FR 번호가 docstring/모듈 헤더에 기재
- [ ] 설계가 달라진 경우 ARCHITECTURE.md 동기화
```

### 계획서 업데이트

1. **체크박스 변경**: `[~]` → `[x]` (DoD 전부 충족 후에만)
2. **진행 로그 append** (문서 최하단):
   ```
   YYYY-MM-DD | 단계 X.Y | completed | 한 줄 메모
   ```
3. **변경/추가 발생 시**: "변경 이력" 표에 한 줄 추가
4. **이슈 발생 시**: `[!]` 로 표시 + 바로 아래 줄에 `> BLOCKED: <이유> / <후속조치>` + 리스크 레지스터에 행 추가

### 막힘 패턴 (`[!]` 처리)

```markdown
- [!] 3.5 학습 실패 격리 로직
  > BLOCKED: xgboost import 실패 (libomp 누락) / R-001 참조
```

그리고 "리스크 / 이슈 레지스터" 표에:

```
| R-001 | 2026-04-18 | 3.5 | xgboost import 실패 (M1) | trainers 테스트 skip | brew install libomp 후 재시도 | open |
```

블록 해제 시 상태를 `resolved` 로 바꾸고 진행 로그에 `resumed` 이벤트 기록.

## 레이어 매핑 치트시트 (FR → 파일)

| FR 영역 | 주요 파일 |
|---------|-----------|
| FR-020~024 프로젝트 관리 | `services/project_service.py`, `pages/01_projects.py` |
| FR-030~035 데이터셋 | `services/dataset_service.py`, `utils/file_utils.py`, `pages/02_dataset_upload.py` |
| FR-040~045 학습 설정 | `services/training_service.py` (config 빌드), `pages/03_training.py` |
| FR-050~054 전처리 | `ml/preprocess.py` |
| FR-060~066 학습 실행 | `ml/trainers.py`, `ml/evaluators.py`, `services/training_service.py` |
| FR-070~075 결과/모델 | `ml/artifacts.py`, `services/model_service.py`, `pages/04_results.py`, `pages/05_models.py` |
| FR-080~085 예측 | `services/prediction_service.py`, `pages/06_predict.py` |
| FR-090~093 이력/관리자 | `repositories/audit_repository.py`, `pages/07_admin.py` |

## 추가 참고

- 상세한 데이터 흐름: `ARCHITECTURE.md` §4
- 알고리즘 등록 방법: `ARCHITECTURE.md` §6.1 + `.cursor/rules/ml-engine.mdc`
- Streamlit 패턴: `.cursor/rules/streamlit-ui.mdc`
- Service/Repository 템플릿: `.cursor/rules/service-layer.mdc`

## 안티패턴 (즉시 중단)

**코드 레이어**
- `pages/*.py` 에서 DB 세션 열기, 모델 `joblib.load`, `sklearn` 직접 사용 → Service로 이동
- Service 반환값이 `pd.DataFrame` 원본 → DTO 또는 `list[dict]` 로 변환
- ML 함수가 `session_state` 나 ORM 객체에 의존 → 인자로 원시 타입/DataFrame만 받도록 수정
- 신규 알고리즘을 학습 루프에 `if/elif` 로 추가 → `AlgoSpec` 등록으로 변경

**계획서 관리**
- 코드를 먼저 작성하고 계획서를 나중에 맞추기 → 순서 반대. 스코프 변경은 **계획서 먼저**
- `[~]` 인 상태로 방치 → 블록되면 `[!]` 로 전환하고 이유 기록
- 검증 없이 `[x]` 체크 → DoD 미충족. 무조건 수용 기준 실행 후에만 체크
- 진행 로그를 건너뛰고 넘어감 → 돌이켜봤을 때 어느 시점에 무엇이 됐는지 추적 불가
- 계획서에 없는 파일을 슬쩍 만들어 넣기 → 레이어 이탈 위험. 먼저 계획서에 항목 추가

**일관성 게이트 (v0.3 추가)**
- 한글 메시지/에러 문구를 페이지에 직접 쓰기 → `utils/messages.py` 의 `Msg` 로 이동
- 로그 이벤트명을 문자열로 하드코딩 → `utils/events.py` 의 `Event` 상수 사용
- Service 반환에 ORM 객체/DataFrame 원본 노출 → `services/dto.py` DTO 로 변환
- 페이지 이동을 URL 하드코딩 → `st.switch_page("pages/xx.py")` 만 사용
- `.env` 의 `MAX_UPLOAD_MB` 변경 후 `.streamlit/config.toml` 미동기화 → `make run` 으로 동기화 스크립트 실행
- 학습 진행률을 별도 스레드에서 처리 → 금지. `on_progress` 콜백은 동기 루프 안에서만 호출
