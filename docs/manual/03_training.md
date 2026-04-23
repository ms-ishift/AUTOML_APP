## 학습 설정 / 실행

선택한 데이터셋에 대해 **문제 유형 · 타깃 · 제외 컬럼 · 테스트 비율 · 기준
지표** 를 정하고, 등록된 여러 알고리즘을 한 번에 돌려 성능을 비교합니다.

### 👤 사용자 관점

**할 수 있는 일**

- 분류 / 회귀 태스크 선택 (`FR-040`)
- 타깃 · 제외 컬럼 지정 (식별자 컬럼은 자동 추천)
- 테스트 비율 · 평가 지표 · 학습명 설정
- **고급 전처리** 로 결측/스케일/이상치·범주 인코딩·datetime 분해·불균형 대응을 선택적으로 조정 (`FR-055~058`)
- **피처 변환 미리보기** 로 실제 학습 전에 원본/변환 후 열 수 · 파생 피처를 확인 (`FR-058`)
- 진행률 표시 (`st.status`) 와 완료 후 요약 카드 확인

**기본 사용 순서**

1. 상단 *"데이터 확인"* 접힘 섹션에서 프로파일을 다시 확인합니다.
2. *"학습 설정"* 에서 타깃/제외/테스트 비율/지표를 정합니다.
3. (선택) 하단 *"고급 전처리 (선택)"* 를 펼쳐 결측/스케일/이상치·범주 인코딩·datetime 분해·불균형 대응을 조정합니다. 기본값은 v0.1.0 과 동일해 아무 것도 건드리지 않아도 됩니다.
4. (선택) **미리보기** 버튼으로 원본/변환 후 열 수와 파생 피처 목록을 확인합니다.
5. **학습 실행** 을 누르고 진행률이 완료되면 *"결과 요약"* 과 "결과 비교로 이동" 버튼이 나타납니다.

**자주 쓰는 버튼/옵션**

| 위젯 | 설명 |
| --- | --- |
| 문제 유형 | 분류/회귀 (`FR-040`). 선택에 따라 지원 지표가 달라집니다. |
| 타깃 컬럼 | 예측 대상 (`FR-041`). |
| 제외 컬럼 | 학습에서 뺄 컬럼 (`FR-042`). 유니크 비율 ≥ 0.95 인 식별자는 자동 추천. |
| 테스트 비율 | 기본 0.2 (`FR-043`), 슬라이더 0.05~0.50. |
| 기준 지표 | 분류: `accuracy / f1 / roc_auc` · 회귀: `rmse / mae / r2` (`FR-044`). |
| 학습명 | 선택 입력. 비워두면 타임스탬프 자동 부여 (`FR-045`). |
| 학습 실행 | 모든 알고리즘을 순차 실행하고 실패 건은 부분 성공으로 처리 (`FR-066`). |

**고급 전처리 (선택) 옵션**

| 축 | 선택지 | 메모 |
| --- | --- | --- |
| 수치 결측 대치 | `median` / `mean` / `most_frequent` / `constant_zero` | 기본 `median` (`FR-055`). |
| 수치 스케일 | `standard` / `minmax` / `robust` / `none` | 기본 `standard`. 이상치 많을 땐 `robust` 권장. |
| 이상치 처리 | `none` / `iqr_clip(k)` / `winsorize(p)` | `iqr_clip` 선택 시 `k` (기본 1.5), `winsorize` 는 `p` (기본 0.01). |
| 범주 결측 대치 | `most_frequent` / `constant_missing` | |
| 범주 인코딩 | `onehot` / `ordinal` / `frequency` | `onehot` + "고카디널리티 자동 감지" 켜짐(기본) 이면 `nunique > 임계` 컬럼은 자동으로 `frequency` 로 강등됩니다. |
| 고카디널리티 임계 | 정수 (기본 50) | 자동 강등 판단 기준. |
| datetime 분해 | 체크박스 + `year/month/day/weekday/hour/is_weekend` multiselect | datetime 컬럼이 자동 감지됐을 때만 효과. 미리보기로 확인 가능 (`FR-057`). |
| bool 수치 통과 | 체크박스 (기본 켜짐) | 끄면 bool 컬럼이 범주 경로로 합류. |
| 불균형 대응 (분류 전용) | `none` / `class_weight` / `smote` | `smote` 는 `imbalanced-learn` 필요. 미설치 환경에서는 선택지에서 숨겨지고 사유가 캡션으로 표시됩니다. |
| SMOTE k_neighbors | 정수 (기본 5) | `smote` 선택 시에만 노출. |

> 고급 전처리의 **어떤 축이라도** 기본값과 달라지면 expander 헤더에 `⚙️ 커스텀 전처리 적용됨` 배지가 붙고, 학습 시 `AuditLog` 에 `training.preprocessing_customized` 이벤트가 기록됩니다. 모든 값이 기본이면 v0.1.0 모델과 **바이트 동치** (아티팩트 / 감사 로그 모두 추가 생성 없음).

**피처 변환 미리보기 (FR-058)**

expander 하단의 *"미리보기"* 버튼을 누르면 실제 학습을 수행하지 않고도 현재 설정이 만들어낼 피처 구조를 확인할 수 있습니다.

- `원본 열` / `변환 후 열` / `파생 피처` 메트릭 3개
- `source → derived_name (kind)` 테이블 — onehot 분해, datetime 파트, frequency 강등 등 변환 유형 표시
- `kind` 값: `onehot` / `ordinal` / `frequency` / `datetime_year` · `datetime_month` 등 / `bool_numeric` / `iqr_clipped` · `winsorized` / `passthrough`
- 고카디널리티로 `onehot → frequency` 로 자동 강등된 컬럼이 있으면 info 안내가 1회 표시됩니다.

**주의사항 & 팁**

- 데이터가 극단적으로 작거나 타깃 고유값이 부족하면 `ValueError: 타깃이 충분히 구별되지 않습니다` 가 뜹니다. 최소 수십 건 + 2개 이상의 클래스/값이 있어야 안정적입니다.
- macOS 에서 `libomp` 가 없으면 **XGBoost / LightGBM 이 후보에서 자동 제외** 됩니다. 페이지 상단에 사유 안내가 뜨면 `brew install libomp` 후 앱 재시작.
- 학습이 오래 걸린다면 테스트 비율을 올리거나 제외 컬럼을 늘려 특성 수를 줄이는 것이 빠릅니다.
- SMOTE 는 **분류 전용** 입니다. 회귀로 전환하면 옵션이 `none` 으로 자동 고정됩니다.
- 스케일을 `robust` 로 바꾸거나 이상치 처리를 `iqr_clip` 으로 두면 outlier 가 많은 데이터에서 학습이 안정됩니다 (선형/거리기반 알고리즘에 특히 효과적).

### 🛠 개발자 관점

| 항목 | 위치 |
| --- | --- |
| 페이지 | `pages/03_training.py` |
| 서비스 | `services/training_service.py` (`run_training`, `get_training_result`, `preview_preprocessing`) |
| ML | `ml/registry.py` · `ml/preprocess.py` · `ml/feature_engineering.py` · `ml/balancing.py` · `ml/trainers.py` · `ml/evaluators.py` · `ml/artifacts.py` · `ml/type_inference.py` |
| 스키마 | `ml/schemas.TrainingConfig` + `PreprocessingConfig` (frozen dataclass, UI/Service 경계의 유일한 계약) |
| 저장 | 학습 아티팩트는 `storage/models/<model_id>/` 에 `model.joblib` · `preprocessor.joblib` · `feature_schema.json` · `metrics.json` · (선택) `preprocessing_config.json` 최대 5 종 (`FR-073`, §9.6). 기본 전처리 설정이면 `preprocessing_config.json` 은 생성되지 않아 v0.1.0 과 바이트 동치를 유지합니다. |
| 감사 | 기본 전처리로 학습 시엔 이벤트 없음. `PreprocessingConfig.is_default == False` 이면 `AuditLog(action_type="training.preprocessing_customized")` 한 줄 기록 (§9.8). |
| FR | FR-040 ~ FR-066 + FR-055~058 (§9) |

**확장 포인트**

- 새 알고리즘 등록: `ml/registry.py` 에 `AlgoSpec` 추가 → factory 함수만 정의하면 UI/서비스는 변경 없음.
- 비동기 학습(§8.2): 현재는 동기 호출. Celery/RQ 로 분리 시 `run_training` 을 job enqueue 로 치환하고 Service 계약은 유지.
- 새 전처리 축 추가(§9.11 후속 범위): `PreprocessingConfig` 에 Literal 필드 1개 + `__post_init__` 유효성 1줄 + `from_dict` 에 누락 키 fallback 을 넣고, `ml/preprocess.py` 의 해당 분기만 확장. 기본값이 기존 동작과 동치여야 하위호환이 보존됩니다.
- 새 트랜스포머 추가: `ml/feature_engineering.py` 에 `BaseEstimator + TransformerMixin` 쌍 구현 → `ColumnTransformer` 에서 조립. `__init__` 하이퍼만 보존하면 sklearn `clone` 은 자동 호환.
- 새 불균형 전략 추가: `ml/balancing.py::apply_imbalance_strategy` 의 분기 + `ImbalanceStrategy` Literal 확장. 회귀 호환성은 `TrainingConfig.__post_init__` 가드로 방어.

### 자주 겪는 오류

| 메시지 | 원인 | 복구 |
| --- | --- | --- |
| `타깃이 충분히 구별되지 않습니다` | 분류인데 고유값이 1개거나 너무 적음 | 타깃을 다른 컬럼으로 바꾸거나 데이터 보강 |
| 일부 알고리즘 `실패 (Traceback)` | 개별 알고리즘 내부 오류(수렴 실패 등) | 다른 알고리즘은 정상 완료됨 — 실패 상세는 카드의 *"원인"* 에 축약 표시 |
| 후보 알고리즘에 XGBoost/LightGBM 없음 | macOS `libomp` 또는 패키지 미설치 | `make doctor` 로 사유 확인 → `brew install libomp` |
| 학습이 너무 느림 | 대용량 특성 / 파라미터 그리드 | 테스트 비율 상향, 제외 컬럼으로 차원 축소 |
| `SMOTE 는 분류(classification) 작업 전용입니다` | 회귀 + SMOTE 조합 | 불균형 대응을 `none` / `class_weight` 중 하나로 변경 |
| `SMOTE 를 사용하려면 imbalanced-learn 패키지가 필요합니다` | `imbalanced-learn` 미설치 상태에서 SMOTE 선택 시도 | `pip install imbalanced-learn` 후 앱 재시작 (또는 전략을 `class_weight` 로 대체) |
| `datetime_decompose=True 이면 datetime_parts 에 최소 1개 파트가 필요합니다` | datetime 분해 켠 상태로 파트 multiselect 가 빈 값 | 파트(`year` 등) 1개 이상 선택 |
| `outlier_iqr_k 는 0보다 커야 합니다` / `winsorize_p 는 (0, 0.5) 범위여야 합니다` | 이상치 파라미터 범위 위반 | number_input 을 적정값으로 재조정 |
