## 자주 겪는 문제 & 복구

*"어디서 문제가 났는지" 가 애매할 때* 여기서부터 점검합니다. 대부분의
증상은 `make doctor` 1 회 실행으로 원인을 파악할 수 있습니다.

### 환경·기동

| 증상 | 점검 / 복구 |
| --- | --- |
| `make run` 이 `python: No such file or directory` | venv 가 아직 없음. `make install` 실행 후 재시도. `make doctor` 로 해석된 PYTHON 확인 |
| 포트 8501 충돌 | `streamlit run app.py --server.port 8502` |
| 업로드 중 "파일 크기 초과" | `.env` 의 `MAX_UPLOAD_MB` 수정 → `make run` (자동 동기화) |

### DB 초기화

| 증상 | 점검 / 복구 |
| --- | --- |
| 홈에 "DB 초기화가 완료되지 않았습니다" 배너 | `python scripts/init_db.py --seed` 실행. 스키마가 있더라도 `SYSTEM_USER_ID` 행 누락 시 발생 |
| `sqlite3.IntegrityError: FOREIGN KEY constraint failed` (audit_logs) | 위와 동일 — seed 누락 |
| 완전 재초기화가 필요 | `python scripts/init_db.py --drop --seed` (데이터 전부 삭제됨) |

### 학습·알고리즘

| 증상 | 점검 / 복구 |
| --- | --- |
| XGBoost / LightGBM 이 후보에 없음 | macOS: `brew install libomp` · Linux: `apt install libgomp1`. `make doctor` 의 *optional backends* 섹션 확인 |
| `XGBoostError: libxgboost.dylib could not be loaded` / `libomp.dylib` 관련 OSError | 위와 동일 |
| 일부 알고리즘만 실패 | 결과 비교 표의 *원인* 열에 축약 메시지. `storage/logs/app.log` 에서 `training.algo_failed` 이벤트 검색 |
| `ValueError: 타깃이 충분히 구별되지 않습니다` | 타깃 컬럼을 다른 컬럼으로 바꾸거나 데이터 크기 확대 |

### 데이터·예측

| 증상 | 점검 / 복구 |
| --- | --- |
| 업로드 `파일이 비어 있습니다` | 0 byte 파일 — 다른 파일 선택 |
| `중복된 컬럼명` / `헤더가 비어 있음` | 원본 CSV 헤더를 유일·값 있게 정리 |
| 예측 `누락된 입력 컬럼` | CSV 헤더가 모델 `schema.json` 과 다름. 학습 시점 컬럼명·대소문자 일치 필요 |
| 예측 결과 CSV 가 비어 있음 | 업로드 CSV 가 유효성 검사에서 전부 탈락. 원본 점검 |

### 테스트·CI

| 증상 | 점검 / 복구 |
| --- | --- |
| `샘플 파일 없음` 으로 테스트 skip | `make samples` 또는 `python scripts/generate_samples.py` |
| `make ci` 실패 | `make lint` → ruff/mypy 메시지 확인. `make cov` 로 coverage 임계(60%) 확인 |

### 진단 명령 요약

```bash
make doctor        # venv / python / streamlit / pytest / optional backends
make bench         # NFR-003 성능 벤치 (업로드·미리보기·예측 시간 측정)
python scripts/init_db.py --drop --seed   # DB 완전 재초기화
```
