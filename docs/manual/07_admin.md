## 이력 / 관리자

학습·예측의 **실행 이력**, **시스템 로그 요약**, **운영 현황**(프로젝트/
데이터/모델/예측 건수 등)을 한 화면에서 확인합니다.

### 👤 사용자 관점

**할 수 있는 일**

- 학습 이력 조회 (기간/상태/대상 프로젝트 필터, `FR-090`)
- 예측 이력 조회 (단건/파일 구분, 건수·성공률, `FR-091`)
- 감사 로그 요약 — 액션 유형별 카운트, 최신 몇 건의 상세 (`FR-092`)
- 운영 현황 — 활성 프로젝트 수, 데이터셋 총량, 모델 수 등 대시보드 지표 (`FR-093`)

**기본 사용 순서**

1. 상단 필터에서 기간·상태를 좁힙니다.
2. 이력 표에서 개별 레코드를 펼쳐 세부(에러 메시지 포함) 를 확인합니다.
3. 필요한 경우 CSV 로 내보내 외부에서 분석합니다.

**자주 쓰는 위젯**

| 위젯 | 설명 |
| --- | --- |
| 기간 필터 | 최근 N 일 또는 커스텀 범위. |
| 상태 필터 | `success / partial / failed` (학습) · `success / failed` (예측). |
| 대상 프로젝트 | 전체 또는 현재 프로젝트만. |

**주의사항 & 팁**

- 이력은 **읽기 전용** 입니다 — 삭제는 프로젝트/모델/데이터셋 페이지에서 원본을 지워야 합니다.
- 감사 로그의 `user_id=0` 은 기본 시스템 사용자(`SYSTEM_USER_ID`) 입니다. §8.1 인증 도입 시 실제 사용자 ID 로 바뀝니다.

### 🛠 개발자 관점

| 항목 | 위치 |
| --- | --- |
| 페이지 | `pages/07_admin.py` |
| 서비스 | `services/admin_service.py` (`summarize_history`, `list_logs`, `counts`) |
| 저장 | `audit_logs` 테이블 (`repositories/audit_repository.py`) + `training_jobs` · `prediction_jobs` |
| 로그 파일 | `storage/logs/app.log` (RotatingFileHandler, `FR-092`) |
| FR | FR-090 ~ FR-093 |

**확장 포인트**

- 성능 차트: Plotly 라인 차트로 일별 학습 건수/성공률.
- 외부 APM 연동: `utils/log_utils` 포맷을 JSON 라인으로 전환하면 Loki/Datadog 로 수집 가능.
- 알람: 실패율 임계치 초과 시 Slack/이메일 트리거 — Service 이벤트 버스 추가.

### 자주 겪는 오류

| 메시지 | 원인 | 복구 |
| --- | --- | --- |
| 이력이 비어 있음 | 앱을 처음 실행했거나 필터 기간이 맞지 않음 | 필터 범위 확장 |
| 감사 로그 누락 | 일부 서비스 흐름이 `audit_repository.write` 를 호출하지 않음 | `services/*_service.py` 에 감사 write 호출 추가 (§7.1 원칙 준수) |
| `FOREIGN KEY constraint failed` (audit_logs) | DB 가 `--seed` 없이 생성돼 `SYSTEM_USER_ID` 가 없음 | `python scripts/init_db.py --seed` 또는 `make doctor` 점검 |
