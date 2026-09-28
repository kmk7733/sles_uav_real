# 2026-09-22 구현·배포 결과

승인된 명칭/권한 분리와 Vicon 충돌 정지 실행 계약을 구현했다. ROGX
`/home/rogx/catkin_ws/src`에 배치했으며 ROS 노드 실행·FCU 서비스 호출·비행은 하지 않았다.

## 변경 파일

| 경로 | 변경 내용 |
|---|---|
| `planner/desimplex.py` | 원래 `DeSimplexSupervisor`의 동일 클래스 alias `DeSimplexSwitcher` |
| `planner/trajectory_safety.py` | 원래 `PlanarSafetyValidator`의 동일 클래스 alias `TrajectorySafetyValidator` |
| `offboard_flight/scripts/planar_producer_factory.py` | 새 역할 이름으로 구성하고 역할·source checksum을 로그에 포함 |
| `offboard_flight/scripts/collision_stop_core.py` | Vicon-only swept risk, 입력 검사, terminal STOP 결정 |
| `offboard_flight/scripts/collision_stop_guard.py` | 별도 ROS gate, shadow/enforce 분리, nominal timestamp 보존, JSONL |
| `offboard_flight/scripts/mission_stop_contract.py` | status session/sequence 검증, failure latch, hold/정지 속도/AUTO.LAND 상태 계약 |
| `offboard_flight/scripts/guarded_mission_node.py` | 원래 mission을 상속하는 별도 실행 파일, frozen local hold와 착륙 실행 |
| `offboard_flight/scripts/guarded_mission.example.yaml` | 필수 운용값이 null인 비활성 검토용 설정 |
| `hardware/collision_stop/collision_stop_profile.example.json` | Vicon map/topic/frame과 필수 수치를 명시하는 비활성 shadow 설정 |
| `hardware/collision_stop/tests/test_*.py` | pure/mock 검사 106개, 실제 guard↔executor JSON 왕복 포함 |
| `offboard_flight/README_COLLISION_STOP.md`, `hardware/collision_stop/README.md` | 드론에서 찾을 운용 문서 |
| `hardware/hpa_integration/README.md`, `TODOLIST.md` | 확정 구조와 남은 전체 HPA/DeSimplex 연결 상태 |

기존 `fly.sh`, `preflight.py`, `record_flight.sh`, `mission_node.py`,
`guidance_library.py`, `planar_planner_node.py`, `vicon_traj`는 수정하지 않았다.
원래 switching/recovery/transition 구현도 그대로다. 지정 V4 모델과 CUDA 환경은 변경하지 않았다.

## 검증

| 검사 | 결과 |
|---|---|
| 로컬 새 가드/mission + 기존 alignment + factory 비교 | **132 passed** |
| ROGX Python 3.8에서 같은 검사 | **132 passed** |
| 기존 mission fake-ROS closed-loop harness | **27/27 checks passed**, 실제 FCU 없음 |
| 기존 switching/recovery/transition | **48 checks passed** |
| 기존 HPA producer 회귀 | **9 tests passed** |
| ROGX 실제 ROS 환경의 새 모듈 import | 통과, ROS node initialized = false |
| 클래스 alias identity | 동일 객체임을 로컬 비교/ROGX import에서 확인 |
| 변경 전 파일 보존 | ROGX 기존 일반 파일 **610개 동일**, 누락/예상 밖 변경 0 |
| 기존 실행 파일·시뮬레이터 핵심 파일 로컬 비교 | 보호 대상 11개 동일 |

기존 파일 중 변경한 코드는 factory 하나이며 원래 MPPI node는 이를 import하지 않는다.
다른 배포 파일은 새로 추가했다. checksum 목록은 [배포 manifest](evidence/install_manifest.json),
실행 결과는 [로컬 테스트](evidence/local_tests.txt), [ROGX 테스트](evidence/rogx_tests.txt),
[기존 파일 비교](evidence/deployment_check.txt)에 있다.

ROGX의 Python 2용 pytest 대신 검증 디렉터리
`/home/rogx/backups/pre-collision-stop-20260922/release/pytest_vendor`에만 pytest 7.4.4와
순수 Python 의존성을 설치했다. 시스템 Python/ROS/PyTorch/CUDA 패키지를 교체하지 않았다.
검증 fixture에 기존 simulator vendor 소스와 frame helper를 복사했다.

검증한 주요 고장 조건: stale/역행 timestamp, 결측 GT, 잘못된 frame/epoch, 이동 장애물,
guard heartbeat 만료, status replay, 저속 샘플 반복·중간 고속 샘플, estimator reset,
AUTO.LAND 응답 거부/지연, 실제 mode 확인, RC 전환, STOP latch 유지, 중복 writer와 topic remap.
이 결과는 ROS transport 전체 replay, 실제 제동, FCU 착륙 또는 비행 안전성 검증이 아니다.

## 백업·실행·복구

변경 전 백업 커밋은 로컬 `63ea8d5`, ROGX `9584d6e`이고 ref는 양쪽 모두
`backup/pre-collision-stop-20260922`이다. 원래 브랜치와 Git index를 유지했다.
이번 구현도 별도 `implementation/collision-stop-20260922` ref에 보관한다.
기존 전체 archive와 이번 source/Vicon archive 경로·checksum은 [백업 manifest](evidence/backups.json)에 있다.

Shadow 실행/중지 명령과 설정은 [README](README.md#설정과-shadow-실행중지)를 따른다.
이번에 새 프로세스를 실행하지 않았으므로 기존 데이터 수집을 위한 복구 명령은 필요 없다.
향후 복구 시 새 chain을 독점 운용 조건에 따라 정리한 뒤 원래 실행 파일을 선택한다.
factory까지 되돌릴 필요가 있다면 backup ref의 해당 파일을 빈 경로로 추출·비교한 뒤
그 파일만 복원한다. 전체 reset이나 원본 데이터 덮어쓰기는 하지 않는다.

## 남은 확인

* 실제 Vicon map/subject/frame, 기체 offset/radius와 개입 margin/horizon.
* 정지 속도·dwell·timeout 운용값, 실제 제동 여유, AUTO.LAND 및 수동 override 동작.
* PX4 포함 ROS replay와 실제 메시지 transport/서비스 장애·reset 검증.
* HPA/DeSimplex reference의 ROS 추종 연결, 모드별 별도 실행/기록 구성.
* 센서·mapping·MPPI/HAA probe·CUDA HPA가 함께 연산하는 처리 지연 측정.

전체 목록은 [통합 TODO](../hpa_integration/TODOLIST.md)에 유지한다.
