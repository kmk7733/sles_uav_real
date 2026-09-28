# 오늘의 작업과 HPA / DeSimplex 완료 TODO

이 문서는 이전 HPA 공통 구현과 2026-09-22 승인된 명칭·Vicon 정지 가드 구현 상태를 정리한다.
실제 노드 실행·FCU 조작은 별도 승인 범위이며, 이번 코드 배포로 실행하지 않는다.

## 1. 오늘 완료한 것

- 지정한 V4 `v4_lr1e3_wd1_b64_s0_e40`, epoch 15, `best.pt`만 사용한다.
  Checksum을 고정하고 ROGX 기존 Jetson PyTorch/CUDA에서 예측 일치를 확인했다.
- `planner/hpa/`에 V4 runtime, 고정 anchor 가속도 회전, clipping/jerk 제한,
  CappedDynamics 적분, reference 생성과 기존 commit 동작을 구현했다.
  출력 가속도 10개에서 위치·속도·가속도·yaw·yaw-rate reference 11개 node를 만든다.
- `planar_producer_factory.py`에서 HAA/HPA/DeSimplex를 명시적으로 생성한다.
  DeSimplex는 기존 supervisor와 transition을 그대로 사용한다.
- `hpa_depth_adapter.py`와 subscriber-only shadow를 학습 계약에 맞췄다.
  실제 depth/K, causal `velocity_local.linear`, causal `odom.twist.twist.angular.z`,
  명시적 pose 보간, float16→float32 scan, stale/reset/frame 차단과 JSONL 로그를 구현했다.
- 로컬 테스트 76개, ROGX 테스트 43개, 기존 switching/transition 48 checks를 통과했다.
  원래 simulator와 HPA 후처리 및 같은 설정의 HAA/DeSimplex 수치 동등성을 확인했다.
- 새 모듈을 ROGX `catkin_ws/src`에 배치했고 기존 파일 603개를 checksum으로 보존 확인했다.
  기존 fly/preflight/record/planner node/mission/guidance 실행 코드는 변경하지 않았다.
- 변경 전 전체 archive와 Git 백업을 만들었다. 로컬 `32e7aec`, ROGX `14c1371`.
  미커밋 변경, ignored 중요 파일, 모델, 데이터, nested Git 이력을 보존했다.
- 사용자 안내의 `start_test_grid.sh vicon`을 RESET=0으로 실행했다.
  Depth 약 15Hz, Vicon 약 50Hz를 관측했다. FCU는 사용자가 의도적으로 꺼둔 상태다.
  PX4 입력이 없어서 EKF 기반 mapper와 MPPI 연산은 대기 조건이다.
- 실제 비행 명령, arm, OFFBOARD 전환, 비행은 실행하지 않았다.

**완료 경계:** 공통 알고리즘·입력 어댑터·CUDA 검증용 shadow까지 준비됐다.
실제 ROS 비행 노드의 producer 선택/추종 연결, 가드의 실제 메시지/FCU 연결 검증,
실제 동시 부하·제동 검증까지 완료된 상태는 아니다.

## 2. 2026-09-22 확정한 역할과 이번 구현

최신 계약과 변경 파일·검증은 [CollisionStop README](../collision_stop/README.md)를 따른다.
아래 P1/P2/P4/P5는 여전히 전체 HPA/DeSimplex 연결의 남은 작업이다.

| 이름 | 역할 | 알고리즘/권한 |
|---|---|---|
| `TrajectorySafetyValidator` | HAA/DeSimplex 내부 동기 경로 검사 | 기존 `PlanarSafetyValidator`의 alias; HPA 단독 외부 gate 아님 |
| `DeSimplexSwitcher` | HPA/HAA/recovery/bridge 선택 | 기존 `DeSimplexSupervisor`의 alias; 시뮬레이터 알고리즘 유지 |
| `CollisionStopGuard` | 별도 Vicon GT 충돌 위험 gate | 통과/terminal STOP; HAA 전환·재계획·재개·FCU 명령 없음 |
| `MissionHoldExecutor` | guarded mission의 정지 실행 | frozen PX4 local 위치·고도·yaw + zero velocity → 정지 속도 지속 확인 → AUTO.LAND |
| `MissionFailsafe` | 가드/상태/좌표계 오류 처리 | 충돌 개입과 사유를 구분하고 nominal을 폐기 |

개입 시행은 failure이며 계속 비행하지 않는다. PX4 velocity는 정지 **실행 확인**용이며,
충돌 **판단**에는 Vicon 기체/장애물 위치와 Vicon-derived velocity만 쓴다.
기존 데이터 수집도 현재는 `mission_node.py` 체인이다. 과거 `setpoint_buffer.py`는 복원하지 않는다.

새 opt-in 체인:
`selected producer → commander/set_pose → CollisionStopGuard → commander/set_pose_safe
→ guarded_mission_node.py → MAVROS/PX4`.
기존 mission/fly/preflight/record/guidance/planner node와 vicon_traj 실행 파일은 보존한다.

## 3. 우선순위별 TODO

### P0 — 명칭과 제어 계약

- [x] DeSimplex/trajectory validator에 역할 alias와 factory 로그 이름을 추가한다.
- [x] 독립 가드와 실행 계층의 요청/ACK/heartbeat를 session·sequence·timestamp로 구분한다.
- [x] Vicon-only 위험 판단과 terminal failure latch를 구현한다.
- [x] 별도 guarded mission에 frozen local hold, 속도 지속 확인 후 AUTO.LAND,
  명령 만료·reset·수동 전환 처리를 구현한다. 원래 mission을 수정하지 않는다.
- [ ] 실제 운용할 기체 radius/offset·Vicon map/topics/frame·projection margin/horizon,
  센서/명령/heartbeat timeout과 정지 속도/dwell을 측정하고 profile에 명시한다.
  비활성 template이나 테스트 수치를 비행 운용값으로 사용하지 않는다.
- [ ] ROS replay에서 두 프로세스가 실제 메시지를 주고받는 전체 chain을 검증한다.
  단위/mock 테스트 통과는 FCU 정지/착륙 검증이 아니다.

### P1 — FCU 없이 입력·알고리즘 전체 경로 검증

- [ ] 원래 depth/K/PX4 odom/velocity_local이 함께 기록된 bag을 확보·확인한다.
  서로 다른 시간의 live depth와 기록 PX4를 섞어 실제 입력 검증이라고 하지 않는다.
- [ ] 별도 ROS master/namespace에서 replay한다. FCU/MAVROS 통신 bridge와
  actuator/setpoint 출력은 연결하지 않는다. 입력이 PX4 출처인지 provenance를 기록한다.
- [ ] 학습의 0.02초 pose 중간 격자 및 각도 보간과 live 정책의 차이를 비교한다.
  격자 위상·전처리 구현을 확인하고, 직접 보간/causal 정책의 상태·scan·action 차이와
  실제 대기 지연을 기록한다. 속도/angular-z의 causal 원본 선택은 유지한다.
- [ ] 명시적 PX4 ENU goal, planning-frame 변환, depth anchor와 시작 상태를 묶는다.
  Vicon pose 또는 commander 이동 setpoint를 HPA 입력/goal로 대체하지 않는다.
- [ ] 실제 실행할 commit, lookahead, handover 설정과 HPA/HAA 물리·tightened limits,
  지도 margin을 profile로 확정한다. 현재 example JSON은 설정 예시이지 최종 운용 승인값이 아니다.
- [ ] 동일한 입력열·설정에서 시뮬레이터와 새 경로의 U/X/reference 및
  DeSimplex mode/source/reason/fault/bridge를 비교한다. V4만 사용한다.

### P2 — ROS reference 추종 경로 연결

- [ ] 기존 `planar_planner_node.py`의 producer 생성을 factory와 연결한다.
  정상 HAA 경로는 같은 입력·설정에서 그대로 재현되는지 확인한다.
- [ ] HPA reference의 t0를 depth anchor로 유지하고, 계획 주기와 이미지 주기를 구분한다.
  지연된 결과 폐기, reference 샘플링/만료, 실제 추종 reference 기반 a_prev를 연결한다.
  지금 shadow의 a_prev=0 독립 재구성을 그대로 제어에 사용하지 않는다.
- [ ] HPA 상태의 원본 body angular-z와 계획 상태의 Euler yaw-rate 변환을 구분한다.
  실제 tilt에서 planar yaw-acceleration 해석 오차도 확인하고 보정은 임의 추가하지 않는다.
- [ ] 목표/지도/state/epoch snapshot을 일관되게 고정하고, reset 시 reference·commit·bridge를
  함께 폐기한다. 정상 DeSimplex switching마다 HPA commit을 초기화하지 않는다.
- [ ] DeSimplex의 전체 `.plan()` 경로를 사용한다. `fault=true` best-effort braking을
  안전 검증 성공으로 오인하지 않도록 실제 제어 계층의 처리와 로그를 연결한다.

### P3 — CollisionStopGuard 후속 검증

- [x] 새 guard/core와 별도 guarded mission/executor를 추가한다. Shadow는 명령 publisher가 없다.
- [x] stale/시간 역행/결측·잘못된 frame, reset, latching, nominal stamp 보존,
  미정지·오래된 저속 데이터의 착륙 차단, 수동 전환을 단위/mock 테스트한다.
- [ ] 실제 PX4/Vicon bag replay로 누락·지연·reset·서비스 지연·프로세스 종료를 검증한다.
- [ ] 실제 기체 외곽, 관측/실행 지연과 물리 제동 성능을 측정해 projection margin을 검증한다.
  GT constant-velocity swept projection은 검증된 braking model/충돌 방지 보장이 아니다.
- [ ] PX4 AUTO.LAND 동작·정지 속도 임계값과 dwell·manual override를 승인된 시험에서 확인한다.
- [ ] guard와 mission 로그/rosbag에서 개입 요청, ACK, frozen local hold,
  정지 확인, AUTO.LAND 요청/관측, failure 원인을 한 session으로 재구성한다.

### P4 — 세 실행 경로 및 관측 로그 정리

- [ ] 기존 `fly.sh`를 유지한 채 `fly_haa.sh`, `fly_hpa.sh`, `fly_desimplex.sh`를 별도로 준비한다.
  실제 생성 producer, V4 hash, profile, guard 사용 여부를 실행 로그에 표시한다.
  초기 기본값은 FCU 명령 없는 shadow이고, live/arm 동작은 별도 명시적 단계로 둔다.
- [ ] 모드별 필요한 입력을 구분한 별도 preflight/record 구성을 만든다.
  기존 `preflight.py`, `record_flight.sh`의 기본 HAA 동작을 바꾸지 않는다.
- [ ] `desimplex_decision`, `collision_stop_intervention`, `mission_failsafe`,
  `manual_override`, `estimator_reset`을 각각 기록한다. 원인·행위자·시각·입력 age,
  요청한 reference와 실제 전달 reference를 보존한다.
- [ ] 변경 파일별 백업·commit과 rollback 명령을 준비하고 기존 HAA 회귀 검사를 통과한다.

### P5 — 처리 지연과 실제 제어 검증

- [ ] FCU OFF에서는 replay의 실제 연산으로 mapper·MPPI/HAA probe와 CUDA를 동시 실행해
  전체 pipeline을 측정한다. 이 결과는 replay 부하 측정으로 표시한다.
- [ ] projection, inference/transfer, HAA/probe, transition, guard, reference 전달 단계별
  p50/p95/p99/max, 실제 입력→결과 age, deadline 초과·폐기율을 수집한다.
  신경망 시간만을 10Hz 동작의 증거로 사용하지 않는다.
- [ ] 후속 명시적 승인으로 FCU 상태를 사용할 수 있는 지상 시험에서 실제 topic/frame,
  물리 카메라 장착, goal/epoch, 정지 요청과 수동 override 경로를 확인한다.
  FCU를 지금 켜라는 요청이나 자동 재연결은 하지 않는다.
- [ ] 실제 센서+mapping+MPPI/HAA probe+HPA+guard 동시 연산 지연을 별도 측정한다.
  모델/알고리즘을 바꿔 수치를 맞추지 않는다. 실패하면 원인과 허용 주기를 검토한다.
- [ ] 실제 제어·arm·OFFBOARD·이륙·비행은 각각 승인된 범위에서만 진행한다.
  코드/추론/replay 통과와 실제 제동/비행 검증 완료를 구분한다.

## 완료 판정

- HPA 완료: 정확한 V4 입력→가속도 적분 reference→기존 추종 경로가 연결되고,
  HPA 단독 정지 가드·timeout/reset/override가 검증되며 별도 실행 경로가 준비된 상태.
- DeSimplex 완료: 같은 입력·설정에서 원래 switching/recovery/bridge가 재현되고,
  실제 ROS 전달·lifecycle·최악 지연·fault 처리가 검증되며 별도 실행 경로가 준비된 상태.
- 실제 비행 준비 완료: 위 소프트웨어 완료에 더해 승인된 지상/제동 검증과 실제
  동시 부하 측정까지 통과한 상태. 현재는 이 단계가 아니다.

상세 기존 증거: [구현 보고서](REPORT.md), [shadow 실행/중지](../hpa_shadow/README.md),
[FCU OFF 센서 관측](evidence/live_grid_check_01/RESULT.md).
