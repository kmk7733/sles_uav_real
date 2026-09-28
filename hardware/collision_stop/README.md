# Vicon 충돌 정지와 DeSimplex 역할 분리

2026-09-22 합의: 기존 데이터 수집 경로를 그대로 두고 새 실행 파일로만 선택한다.
이번 단계는 명칭 정리와 **Vicon 충돌 판단 → frozen PX4 local hold → 정지 속도 확인
→ PX4 AUTO.LAND** 실행 계약이다. 노드 실행·arm·OFFBOARD·이륙·비행 승인은 포함하지 않는다.

배포·검증 결과와 변경 파일은 [구현 보고서](REPORT.md)에 정리했다.
로컬 및 ROGX의 pure/mock 회귀 검사 각각 132개가 통과했다. 실제 FCU 정지·착륙 검증은 아니다.

## 역할과 권한

| 이름 | 입력/실행 위치 | 결정과 권한 |
|---|---|---|
| `TrajectorySafetyValidator` | 기존 planner occupancy, HAA/DeSimplex 내부 동기 호출 | 후보 경로 검사. 기존 `PlanarSafetyValidator`와 같은 클래스 |
| `DeSimplexSwitcher` | 기존 simulator producer `.plan()` | HPA/HAA, recovery, hand-back, transition 선택. 기존 `DeSimplexSupervisor`와 같은 클래스 |
| `CollisionStopGuard` | 별도 ROS 프로세스, Vicon GT 기체/장애물 | 정상 명령 통과 또는 terminal STOP 요청. HAA 전환·재계획·재개·FCU 명령 권한 없음 |
| `MissionHoldExecutor` | 별도 `guarded_mission_node.py` 안 | fresh PX4 local 위치·고도·yaw 고정, zero velocity hold, 정지 확인 뒤 AUTO.LAND 요청 |
| `MissionFailsafe` | 같은 guarded mission 안 | 가드 heartbeat/입력/좌표계 오류 등 실행 장애를 충돌 개입과 별도 사유로 기록 |

`TrajectorySafetyValidator`와 DeSimplex recovery는 기존 시뮬레이터에 있던 알고리즘이다.
이름 alias를 추가할 뿐 수치·순서·제한값·회복/transition 코드를 바꾸지 않는다.
HPA 단독에 DeSimplex나 추가 경로 validator를 넣지 않는다. HPA 단독에서 외부 가드가
개입하면 해당 시행은 failure이며 HAA로 바꾸어 계속 날지 않는다.

```text
선택된 HAA / HPA / DeSimplex producer
  → 기존 reference 추종/ROS adapter
  → commander/set_pose
  → CollisionStopGuard
  → commander/set_pose_safe
  → guarded_mission_node (raw setpoint와 mode 요청의 유일한 실행자)
  → MAVROS/PX4

Vicon 기체·장애물 ─→ CollisionStopGuard ─STOP/status─→ guarded_mission_node
PX4 local pose/velocity·frame epoch ────────────────→ guarded_mission_node
```

가드는 위험 시 정상 setpoint 전달을 중단하고 STOP heartbeat를 유지한다. mission은
그 요청을 latch하고 fresh PX4 local pose에서 한 번 캡처한 위치·고도·yaw를 유지한다.
정지 기준은 **3차원 PX4 local 속도가 설정 기준 이하인 상태를 설정 시간 동안 관측**하는
것이다. AUTO.LAND 서비스 응답만으로 성공이라 판단하지 않고 FCU mode를 확인한다.
수동 mode 전환은 우선하며 제어권을 자동으로 되찾지 않는다. reset 시 이전 local hold를
폐기하고 새 epoch의 fresh local pose를 기다린다. 상태를 신뢰할 수 없으면 예전 명령을
재생하거나 정지/착륙 성공으로 표시하지 않는다.

## GT 판단과 PX4 실행의 구분

위험 판단의 위치와 속도는 Vicon만 사용한다. PX4 velocity fallback, Vicon 유실 뒤
dead-reckoning, depth occupancy 대체는 하지 않는다. 지도에 등록된 obstacle의 live Vicon
rigid-body pose와 marker-fitted footprint offset을 사용한다. empty/missing/stale 입력은
안전한 자유 공간으로 처리하지 않는다. 지도 밖 미등록 물체를 검출하는 기능은 아니다.

기존 원격 `vicon_safety_supervisor.py`의 PX4 속도 우선 정책, stale coast,
nominal timestamp 재기록, 종료 뒤 옛 buffer에 의존하는 방식은 새 계약에 사용하지 않는다.
기존 파일과 `vicon_traj/run.sh`는 보존한다. 기존 reference는 `reference/`에 있다.

GT swept projection 검사는 관측한 속도·기하와 명시된 예측시간/여유를 사용한다.
물리적으로 검증된 braking model이나 충돌 방지 보장은 아니다. 개입 거리·예측시간·
기체 반경·센서 timeout과 정지 속도/dwell의 실기체 운용값은 측정 후 설정해야 한다.
테스트 수치는 비행 승인값이 아니다.

위험 감시는 STREAM/CLIMB/MISSION/HOLD와 stop 실행 단계에서 활성화한다. 정상 goal 도착의
LAND/DISARM은 기존 mission의 착륙 절차를 보존하며 새 nominal을 전달하지 않는 passive 단계다.
AUTO_LAND/PILOT/DONE에서도 전달하지 않는다. 이미 발생한 failure는 어떤 단계에서도 지워지지 않는다.

## 설정과 shadow 실행/중지

ROGX 설치 기준 경로는 `/home/rogx/catkin_ws/src`이다.

* 가드 template: `hardware/collision_stop/collision_stop_profile.example.json`.
  `enabled=false`, 숫자는 `null`이다. Vicon map의 **모든** pillar에 subject topic을 지정하고
  실측 기체 offset/radius, 실제 header frame, timeout과 projection 값을 작성해야 한다.
* mission template: `offboard_flight/scripts/guarded_mission.example.yaml`.
  `auto_start=false`, `require_frame_alignment=true`, `stop_policy=settled_velocity`이다.
  session ID는 guard와 같아야 한다. 정지 속도/dwell, velocity·pose·alignment·guard·MAVROS·
  setpoint timeout, AUTO.LAND 재시도 간격/횟수가 필수다.
* shadow는 `commander/collision_stop_shadow_status` 진단만 발행한다. enforce는 별도의
  `commander/collision_stop_status`를 사용하며 profile의 `mode=enforce`와 CLI `--enforce`가
  함께 있어야 활성화된다. guard의 shadow 상태를 live mission이 받아 시작할 수 없다.

숫자와 경로를 채운 **shadow 전용** profile을 `/home/rogx/collision_stop_shadow.json`에
별도로 작성한 뒤 사용할 명령이다. 아래 명령은 이번 배포에서 실행하지 않았다.

```bash
source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash
python3 /home/rogx/catkin_ws/src/offboard_flight/scripts/collision_stop_guard.py \
  --profile /home/rogx/collision_stop_shadow.json __ns:=/rogx2
# 해당 shadow만 중지: 실행 터미널에서 Ctrl-C 또는
rosnode kill /rogx2/collision_stop_guard_shadow
```

FCU/mission이 꺼져 있어도 Vicon 기하와 `would_stop`을 기록한다. 이 경우 `ready=false`이며
실제 비행 개입이라고 표시하지 않는다. log_path는 새 파일이어야 하며 기존 로그를 덮어쓰지 않는다.
timestamp/age, GT 속도와 margin, nominal 원본, execution ACK/phase/hold 위치·관측 mode가 JSONL에 남는다.
guard 프로세스가 죽은 동안의 실행 상태는 별도 rosbag에서 execution topic을 기록해야 한다.

live executor는 새 운용 구성에서 `guarded_mission_node.py`를 별도로 선택하는 방식이다.
기존 mission/fly와 함께 시작하지 않는다. 원래 start 서비스의 takeoff 동작을 상속하므로
본 문서의 shadow 확인 과정에서는 mission start 서비스나 arm/OFFBOARD 서비스를 호출하지 않는다.
새 노드는 guard heartbeat/단일 raw writer/epoch 조건 없이 mission을 시작하지 않는다.
ROS1 publisher 독점 검사는 실행 중 나중에 추가되는 임의의 다른 writer까지 강제로 막지는 못한다.

AUTO.LAND 요청은 비동기로 수행해 hold loop를 막지 않는다. 이미 전송된 service 요청은
취소할 수 없으며, 관측된 수동 전환/reset 뒤에는 새 요청을 보내지 않는다. 재시도가
소진되거나 정지 속도를 확인하지 못하면 hold를 유지하고 이유를 기록한다. PX4 local 상태까지
유효하지 않으면 오래된 command를 재생하지 않고 FCU failsafe/수동 제어에 맡긴다.

## 기존 데이터 수집 보존

기존 `fly.sh`, `preflight.py`, `record_flight.sh`, `mission_node.py`,
`guidance_library.py`, `planar_planner_node.py`, `vicon_traj/run.sh`는 수정하지 않는다.
기존 데이터 수집도 현재는 mission 경로이며, `setpoint_buffer.py`는 과거 경로다.
이번 변경으로 옛 buffer를 복원하지 않는다.

기존 MPPI 노드는 새 factory를 import하지 않으므로 factory의 명칭/로그 변경은
기존 실행을 바꾸지 않는다. 새 live chain은 기존 mission과 동시에 실행하면 안 된다.
기존 `fly.sh`의 stop/start 정리 명령은 `mission_node.py` 문자열로 프로세스를 찾으므로
새 `guarded_mission_node.py`까지 종료할 수 있다. 새 chain 실행 중 기존 fly.sh를 호출하지 않는다.
기존 기록기는 새 safety protocol topic을 자동 포함하지 않으므로 별도 기록 설정이 필요하다.

## 백업과 복구

[백업 manifest](evidence/backups.json)에 현재 소스 Git 스냅샷과 archive checksum을 기록한다.
기존 전체 모델·bag·무시 파일 백업도 유지한다. 원래 브랜치와 Git index는 변경하지 않는다.

새 노드를 실행하지 않았으면 기존 수집으로 돌아가기 위한 restart/reset은 필요 없다.
복구 검토는 archive를 빈 디렉터리에 풀어 파일 단위로 비교한다. 운용 디렉터리 전체에
archive를 덮어쓰거나 `git reset --hard`를 사용하지 않는다. 실제 실행 중에는 mission을
종료하는 것이 안전 정지를 뜻하지 않으며, 본 작업에서는 실행하지 않는다.

## 남은 전체 HPA/DeSimplex 연결

이번 가드 구현이 실제 producer ROS 연결 전체를 완료하지는 않는다. V4 checkpoint,
depth/PX4 계약, CUDA runtime은 기존 구현을 유지하며 다음 작업은
[통합 TODO](../hpa_integration/TODOLIST.md)의 ROS reference 시간·epoch·추종 연결,
세 모드별 실행/기록 구성, PX4 포함 replay, 센서+mapping+MPPI+CUDA 동시 지연 측정이다.
추론/단위 테스트 성공은 실제 제동·비행 안전성 검증을 뜻하지 않는다.
