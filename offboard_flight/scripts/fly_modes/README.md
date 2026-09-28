# 세 실행 경로와 DeSimplex ROS 연결 — 2026-09-25 밤 작업

모델은 승인된 V4 epoch 35 그대로다(`best.pt` SHA256 `d05e6dc4…edc2`).
FCU 명령, arm, OFFBOARD, 이륙·비행, 정지·착륙 서비스 호출은 하지 않았다. Vicon은 꺼져 있었고,
모든 검증은 ROGX의 전용 loopback ROS master(포트 11350~11354)에서 녹화 데이터
`session_20260922_030246`으로 했다. 기존 노드와 서비스는 시작하거나 멈추지 않았다.

용어: **DeSimplex safety monitor** = 경로 검사 + HPA/HAA 전환·recovery·bridge(지도 기반).
**StopBeforeCollision** = CollisionStopGuard(Vicon GT로 충돌 판단) + guarded mission의 정지 실행
(고정 hold → PX4 속도 확인 → AUTO.LAND).

---

## 확인해 주실 것

1. **DeSimplex에서 0.5초 경로 유효 시간.** HPA 단독은 문제없다(만료 hold 20회).
   DeSimplex는 tick 처리 중앙값이 205ms(supervisor만 130ms)이고, 계산이 끝날 때 depth 나이가
   중앙값 344ms, tick 간격이 0.2초다. 그래서 매 tick 끝에 약 40ms씩 속도 0 hold가 들어간다
   (녹화 재생에서 732회, 목표 메시지 발행은 50Hz로 끊기지 않음). 실비행에서는
   "이동 → 짧은 정지 명령 → 이동"이 반복되어 움직임이 끊길 수 있다.
   선택지: (a) 0.5초 유지 후 실비행에서 확인, (b) DeSimplex만 1.0초, (c) 계산 시간 단축.
   코드는 (a) 상태다.
2. **StopBeforeCollision 수치.** 아래 표의 값을 확인해 주세요. 기존 `vicon_safety_supervisor.py`에
   없던 값은 제가 정했고, 근거를 함께 적었다.
3. **HPA/DeSimplex 시작 대기.** 새 순서에서는 hover 후에 planner를 시작한다. V4 모델 로드와
   CUDA 초기화에 10~20초가 걸리고, 그동안 기체는 mission이 정한 위치에서 hover한다.
4. **첫 경로 전 hold 없음.** 첫 V4 경로가 나오기 전에는 명령을 보내지 않는다. 기체는
   mission의 hover를 유지하고, 첫 명령이 오면 MISSION으로 들어간다.
5. **경로 만료 후 hold가 오래 지속될 수 있음.** 추론이 멈춰도 hold는 계속 나가고,
   mission_timeout(120초)에서 끝난다. 시뮬레이터와 기존 HAA planner의 동작과 같다.
   독립 검토에서 이 점을 지적했다. 추론이 멈추면 발행도 멈추는 방식이 더 좋다면 알려 주세요.
6. **`fly_*.sh`는 실제 시스템에서 아직 실행하지 않았다.** MAVROS/FCU가 필요하다. 처음에는
   FCU를 켜고 모터 전원을 분리한 상태로 `start`와 `state`까지만 확인하는 것을 권한다.
   `go`는 별도 승인 후에 한다.
7. **Vicon PC 시계.** guard는 이 컴퓨터보다 0.05초 넘게 앞선 Vicon 시각을 거부한다.
   `start`의 사전 점검이 이를 확인한다.
8. **`fly.sh stop` 주의.** 기존 `fly.sh stop`의 `pkill -f "mission_node.py"`는
   `guarded_mission_node.py`도 종료한다. `fly_*.sh`로 비행하는 동안에는 쓰면 안 된다.

---

## 1. 합의대로 바꾼 것

| 항목 | 이전 | 지금 |
|---|---|---|
| 출력 시점 최신 depth 250ms 검사 | 넘으면 발행 중단 | 없음. PX4 pose·속도, FCU, 좌표 epoch 검사는 유지 |
| 경로 유효 시간 | depth 시각 + 1초 | depth 시각 + **0.5초**(시뮬레이터 `ReferenceHolder`, 기존 planner `plan_timeout`) |
| 경로가 끝난 뒤 | 발행 중단 | 마지막 명령 위치·yaw에서 속도·가속도 0 hold(시뮬레이터와 같음) |
| 경로 길이 | 11 node(HPA)만 수락 | 2 node 이상 모두. 끝을 지나면 마지막 node로 clamp |
| 이전 가속도(a_prev) | 경로 만료 뒤 0 | 마지막 수락 경로의 a[0](시뮬레이터 harness, 기존 planner) |
| `--duration` | 기본 60초 | 실제 제어 모드에는 없음. 재생·preview에서만 사용 |
| 실행 순서 | planner → start → 이륙 → hover → 추종 | **start → go: 이륙 → hover → planner 시작 → 첫 명령에서 자동 MISSION** |

### StopBeforeCollision 수치

| 설정 | 값 | 근거 |
|---|---:|---|
| 기체 반경 | 0.31m | 기존 `--drone-radius` |
| lookahead | 0.5초 | 기존 `--lookahead` |
| 정지 여유 | 0.10m | 기존 `--d-stop` (0.5m/s에서 약 10cm 전 정지로 보정된 값) |
| 판단 주기 | 20Hz | 기존 `--rate` |
| Vicon 끊김 허용 | 0.5초 | 기존 `--vicon-timeout` 설명(코드 기본값 1.0초와 다름) |
| Vicon 시각 앞섬 허용 | 0.05초 | 기존 mission의 명령 시각 허용치 |
| Vicon 속도 차분 간격 / subject 간 시각차 | 0.1초 | 기존 `--vel-window` |
| 비정상 속도 거부 | 3.0m/s | 기존 `--vel-sane` |
| 기둥 회전 속도 한도 | 5 rad/s | 기둥은 고정. 마커 뒤바뀜만 거름 |
| mission 상태 / planner 명령 나이 | 0.5초 | 기존 mission `sp_timeout` |
| 정지 확인 → AUTO.LAND | 0.1m/s 이하 1.0초 | 사용자 결정 |
| PX4 속도 샘플 허용 | 0.5초 | 기존 mission `pose_timeout` |
| guard heartbeat | 0.5초 | = `sp_timeout` |
| MAVROS state 나이 | 2.0초 | 기존 preflight(state 1Hz) |
| AUTO.LAND 재요청 / 최대 횟수 | 0.5초 / 20회 | 기존 mission 0.5초 간격, 10초 한도 |

**guard heartbeat가 필요한 이유.** 새 순서에서는 이륙·hover·착륙 동안 planner 명령이 없다.
이 구간에 guard 프로세스가 죽으면, mission이 그 사실을 알 수 있는 방법은 guard 상태 메시지가
끊기는 것뿐이다. 그래서 유지했다. 기존 guard는 `ready`에 planner 명령을 요구해
planner 없이는 이륙할 수 없었다. 이를 위해 planner 명령과 무관한 `takeoff_ready`를 추가했다.

## 2. 실행 방법 (ROGX)

먼저 `~/start_test_grid.sh vicon`을 실행한다(MAVROS, ZED, Vicon, 정렬, mapper).

```bash
cd /home/rogx/catkin_ws/src/offboard_flight/scripts
MAP=/home/rogx/traj/<오늘 기록한 세션>/map.yaml GOAL_X=2.0 GOAL_Y=0.0 ./fly_haa.sh start
MAP=... GOAL_X=... GOAL_Y=... MOUNT_CONFIRMED=1 ./fly_hpa.sh start
MAP=... GOAL_X=... GOAL_Y=... MOUNT_CONFIRMED=1 ./fly_desimplex.sh start
./fly_<mode>.sh go      # 이륙 → hover 확인 → 그 고도로 planner 시작 → 자동 MISSION
./fly_<mode>.sh state   # guard·mission·FCU 상태
./fly_<mode>.sh land    # 착륙
./fly_<mode>.sh stop    # 착륙·disarm 뒤에만(비행 중에는 거부, 지상에서 FORCE=1)
```

- goal은 Vicon world 좌표로 준다. HPA와 DeSimplex에는 스크립트가 공유 정렬로 PX4 local로 바꿔
  넘긴다. HPA 입력에 Vicon을 넣는 것은 아니다.
- 이륙 고도는 mission의 `home_z + TAKEOFF_HEIGHT`(기본 1.0m)다. hover가 확인되면 스크립트가
  같은 고도를 planner에 넘긴다(HAA는 world z, HPA/DeSimplex는 local z).
- 기록: `/home/rogx/flights/<mode>_<시각>/`에 bag, guard 로그, planner 로그, guard profile을 남긴다.
  HPA/DeSimplex 모드의 bag에는 raw depth도 포함된다.
- 프로세스는 PID 파일로만 멈춘다(`pkill -f` 사용 안 함).

## 3. DeSimplex ROS 연결

`desimplex_planner_node.py`는 HPA 노드의 입력, 출력, 경로 수명, 도착 처리를 그대로 쓴다.
planner만 원본 `DeSimplexSupervisor`의 전체 `.plan()`(factory `build_producer("desimplex")`)으로 바뀐다.

- **tick:** 동기화된 depth 1장이 `.plan()` 1회다. 시뮬레이터도 learned HPA를 매 10Hz tick마다
  다시 부르므로(commit 1) 같은 의미다. commit·bridge 카운터는 ROS timer가 아니라 호출 횟수로 진행된다.
- **좌표:** HAA는 현재 epoch의 `/grid_map`(world)에서 계획한다. depth 시각의 PX4 상태를 local에서
  world로 바꾸고, 결과 경로를 같은 정렬로 world에서 local로 되돌린다. V4 입력은 PX4 local 그대로이고,
  V4 가속도를 회전하는 yaw만 world 기준으로 넘긴다.
- **지도:** 기존 HAA planner와 같은 규칙이다(unknown은 위험, 기체 발밑은 `r_safe + 0.05` 비움).
- **설정:** `planar_producer_config.desimplex_parity.json`. 시뮬레이터 `config.yaml`에 기존 ROGX HAA 값
  두 개만 바꿨다(`mppi.horizon=60`, `r_safe` 0.51). ROS 노드와 시뮬레이터 비교가 같은 파일을 쓴다.
- **기록:** 모든 tick(`desimplex_tick`)에 mode, source, reason, fault, switched, bridge, U, X,
  world 경로, 입력(state, goal, a_prev, V4 chunk, 지도 파일)을 남긴다. `fault=True`는
  최선의 제동이지 검증된 안전 recovery가 아니라고 로그에 명시한다.
- **실패 처리:** supervisor 내부 예외는 세션을 종료한다. 계산은 됐지만 늦어서 실행하지 못한
  tick은 `desimplex_tick_not_flown`으로 따로 센다(시뮬레이터는 모든 tick을 실행한다).

## 4. 검증 결과

| 검증 | 결과 |
|---|---|
| HPA 녹화 재생 + mapping·MPPI 동시(`evidence/hpa_hold05_02`) | 목표 메시지 50.00Hz, 최대 간격 47ms, 100ms 초과 **0회**(이전 10~13회, 최대 약 320ms). 계산 517회 모두 수락, 만료 hold 20회. 도착 판정 후 hold 1,569개(leash 계산과 모두 일치). 기존 mission 수신 검사 4,141/4,141. mapping 1,195회·MPPI 1,193회 |
| DeSimplex 녹화 재생 + 기존 정렬·Andert mapper 노드(`../desimplex_replay/evidence/desimplex_replay_03`) | tick 231회(HPA 206, RECOVERY 1, HAA 24, 전환 1, fault 0). 경로 수락 230, 늦어서 미실행 1. 지도 788장 수신. 목표 메시지 50.00Hz, 100ms 초과 0회. 만료 hold 732회(확인 1번) |
| 시뮬레이터 동일 입력 비교 | **결정** mode/source/reason/fault/switched/bridge: 231/231 일치. **경로** HPA·RECOVERY는 완전히 같고, HAA 24 tick은 최대 4.5×10⁻⁷ m 차이. 원인은 CPU 차이다: ROGX(aarch64)에서 새 factory로 재현하면 노드 출력과 차이 0(231/231). 로컬(x86)에서 factory와 시뮬레이터 생성자는 서로 완전히 같다 |
| 비교 도구 음성 대조 | U 1e-6 변경과 source 변경을 불일치로 판정 |
| 단위 시험 | 로컬: unittest 136개 + pytest 117개 통과(collision_stop 112, analyzer 5). ROGX(CUDA 포함) 120개 통과 |
| 독립 검토 | 치명 2건(실제 제어 모드 시작 시 JSON 오류, HAA·recovery 경로 폐기)과 주요 2건(a_prev, 비행 중 stop)을 고치고 시험을 추가했다 |
| 온도 | 모든 재생에서 최고 37°C(시작 전 50°C 이하 대기, 75°C에서 중단하도록 설정). fan은 cool 프로파일 |

**검증하지 않은 것:** 실제 FCU와 모터가 있는 `fly_*.sh go`, StopBeforeCollision의 실제 정지·착륙,
실시간 ZED·Vicon을 쓰는 지상 시험, 비행 중 추종 성능.

## 5. 파일

- 새로 추가: `offboard_flight/scripts/{desimplex_planner_node.py, fly_haa.sh, fly_hpa.sh, fly_desimplex.sh,
  planar_producer_config.desimplex_parity.json}`, `offboard_flight/scripts/fly_modes/{fly_common.sh,
  fly_helper.py, guarded_mission.rogx.yaml}`, `hardware/desimplex_replay/*`, `hardware/thermal/thermal_gate.py`,
  `hardware/rogx_deploy/deploy.py`.
- 수정: `hpa_planner_node.py`, `collision_stop_core.py`, `mission_stop_contract.py`,
  `guarded_mission_node.py`, `hardware/hpa_shadow/shadow_node.py`, `hardware/hpa_today/{ros_shadow.py,
  reference_lifecycle.py, feed_ros_bag.py}` 및 관련 시험.
- 바꾸지 않음: `fly.sh`, `preflight.py`, `record_flight.sh`, `mission_node.py`, `guidance_library.py`,
  `planar_planner_node.py`, `start_test_grid.sh`, 데이터 수집 설정(작업 후 hash 확인).

## 6. 백업·복구·Git

ROGX 설치는 3단계로, 각 파일의 이전 hash가 계획과 같을 때만 바꿨다. 되돌릴 때는 **최신부터** 실행한다.

```bash
python3 -B /home/rogx/backups/review-fixes-20260925T052830Z/rollback.py
python3 -B /home/rogx/backups/replay-tfstatic-20260925T052313Z/rollback.py
python3 -B /home/rogx/backups/fly-modes-desimplex-20260925T051456Z/rollback.py
```

Git(원래 branch·HEAD·index는 그대로 두고 별도 ref에 저장):
로컬 `refs/heads/implementation/fly-modes-desimplex-20260925`(부모 `767e2a3`),
ROGX `/home/rogx/catkin_ws/src`의 같은 이름 ref `04aed9f`(부모 `4f27784`).

그 이전 단계(목표 도착 처리)의 복구는 `../hpa_goal_arrival/README.md`를 따른다. 대용량 로그 3개는
Git에 넣지 않았다. 원본은 ROGX `results/`에 있고, 로컬 압축본은
`/home/acrl/research/rogx_backups/fly-modes-20260925/evidence_logs.tar.xz`(SHA256 `3f8e6cb1…`),
파일별 SHA256은 `evidence/LARGE_FILES.sha256`에 있다. 시뮬레이터 비교를 다시 하려면 이 압축본의
`desimplex.jsonl`과 `grids/`가 필요하다.

## 7. 남은 일

1. 위 "확인해 주실 것" 결정.
2. FCU 켜고 모터 분리 상태에서 `fly_*.sh start/state` 지상 확인. `go`는 승인 후.
3. 실시간 센서 지상 측정(ZED stereo, PX4, Vicon, mapping, planner, guard 동시).
4. 학습의 0.02초 격자 보간과 live 보간 정책 비교(이전부터 남은 항목).
