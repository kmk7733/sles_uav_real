# HPA 전처리와 기존 제어기 연결

이번 변경은 V4 모델과 시뮬레이션 알고리즘을 그대로 사용한다. 재학습하지 않는다.
기준 모델은 epoch 35, SHA256
`d05e6dc4b0cb410eea1cf2f274946481244ad1fa6c23cc77856d1201c9d2edc2`이다.

## 세 가지 변경

1. **반복 계산 줄이기.** K와 해상도가 같으면 픽셀별 광선 방향을 다시 만들지 않는다.
   Depth, 자세, 높이 필터, 512방향별 최소 거리 계산은 원본과 같다.
   배포 패키지 파일은 수정하지 않았고, 검증된 원본 함수의 고정 계산 부분에만
   캐시를 적용한다. 원본 함수 내용이 달라지면 적용을 거부한다.
2. **계산한 경로를 사용하는 시간 수정.** 사용자 요청으로 계산 완료 후의 입력
   허용 시간을 1초로 변경했다(`--completion-max-age 1.0`). 계산 시작 및 최신
   센서 수신 검사인 `--max-age 0.25`는 유지한다. 받은 경로는 depth 시각부터
   시작한 1초짜리 경로로 보관한다. 완료가 늦어져도 경로 종료 시각을 늦추지 않는다.
   예를 들어 depth 시각이 10.00초이고 계산이 10.22초에 끝났으면,
   10.40초에는 그 경로의 0.40초 위치를 사용한다. 10.25초에 버리지 않는다.
   계산 완료 시각을 경로 시작 시각으로 바꾸지도 않는다.
3. **기존 제어기 입력으로 변환.** 새 `hpa_planner_node.py`가 기존 HPA factory와
   적분 코드를 호출한다. 경로의 위치·속도·가속도·yaw·yaw rate를 기존
   `Controller.construct_target_full`로 변환한다. 고도는 명시한 local ENU 값으로
   고정하며 수직 속도와 가속도는 0이다. 기존 1.2m 위치 명령 제한도 유지한다.

최신 depth·PX4·FCU 상태는 출력할 때마다 별도로 검사한다. 입력이 끊기거나
잘못된 값, 시간 역행, 좌표 epoch 변경이 확인되면 출력하지 않는다.
CameraInfo는 최초 유효 K를 보관하며 매번 새 CameraInfo를 기다리지 않는다.
경로의 끝점은 반복 발행하지 않는다. 기존 mission의 명령 timeout은 그대로이므로,
HPA 발행 중단 시각과 기체의 정지 시각은 동일하다는 뜻이 아니다.

## 파일과 실행 경로

ROGX의 실제 사용 파일:

- `/home/rogx/catkin_ws/src/offboard_flight/scripts/hpa_planner_node.py`: 새 HPA 실행 파일
- 같은 디렉터리의 `hpa_controller_bridge.py`: 기존 Controller에 전달할 값 구성
- `/home/rogx/catkin_ws/src/planner/hpa/scan_cache.py`: 고정 광선 계산 캐시
- `/home/rogx/hpa_current`: 이번 검증이 끝난 HPA 코드·모델 디렉터리

기존 `fly.sh`, `preflight.py`, `record_flight.sh`, `mission_node.py`,
`guidance_library.py`, `planar_planner_node.py`와 데이터 수집 파일은 변경하지 않는다.
기존 `fly.sh`는 계속 기존 MPPI를 실행한다. 새 HPA 노드는 별도로 실행한다.
이 노드는 HPA 단독용이다. DeSimplex의 switching/recovery 코드는 변경하지 않았으며,
DeSimplex를 실제 ROS 실행 경로에 연결하는 작업은 이 노드에 포함되지 않는다.

## Shadow 실행과 중지

Shadow는 실제 입력을 받아 경로를 만들되 기체 명령 대신 시험용 토픽으로 보낸다.
다음 명령은 기존 노드나 센서를 시작·종료하지 않는다. PX4가 꺼져 있으면 입력을
기다리며, Vicon으로 PX4 입력을 대체하지 않는다.

```bash
source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -B /home/rogx/catkin_ws/src/offboard_flight/scripts/hpa_planner_node.py \
  --bundle /home/rogx/hpa_current/deploy/rogx_hpa_v4 \
  --log /tmp/hpa-preview-$(date +%Y%m%dT%H%M%S).jsonl \
  --device cuda --goal-current-position --z-local 1.0 \
  --depth-frame zed2i_left_camera_optical_frame --duration 60
```

`--goal-current-position`은 시작 위치를 시험용 goal로 사용하는 선택이다.
`--z-local 1.0`도 이 명령에서 출력 형식을 확인하기 위한 예시 고도다.
실제 navigation에는 같은 PX4 local ENU의 `--goal-local X Y Z`와 고도를 지정한다.
출력 토픽 기본값은 `/hpa_shadow/position_target`이다. 중지는 실행한 터미널의 Ctrl-C.
이 노드에는 arm·OFFBOARD·이륙·착륙 서비스를 호출하는 코드가 없다.

## 실제 제어 연결

코드는 다음 경로를 지원하지만 이번 작업에서는 이 경로로 실행하지 않는다.

```text
depth + PX4 → V4 → 기존 trajectory 적분 → 기존 Controller 메시지 생성
  → commander/set_pose → CollisionStopGuard → commander/set_pose_safe
  → guarded_mission_node → MAVROS
```

`--controller-output`을 명시해야 nominal `commander/set_pose`를 발행한다.
이때 CUDA, 명시적 local ENU goal, 카메라 장착 확인, 현재 유효한 공유 좌표 epoch가
필요하다. HPA 입력과 출력은 이미 local ENU이므로 Vicon/world 회전을 다시 하지 않는다.
공유 alignment는 기존 mission과 동일한 epoch 확인에만 사용한다.
같은 토픽의 기존 planner 발행자가 있으면 시작을 거부하며 자동으로 중단하지 않는다.
기존 mission 직결 대신 지정한 CollisionStopGuard가 수신하도록 구성해야 한다.
기록기가 같은 입력 토픽을 구독한다면 `--observer-node /기록기노드이름`으로 명시한다.
이 목록에는 수동 관측·기록 노드만 넣고 mission이나 명령 전달 노드는 넣지 않는다.

CollisionStopGuard는 **Vicon GT로 충돌을 판단하는 독립 정지 기능**이다.
DeSimplexSwitcher의 HPA/HAA 선택·recovery와 역할이 다르다. 정지 후 고정 local hold,
속도 정지 확인, AUTO.LAND는 기존 별도 `guarded_mission_node.py`가 담당한다.
이번 변경으로 그 실행 조건이나 수치를 새로 정하지 않는다.

새 HPA 노드는 기존 planner의 goal-arrived 토픽을 아직 발행하지 않는다.
따라서 목표 도착 후 mission 종료·착륙까지 검증됐다고 해석하면 안 된다.
실제 실행 전에는 기존 정지 가드 설정, 공유 epoch 공급, 명시적 goal/고도,
목표 도착 처리와 최신 센서·카메라 부하에서의 지상 확인이 남아 있다.

## 검증과 복구

`RESULT.md`와 `evidence/`에 실행 결과를 보관한다. Bag 재생 측정에는 실제 mapping과
MPPI 계산을 포함하지만 현재 카메라의 GPU stereo 계산과 실제 FCU는 포함하지 않는다.
제어기 연결 검사는 실제 ROS PositionTarget과 기존 mission의 입력 검사·forward 함수를
사용하며, mission 생성자/실행 루프와 FCU 출력은 호출하지 않는다.

변경 전 ROGX 전체 소스 백업:
`/home/rogx/backups/hpa-controller-20260924/src-before.tar.gz`.
Git 백업 커밋: `1d6d3a8c5d7eee736c3eb77cdbda984559dc736c`.
기존 Git 브랜치와 index는 보존한다. 이번 코드만 되돌리려면 다음 명령을 사용한다. 이후 사용자 수정이 있으면 덮어쓰지 않고 거부한다.

```bash
python3 -B /home/rogx/backups/hpa-controller-20260924/install_rogx.py --rollback
```

기존 `fly.sh` 실행 경로는 복구 없이 그대로 사용할 수 있다.
