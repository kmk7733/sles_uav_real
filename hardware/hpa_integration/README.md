# HPA V4 / DeSimplex 공통 producer 구현

2026-09-22 후속: 역할 이름은 `DeSimplexSwitcher` / `TrajectorySafetyValidator`로
구분한다. 기존 클래스와 같은 alias이며 알고리즘은 바꾸지 않는다. 별도 Vicon
`CollisionStopGuard`와 `guarded_mission_node.py`의 정지→속도 확인→AUTO.LAND 계약은
[정지 가드 README](../collision_stop/README.md)를 따른다. 기존 데이터 수집용 실행 파일은 보존한다.

이번 범위는 `planner/hpa/`, `offboard_flight/scripts/planar_producer_factory.py`,
`offboard_flight/scripts/hpa_depth_adapter.py` 및 subscriber-only shadow 연결이다. 기존
`planner/supervisor.py`, `planner/transition.py`의 알고리즘은 그대로 사용한다.
기체에서도 같은 Python 파일을 사용한다.

`haa`는 FrontierMPPI, `hpa`는 V4 단독 producer, `desimplex`는 V4와
두 개의 HAA 인스턴스(실행용과 안전 집합 검사용)를 연결한 supervisor이다.
각 모드의 결과는 `MPPIResult`와 `PlanarReferenceSequence`이다.
HPA의 가속도 출력을 ROS 속도·위치·추력 명령으로 직접 해석하지 않는다.

## 역할과 변경 범위

* HPA: 고정된 관측 시점의 body FLU 가속도 10개를 하나의 기준 yaw로
  계획 좌표계에 회전하고, 기존 가속도·jerk 제한과 CappedDynamics를 거쳐
  reference를 만든다. 지도 기반 충돌 회피 보장은 HPA 단독에 추가하지 않는다.
* Supervisor: 충돌 후 검출기가 아니라, HPA 제안의 허용 가능성·예측 경로·
  회복 가능성을 검사하여 HPA/HAA/회복 동작을 선택하는 DeSimplex이다.
  `plan()`을 호출해야 reference transition 검사까지 실행된다.
* Transition: 선택된 두 reference 사이를 연결하고 연결 경로의 제약을 검사한다.
  ROS 메시지 동기화나 publish 루프는 담당하지 않는다.
* Factory: 명시한 모드와 설정으로 위 객체를 구성한다. ROS 노드를 시작하지 않는다.

`planar_planner_node.py`, `fly.sh`, `preflight.py`, `record_flight.sh`,
`mission_node.py`, `guidance_library.py`는 이번 작업에서 수정하지 않는다.
기존 fly.sh의 HAA 실행 경로는 그대로이다. 새 factory만 추가한다고 기존 노드가
HPA나 DeSimplex로 전환되는 것은 아니다.

## 센서 연결 시 지켜야 할 계약

공통 producer는 ROS 노드가 아니다. ROS depth adapter는 메시지 timestamp와
학습 필드를 연결하며, 실제 구독·추론·trajectory 로그는 `hardware/hpa_shadow/`가 담당한다.
그 runner에도 ROS publisher나 FCU service client는 없다.

호출자는 depth, 실제 CameraInfo K, PX4 fused local ENU 상태와 동일 좌표계 goal을
동기화하고 관측 시각·frame·epoch를 검증해야 한다. Vicon을 사용하지 않는다.
V4에는 원시 PX4 yaw와 causal 원본 `odom.twist.twist.angular.z`를 넣는다.
선형 속도는 causal 원본 `velocity_local.twist.linear`만 사용한다. 계획 상태의 yaw rate는
시뮬레이터와 같은 Euler yaw 미분이므로 body angular-z와 구분한다.
PX4 local과 정렬된 계획 좌표계가 다르면 goal·상태·출력 기준 yaw를 함께 변환해야 한다.
관측 상태와 reference의 시작 상태는 같은 시점이어야 한다.

ROS 계층은 매 tick의 stale 입력, 비정상 시간, local-position/yaw reset과 FCU 연결을
검사해야 한다. cached chunk나 진행 중인 transition을 사용 중이어도 생략할 수 없다.
계획 시각은 입력 시점이며 추론 완료 시각으로 늦추지 않는다.
새 mission 또는 좌표 epoch에는 factory로 producer 전체를 다시 생성한다.
정상 DeSimplex 전환마다 HPA의 commit 상태를 초기화하지 않는다.

Supervisor의 `last_decision.fault`는 별도로 기록해야 한다. 최후의 best-effort
braking도 결과를 낼 수 있으므로, reference 존재나 `BRAKING` 상태만으로
안전 검증 성공으로 판정하지 않는다. 로그를 위해 `classify()`를 추가 호출하면
probe 연산과 상태에 영향을 줄 수 있으므로 실제 `last_decision`을 기록한다.

## 변경 전 백업

원본 브랜치와 Git index는 유지한 채 별도 백업 ref에 커밋했다. push는 하지 않았다.

| 대상 | 백업 커밋 | 전체 아카이브 |
|---|---|---|
| 로컬 | `32e7aec69ebee1f5e5501d04301224608cfd422b` | `/home/acrl/research/rogx_backups/20260920T_hpa_local/complete-before.tar` |
| ROGX | `14c13711ebe59f1b62cd8a9e3f7583e0f10fe4fd` | `/home/rogx/backups/20260920T_hpa_src/complete-before.tar` |

ROGX 아카이브는 로컬
`/home/acrl/research/rogx_backups/20260920T_hpa_rogx_copy/`에도 복사하고 SHA256을 확인했다.
전체 아카이브는 무시된 파일, 대용량 학습 데이터·bag, nested Git 이력을 포함한다.
백업 커밋은 무시된 소스·설정·모델과 외부 패키지 실제 소스를 포함한다.
생성 캐시 및 64 MiB 초과 미추적 파일은 Git 객체 중복을 피하고 전체 아카이브에
보관했다. 정확한 목록은 각 백업 디렉터리의 `manifest.json`에 있다.

원본을 덮어쓰지 않고 ROGX의 백업을 검토하는 방법:

```bash
mkdir /home/rogx/review_pre_hpa_backup
tar -xf /home/rogx/backups/20260920T_hpa_src/complete-before.tar \
  -C /home/rogx/review_pre_hpa_backup
# 복구본: /home/rogx/review_pre_hpa_backup/src/
git -C /home/rogx/catkin_ws/src show \
  backup/pre-hpa-20260920T_hpa_src:offboard_flight/scripts/fly.sh
```

새 모듈은 기존 실행 경로에 연결하지 않았으므로 기존 HAA로 되돌리기 위한
서비스 재시작·브랜치 reset·파일 덮어쓰기는 필요 없다.
후속 변경이 있으면 백업과 비교하여 필요한 파일만 복원한다.

## 검증의 범위

검증 결과는 `evidence/`에 기록한다. 기존 switching/transition 회귀 검증,
기존 HPA 후처리·commit과의 수치 비교, 세 factory 모드, 고정 V4 fixture와
CUDA 출력 일치를 검사한다. 다른 체크포인트나 test40 성적은 사용하지 않는다.

ROS·센서·mapping·MPPI가 모두 실제 입력으로 계산 중인 조건의 실기체 지연은
PX4 heartbeat와 입력 복구 후 따로 측정해야 한다. synthetic 검증 시간을 그 조건의
지연으로 보고하지 않는다. 이 구현 및 검증은 arm·OFFBOARD·이륙·비행을 실행하지 않는다.
