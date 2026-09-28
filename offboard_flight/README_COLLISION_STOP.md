# 새 Vicon 충돌 정지 운용 경로

명칭·권한·구성·shadow 명령·백업/복구·검증 결과는
[CollisionStop 문서](../hardware/collision_stop/README.md)를 따른다.

기존 `fly.sh`, `preflight.py`, `record_flight.sh`, `mission_node.py`,
`guidance_library.py`, `planar_planner_node.py`와 데이터 수집 경로는 유지한다.

새 opt-in 경로는 `commander/set_pose → collision_stop_guard.py → commander/set_pose_safe
→ guarded_mission_node.py → MAVROS/PX4`이다. 기본 예제는 비활성이며 배포만으로 실행되지 않는다.
HPA/HAA/DeSimplex producer 선택과 ROS reference 연결 전체의 완료를 의미하지 않는다.
