# mpc_controller

Sampling-based MPC (MPPI) trajectory node for a quadrotor. It consumes the ZED 2D
occupancy grid (`/grid_map`) and the vehicle pose, runs an MPPI planner (ported
from simulation), and publishes the planned trajectory as a `nav_msgs/Path` for
visualization in Foxglove / RViz.

**Current stage:** visualization only — the node publishes a trajectory but does
**not** yet command MAVROS. Control-to-MAVROS is the next stage.

---

## Package layout

```
mpc_controller/
├── scripts/
│   └── mpc_node.py          # ROS node: grid + pose + goal -> /mpc/trajectory
├── src/mpc_controller/      # importable package (MPPI planner, ported from sim)
│   ├── mppi.py              # MPPI core sampler + MPPIPlanner.plan()
│   ├── quadrotor.py         # PlanarHolonomic (plan surrogate) + full Quadrotor
│   ├── obstacle_map.py      # OccupancyGrid -> collision/EDT adapter (the bridge)
│   ├── cost_to_go.py        # geodesic Dijkstra field (avoids head-on local minima)
│   ├── haa_planner.py       # tube-MPPI safety wrapper (unused by node yet)
│   ├── tracker.py           # geometric tracker (unused by node yet)
│   └── drone_config.py      # real-drone plant/tracker constants
├── launch/mpc.launch
├── CMakeLists.txt / package.xml / setup.py
└── README.md
```

## I/O

| Direction | Topic | Type | Note |
|-----------|-------|------|------|
| in  | `/grid_map` | `nav_msgs/OccupancyGrid` | ZED depth grid, frame `map` |
| in  | `/zed/zed_node/pose` | `geometry_msgs/PoseStamped` | pose (frame `map`), vel via finite-diff |
| in  | `/move_base_simple/goal` | `geometry_msgs/PoseStamped` | click a goal in Foxglove/RViz (optional) |
| out | `/mpc/trajectory` | `nav_msgs/Path` | planned trajectory, at `z_hold` |
| out | `/mpc/goal_marker` | `visualization_msgs/Marker` | goal sphere |

The node plans toward a default goal (`~goal_x`,`~goal_y`, default `(3, 0)`)
immediately; publishing `/move_base_simple/goal` overrides it at runtime.

---

## Build (once)

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg mpc_controller
source devel/setup.bash
```

---

## Run the full stack

The whole pipeline is bound to the LAN IP so a remote PC can subscribe via
Foxglove. Replace `10.193.212.240` with this machine's LAN IP (`hostname -I`).

Set the environment in **every** terminal first:

```bash
export ROS_MASTER_URI=http://10.193.212.240:11311 ROS_IP=10.193.212.240
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

### 1. Master

```bash
roscore
```

### 2. ZED depth -> occupancy grid (`/grid_map`)

Start the ZED **on the ground before takeoff** (grid z=0 = tracking start pose).

roslaunch zed_rtabmap_example zed_depth_grid.launch camera_model:=zed2i open_rviz:=false
```

### 3. Foxglove bridge (for the remote PC)

Whitelist is limited on purpose — advertising all ~190 ZED topics makes Foxglove
silently fail to subscribe to `/grid_map`.

```bash
roslaunch foxglove_bridge foxglove_bridge.launch port:=8765 \
  send_buffer_limit:=100000000 \
  topic_whitelist:="['/grid_map', '/mpc/trajectory', '/mpc/goal_marker', '/rogx2/zed2i/zed_node/pose', '/move_base_simple/goal', '/tf', '/tf_static']"
```

### 4. MPC node

```bash
roslaunch mpc_controller mpc.launch pose_topic:=/rogx2/zed2i/zed_node/pose
# Or
roslaunch mpc_controller mpc.launch pose_topic:=/rogx2/mavros/local_position/pose
```

---

## View on the remote PC (Foxglove)

1. Foxglove Studio → **Open Connection → Foxglove WebSocket** → `ws://10.193.212.240:8765`
2. 3D panel → **Display frame = `map`**
3. Enable `/grid_map`, `/mpc/trajectory`, `/mpc/goal_marker`
4. Set a goal: use a Publish action / 3D "Publish Pose" on `/move_base_simple/goal`
   (`geometry_msgs/PoseStamped`) and click a point — the trajectory re-plans toward it.

Publish a goal from the CLI instead:

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

---

## Iterating on the MPPI

Edit `scripts/mpc_node.py` or `src/mpc_controller/*.py`, then restart **only** the
MPC node — leave roscore / ZED / bridge running:

```bash
rosnode kill /mpc_node
roslaunch mpc_controller mpc.launch pose_topic:=/zed/zed_node/pose
```

To restart only the grid node (edited `depth_to_grid.py`) without re-initializing
the ZED camera, kill it and re-run standalone with the launch's params:

```bash
rosnode kill /depth_to_grid
rosrun zed_rtabmap_example depth_to_grid.py \
  /depth_to_grid/cloud_in:=/zed/zed_node/point_cloud/cloud_registered \
  /depth_to_grid/grid_out:=/grid_map \
  _world_frame:=map _resolution:=0.05 \
  _grid_min_x:=-1.0 _grid_max_x:=10.0 _grid_min_y:=-5.0 _grid_max_y:=5.0 \
  _obstacle_min_height:=0.75 _obstacle_max_height:=1.25 \
  _auto_floor:=false _floor_z:=0.0 _max_range:=6.0
```

---

## Key parameters (`mpc.launch`)

| Param | Default | Meaning |
|-------|---------|---------|
| `horizon` | 30 | MPPI horizon N (lookahead = N·`mppi_dt`) |
| `num_rollouts` | 256 | MPPI samples K |
| `mppi_iters` | 1 | MPPI refinement iterations |
| `mppi_dt` | 0.1 | rollout timestep [s] |
| `plan_rate` | 10.0 | planning/publish rate [Hz] |
| `ctg_period` | 0.5 | geodesic cost-to-go refresh interval [s] |
| `robot_radius` | 0.31 | body radius for collision [m] |
| `robot_clear_radius` | 0.30 | clear robot footprint in the internal map [m] |
| `v_cap` | 0.8 | soft cruise speed target [m/s] |
| `a_max` | 0.7 | planar accel authority [m/s²] |
| `z_hold` | 1.0 | Path altitude for 3D viz [m] |
| `use_default_goal` / `goal_x` / `goal_y` | true / 3.0 / 0.0 | plan immediately to this point |

## Performance notes

- Plan tick ~40 ms → publishing at the 10 Hz `plan_rate` cap.
- Speedups already applied: cost-to-go throttled to `ctg_period`, EDT via
  `cv2.distanceTransform`, `mppi_iters=1`, `num_rollouts=256`.
- More speed: lower `horizon` (shortens lookahead), coarsen grid resolution
  (5→10 cm), or raise `plan_rate`.

## Not yet done / caveats

- **No MAVROS control output** — visualization only so far.
- **Frame consistency:** pose is taken from `/zed/zed_node/pose` (frame `map`) to
  match the grid. MAVROS `local_position` is in a different ENU frame — resolve
  before closing the control loop.
