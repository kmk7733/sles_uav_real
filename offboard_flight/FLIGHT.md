# Flight checklist

The order to run things in for an MPPI flight, and what each step has to say
before you go on to the next. Nothing arms until step 7.

Everything below lives in `offboard_flight/scripts/` unless the path says
otherwise. `~/start_test_grid.sh` and `~/script_startup_flight.sh` are host
scripts and sit outside this repo.

```
SCRIPTS=~/catkin_ws/src/offboard_flight/scripts
```

---

## 1 · Vicon on, before anything else

Power the Vicon system and confirm **ROGX2 is visible in Tracker**.

This is first because `vicon_bridge` opens a blocking SDK connection: started
against a host that is switched off it sits in a TCP connect for over 107 s
without completing a single retry. If you get the order wrong, restart just
the bridge with `~/script_startup_flight.sh vicon`.

While you are in Tracker: **are the obstacles subjects?** If they are, step 8
records their ground truth automatically. If they are not, the bag will hold
the aircraft only and you will be measuring clearance against positions
written on paper.

## 2 · Bring up the stack

```bash
~/catkin_ws/src/perception/restart_stack.sh
```

roscore, MAVROS, ZED, Vicon, foxglove, the Andert mapper and a dry-run
planner. Takes about 50 s.

For the real run, without the visualisation that costs measurable time:

```bash
FOXGLOVE=0 VIZ=0 ~/catkin_ws/src/perception/restart_stack.sh
```

## 3 · Health check

```bash
~/start_test_grid.sh check
```

Wanted: **FCU connected `True`** and **Vicon streaming `yes`**. Stop here if
either is missing.

> If this says the FCU is down, check the timeout before you check the wiring.
> It asks with `rostopic echo -n1` and every invocation pays a node
> registration first; against `/mavros/state` at 1 Hz a short budget loses that
> race on a healthy link. It is 12 s now — measured, 3 s reported NO DATA on a
> link that answered at 12.

## 4 · The three numbers that decide whether you fly

```bash
rostopic hz /grid_map                                  # ~7 Hz
grep -a "world->FCU" /tmp/planner_dry.log | tail -1    # n >= 10, not NOT READY
rostopic echo -n3 /rogx2/planar_planner_node/status    # valid=, solve=
```

| | expected | if not |
|---|---|---|
| `/grid_map` | ~7 Hz | the mapper is behind; check `~/gridmap_output.log` |
| `world->FCU` | `t=[...] n>=10` | the planner will publish NO setpoints and the mission node will never arm. Both `/robot/pose_world` and `mavros/local_position/pose` must be live, with stamps inside 0.3 s of each other |
| `valid=` | 50–70% | below ~30% the map is mostly unknown; look at `~inflated` in Foxglove |
| `solve=` | p50 ~70 ms | see below |

`fly.sh` defaults to `_plan_rate:=10 _num_samples:=96 _use_geodesic:=false`,
measured at p50 70 ms / p95 134 with `FOXGLOVE=0`. **Two things about that:**

**It needs `FOXGLOVE=0`.** `foxglove_nodelet_manager` is 27% of a core, and the
same config measures p50 119 ms with it running. Watching live costs the plan
rate; pick one.

**It is not config.yaml's planner.** Lowering `num_samples` cannot reach 10 Hz —
with the geodesic on, K 192 → 128 made it *worse*, and with it off, K 96 → 64
changed nothing. The geodesic is the whole difference (104 ms vs 70 at the same
K=96), and it is what removes the local minima of `‖p − goal‖` when the goal
sits behind an obstacle. To fly the validated planner instead:

```bash
PLANNER_ARGS="_plan_rate:=5" $SCRIPTS/fly.sh    # K=192, geodesic on
```

That measures p50 119 / p95 154 against a 200 ms budget — more margin than
anything in the 10 Hz column. Rate is not the thing to protect; replanning
*distance* is, and at v_max 0.31 m/s, 5 Hz is 6.2 cm per cycle against a
3.0 s / 0.93 m horizon.

## 5 · Planner live, mission node up, recording on

```bash
$SCRIPTS/fly.sh
```

Runs the pre-flight, replaces the dry-run planner with a live one, starts the
mission node, **starts the bag**, and then stops. It refuses to go on if any of
the four pre-flight topics is missing.

```bash
RECORD=depth $SCRIPTS/fly.sh     # + compressedDepth, to re-run the mapper offline
RECORD=0     $SCRIPTS/fly.sh     # no bag
```

Wait for this line before continuing:

```
[mission] READY -- rosservice call /rogx2/mission_node/start
```

## 6 · RC in hand

- Transmitter on, **Position mode**, kill switch located.
- Foxglove: `/grid_map`, `nominal_path`, `rollouts`, goal marker, the 0.31 m
  footprint circle.
- Confirm the goal marker is where you expect: **(2.0, 0.0)**.

## 7 · Fly

```bash
$SCRIPTS/fly.sh go
```

Three-second countdown, then `~start`. From here it is automatic:

| state | what it does | leaves when |
|---|---|---|
| `STREAM` | streams the current position, requests OFFBOARD then ARM | `/mavros/state` reports armed **and** OFFBOARD |
| `CLIMB` | ramps to the planner's altitude at 0.3 m/s | within 0.15 m for 3 s, planner alive |
| `MISSION` | forwards the planner's setpoints | `/goal_arrive_tf` true for 1 s |
| `LAND` | descends at 0.4 m/s, x/y frozen | touchdown confirmed from `extended_state` |
| `DISARM` | requests disarm at 2 Hz | `/mavros/state` reports disarmed |

## 8 · Watching, and getting out

```bash
$SCRIPTS/fly.sh state      # mission state, mav mode, arrived flag, bag path
$SCRIPTS/fly.sh land       # land now, from wherever it is
$SCRIPTS/fly.sh stop       # close the bag, stop both nodes
```

**The RC always wins.** Switching out of OFFBOARD puts the mission node in
`PILOT`: it stops publishing and does not resume on its own, even if OFFBOARD
comes back. To fly again, go back to step 5.

Failure sinks, none of which land the aircraft by themselves — they freeze it
and hand you the decision:

| `HOLD` because | why it stopped |
|---|---|
| planner silent > 0.5 s | no setpoints to forward |
| non-finite setpoint | refused, never passed to PX4 |
| geofence | > 6 m from where it armed |
| `mission_timeout` | 120 s without reaching the goal |
| landing did not confirm | no touchdown after the nominal descent + 10 s. **It does not disarm.** Take it with the RC |

Logs: `/tmp/planner_live.log`, `/tmp/mission.log`, `/tmp/record.log`.

---

## After the flight

```bash
ls -lh ~/bags/                         # flight_<date>_<profile>.bag
$SCRIPTS/analyze_flight.py ~/bags/flight_XXXX.bag --geometry ~/obstacles.yaml
```

`analyze_flight.py` prints what flew (from `~config`), the state timeline with
durations, the ground-truth trajectory, **clearance to every Vicon obstacle**,
and the planner's solve/valid statistics.

The clearance block is the one that has never been measured on hardware.
`r_safe = 0.51 m` is what the whole safety argument rests on, and until now it
had only ever been checked against the occupancy grid — which is the planner's
*belief*. With the obstacles in Vicon it becomes a measurement.

**You do not need to measure the obstacles.** `/vicon/markers` is recorded, the
markers are stuck to the corners, so their convex hull *is* the footprint. The
tool derives it and prints what it found:

```
pillar1    at ( -0.71,   0.20)  min surface 0.482 m  at t+10.2 s
             footprint from markers: 4 corners, sides 0.164/0.154/0.155/0.156 m, 0.236 m across
```

Those sides against a nominal 6 in = 0.1524 m. Marker centres sit ~2 mm
outboard of the face they are stuck to, so the hull is very slightly *inside*
the true surface — conservative, which is the right direction for a clearance
number.

`--geometry` still overrides, but prefer not to. A `box` there is axis-aligned
in the *subject's* frame, and an object need not be square to its own frame:
pillar1's 6 in square measures **27.6° rotated** inside its subject frame,
where an axis-aligned box would have to be 0.226 × 0.202 m to contain a
0.152 m pillar. The marker hull has no such problem.

How to read the verdict:

| worst surface clearance | means |
|---|---|
| `>= r_safe` (0.51) | as designed |
| between `r_quad` (0.31) and `r_safe` | the airframe was clear, but the gate is on the belief map — so **the grid did not see the obstacle where Vicon says it is**. A perception result, not a planner one |
| `< r_quad` | the airframe disc intersected a real obstacle |

What is in it and why:

| group | topics |
|---|---|
| ground truth | `/vicon/.*` — **every** subject: `ROGX2`, plus `pillar1`, `pillar2`, …, `wall1`, … The bridge has no subject filter, so anything in Tracker is recorded without editing a script. `/robot/pose_world` |
| perception | `/grid_map`, `~inflated`, `~inflated_outer` |
| decision | `~config`, `~status`, `~nominal_path`, `~rollouts`, `~goal_marker`, `/goal_arrive_tf` |
| execution | `commander/set_pose`, `setpoint_raw/local`, `setpoint_raw/target_local`, `local_position/pose`, `velocity_local`, `imu/data`, `battery` |
| state machine | `mission_node/state`, `mavros/state`, `mavros/extended_state`, `/rosout_agg` |

`~config` is the one to read first. It is latched JSON and says which producer
flew (`haa` / `hpa` / `desimplex`), its class, the full limits, the derived
sigma, every cost weight, the safety radii, the goal, and the git SHA of the
tree `planner/` was imported from — plus whether that tree was dirty:

```bash
python3 - <<'PY'
import json, rosbag
b = rosbag.Bag('/home/rogx/bags/flight_XXXX.bag')
for _, m, _ in b.read_messages('/rogx2/planar_planner_node/config'):
    print(json.dumps(json.loads(m.data), indent=2)); break
PY
```

`~inflated` is worth as much as `/grid_map`. The raw grid shows neither the
unknown-is-unsafe rule nor the r_safe growth, so a path that looks needlessly
timid against it is usually hugging the inflated set instead.

---

## OPEN: recording is not a passive observer

**Suspected, mechanism confirmed, effect not yet measured.** After the bag was
added, a flight showed the yaw oscillating left-right-left-right while
advancing. No weight or parameter changed across that commit — `PLANNER_ARGS`
was `_plan_rate:=5` before and after, and the only planner diff was a
read-only `~config` publisher. What changed is the LOAD.

`record_flight.sh` subscribes to three topics that the planner **skips
entirely when nobody is subscribed** (`planar_planner_node.py:682, 741, 793`,
`get_num_connections() == 0`):

    ~rollouts          a 30 x 31 point MarkerArray, rebuilt every solve
    ~inflated          two grid-wide distance_transform_edt every solve
    ~inflated_outer    another EDT

All three run **inside `plan_once`, on the same thread as the solve**. So
starting the bag makes the planner do three pieces of work per tick that it
was not doing before. A longer plan period means each plan's yaw trajectory is
followed for longer, and nothing damps yaw: `w_yaw = 0`, so yaw appears in no
cost term and is a random walk weighted only by the position cost.

`/vicon/markers` is the same shape — `vicon_bridge` streams markers only while
something subscribes, so recording turns that on too (5532 messages in the
first flight bag).

TO MEASURE, and it is one comparison: run the planner with `RECORD=0`, read
`solve=` off `~status`, then start `record_flight.sh` separately and read it
again. Same planner, only the recorder changes.

IF CONFIRMED, the fix is not to stop recording. Move the visualisation
publishing off the plan thread, or gate it on a rate rather than on a
subscriber, so that observing the planner does not change it.

## Not yet exercised

Honest list, so nothing here is a surprise in the air:

- **`/goal_arrive_tf` has never fired.** If it does not, `MISSION` ends in
  `HOLD` at `mission_timeout` — it will not descend. Land with `fly.sh land`.
- **`FOXGLOVE=0` has not been run end to end.**
- **`RC_MAP_KILL_SW` has not been confirmed on the ground.** Do this before
  the first arm of the day.
- **A bag has been closed cleanly exactly zero times.** The first flight bag
  came out `.active` because the recorder did not get its SIGINT. `rosbag
  reindex` recovered it whole, but `fly.sh stop` is what should be closing it
  and that path has not been shown to work.
- **`FOXGLOVE=0` is now measured** (it is worth 119 → 70 ms), but no flight has
  been flown with it off, so the Foxglove-blind procedure itself is untried.
- **No flight has used `_use_geodesic:=false`.** The 10 Hz default trades the
  planner's only defence against local minima for the rate.
