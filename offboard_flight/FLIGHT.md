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
~/catkin_ws/src/perception/restart_stack.sh          # no Foxglove
FOXGLOVE=1 ~/catkin_ws/src/perception/restart_stack.sh   # watch it live
```

roscore, MAVROS, ZED, Vicon, the Andert mapper and a dry-run planner. ~50 s.

**Foxglove is off by default and that is a measurement, not a preference.**
`foxglove_nodelet_manager` is 27.3% of a core, and the same planner measures
p50 119 ms with it running against 70 ms without — about 40% of the plan
budget to watch. It also decides what the planner computes: `~rollouts`,
`~inflated` and `~inflated_outer` skip their work whenever nobody subscribes,
so with no bridge they cost nothing whatever their parameters say.

`VIZ` is a different switch and a much smaller one — it only controls the
mapper's FOV wedge, which gates on subscribers like everything else, so with
the bridge off `VIZ=0` and `VIZ=1` do the same thing. Left at 1 so that
`FOXGLOVE=1` shows the whole picture.

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

## 4 · The numbers that decide whether you fly

```bash
rostopic hz /grid_map                                  # ~7-10 Hz
grep -a "world->FCU" /tmp/planner_dry.log | tail -1    # n >= 10, not NOT READY
rostopic echo -n3 /rogx2/planar_planner_node/status    # valid=, solve=, viz=
```

| | expected | if not |
|---|---|---|
| `/grid_map` | ~7–10 Hz | the mapper is behind; check `~/gridmap_output.log` |
| `world->FCU` | `t=[...] n>=10` | the planner publishes NO setpoints and the mission node never arms. Both `/robot/pose_world` and `mavros/local_position/pose` must be live, with stamps inside 0.3 s of each other |
| `valid=` | 50–70% | below ~30% the map is mostly unknown. Not visible live any more — `analyze_flight.py` rebuilds the inflated set from the bag, or set `_publish_inflated:=true` and pay 8.5 ms a tick |
| `solve=` | p50 ~120 ms | against a 200 ms budget at `plan_rate` 5 |
| `viz=` | **0 ms** | anything else means something subscribed to `~rollouts` or `~inflated` and the planner is doing work for it, inside the solve's own thread. Usually a Foxglove panel or a bag |

### What is flying, and why the rate is 5

`fly.sh` passes only `_plan_rate:=5`. Everything else is the node's default,
and the node's defaults are config.yaml's: **K=192, horizon 30, geodesic on,
R_dnu (1, 1, 0.2), d_influence 0.60, w_frontier 5.0, v_max 0.31**. That is the
planner every simulator result was produced with, and only the rate is
conceded to the hardware.

Lowering `num_samples` does not buy a faster tick. Measured on the node with
`FOXGLOVE=0`: with the geodesic on, K 192 → 128 made it *worse* (119 → 131 ms);
with it off, K 96 → 64 changed nothing (70 → 70). The geodesic is the whole
difference — 104 ms against 70 at the same K=96 — because it is a grid-wide
Dijkstra rebuilt once per solve that does not care how many samples were drawn.

So 10 Hz is available only by deleting the geodesic, which is what removes the
local minima of `‖p − goal‖` when the goal sits behind an obstacle. Rate is the
cheaper thing to give up: replanning *distance* is what matters, and at
v_max 0.31 m/s, 5 Hz is 6.2 cm per cycle against a 3.0 s / 0.93 m horizon.

If you want it anyway:

```bash
PLANNER_ARGS="_plan_rate:=10 _num_samples:=96 _use_geodesic:=false" $SCRIPTS/fly.sh
```

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
- Confirm the goal: `rostopic echo -n1 /rogx2/planar_planner_node/config`
  should say `"goal": [2.0, 0.0]`.
- **There is nothing to watch unless you started with `FOXGLOVE=1`**, and if
  you did, `viz=` in the status line will no longer be 0 and the plan rate is
  paying for it. On a measurement run, fly blind and read the bag afterwards.

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
| perception | `/grid_map` — `~inflated` is NOT recorded; `analyze_flight.py` rebuilds it from this plus `r_safe` |
| decision | `~config`, `~status`, `~nominal_path`, `~goal_marker`, `/goal_arrive_tf` |
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

The raw grid shows neither the unknown-is-unsafe rule nor the `r_safe` growth,
so a path that looks needlessly timid against it is usually hugging the
inflated set instead. `analyze_flight.py` prints that set — rebuilt, not
recorded — under **WHAT THE VALIDATOR GATED ON**.

---

## Observing the planner used to change it

Kept because the shape of the bug is worth recognising again. After the bag
was added, a flight showed the yaw oscillating left-right-left-right while
advancing. No weight or parameter had changed across that commit — the only
planner diff was a read-only `~config` publisher. What changed was the LOAD.

`~rollouts`, `~inflated` and `~inflated_outer` **skip their work entirely when
nobody is subscribed** (`planar_planner_node.py:682, 741, 793`), and all three
run **inside `plan_once`, on the solve's own thread**. `record_flight.sh` was
subscribing to them, so starting the bag switched on 8.5 ms (p95 22.8) of
distance transforms and 2.6 ms of marker building per tick. The recorder was
not an observer; it was a load.

Fixed three ways, so it cannot come back quietly:

- `~inflated` and `~rollouts` default **off**. Nothing is lost: the inflated
  set is a pure function of `/grid_map` and `r_safe`, both in the bag, and
  `analyze_flight.py` rebuilds it exactly.
- The recorder no longer subscribes to any of the three.
- The status line carries `viz=Nms` beside `solve=Nms`, and the node warns
  when visualisation exceeds 15% of a tick. **`viz=` should read 0.**

Still open: nobody has measured what that load did to the *yaw* specifically.
11 ms on a ~120 ms tick is ~9%, which is real but may not be the whole story —
`w_yaw = 0`, so yaw appears in no cost term at all and is a random walk
weighted only by the position cost. If the oscillation survives with `viz=0`,
that is where to look next.

## Not yet exercised

Honest list, so nothing here is a surprise in the air:

- **`/goal_arrive_tf` has never fired.** If it does not, `MISSION` ends in
  `HOLD` at `mission_timeout` — it will **not** descend. Land with
  `fly.sh land`.
- **`RC_MAP_KILL_SW` has not been confirmed on the ground.** Do this before
  the first arm of the day.
- **No bag has ever been closed cleanly.** The first flight bag came out
  `.active` because the recorder did not get its SIGINT; `rosbag reindex`
  recovered it whole, but `fly.sh stop` is what should be closing it and that
  path is unproven. Check for `*.active` in `~/bags/` after every flight.
- **No flight has been flown with Foxglove off.** The saving is measured
  (119 → 70 ms) but flying blind, with the bag as the only record, is a
  different procedure from the ones flown so far.
- **`_use_geodesic:=false` has never flown.** It is opt-in now, not the
  default, but if you reach for the 10 Hz line above, that is what you are
  flying.
- **The yaw oscillation is not explained.** See the section above: the
  recording load is fixed and `viz=` now makes it visible, but whether that
  was the cause is unmeasured, and `w_yaw = 0` remains a candidate.
