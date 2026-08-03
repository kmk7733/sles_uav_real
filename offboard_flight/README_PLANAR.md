# Fixed-altitude planar MPPI planner

A replacement for the 12-state CTBR MPPI in `haa_mppi.py`. Plans in a 6-state
planar model at constant altitude, validates every plan against a swept-path
safety check, and emits a controller-independent reference.

Every number in this document was measured on this vehicle's Jetson Xavier NX
against the live `/grid_map`, not taken from a datasheet or an idle benchmark.
Where something is unverified it says so.

---

## 1. Architecture

```
2D map ──> planar MPPI ──> validated planar trajectory ──> fixed-altitude 3D
                                                            reference
                                                              │
                                                    existing low-level
                                                    controller (PX4)
```

The planner core imports **no ROS, no PX4, no simulator types**. Only
`planar_planner_node.py` does. Everything else runs and is tested with plain
`python3`.

| module | role | ROS? |
|---|---|---|
| `planar_types.py` | `PlanarState`, `PlanarReferenceSequence`, `FixedAltitudeReference`, `MPPIResult`, `PlannerStatus` | no |
| `planar_dynamics.py` | 6-state model, hard limits, input clipping, braking | no |
| `planar_map.py` | occupancy-grid adapter, EDT clearance | no |
| `planar_safety.py` | `r_safe`, swept-path validator | no |
| `planar_mppi.py` | solver + safety chain | no |
| `planar_scenarios.py` | test/demo maps at the real grid geometry | no |
| `planar_planner_node.py` | ROS adapter, frame conversion, publishing | **yes** |
| `test_planar.py` | 9-section offline test suite | no |
| `demo_planar.py` | demonstration + live grid capture/replay | optional |

Additive changes to existing files (no existing function modified):

* `guidance_library.py` — `construct_target_full()`, a `PositionTarget` carrying
  position **and** velocity **and** acceleration **and** yaw **and** yaw-rate.
  The original `construct_target()` masks everything but position and yaw,
  which throws away most of what a trajectory planner produces.
* `haa_frames.py` — `rotate_to_fcu()`, for free vectors. Velocities and
  accelerations must be rotated but **not** translated; passing them through
  `to_fcu()` would add the frame offset and turn a 1 m/s reference into
  nonsense.

## 2. Model

State and input, both world-frame:

```
xi = [x, y, vx, vy, psi, omega]        nu = [ax, ay, alpha]

p_{k+1}     = p_k + dt v_k + 0.5 dt^2 a_k
v_{k+1}     = v_k + dt a_k
psi_{k+1}   = wrap(psi_k + dt omega_k + 0.5 dt^2 alpha_k)
omega_{k+1} = omega_k + dt alpha_k
```

**Roll and pitch are not planning variables.** They are the low-level
controller's response to the commanded acceleration, related by
`a_xy = g tan(theta)`. Planning in acceleration and bounding
`|a| <= g tan(theta_max)` is the same constraint one level up, and it removes
the attitude random-walk that made moment-sampled rollouts useless (only 0.9%
of samples survived a 4 s horizon — see `haa_dynamics.py`).

The half-`dt^2` terms make the integration **exact** for piecewise-constant
acceleration. Explicit Euler would lag position by `0.5 dt^2 a` per step, about
0.2 m over a 20-step horizon at 2 m/s^2 — comparable to the safety radius the
whole planner is built around. A useful consequence: raising `dt` costs no
integration accuracy at all.

## 3. Constraints, not cost terms

Hard limits, enforced by clipping (inputs) or rejection (states):

| limit | default | enforced by |
|---|---|---|
| `\|v\|_2 <= v_max` | 1.5 m/s | rejection (v is a state) |
| `\|omega\| <= omega_max` | 1.5 rad/s | rejection (state) |
| `\|a\|_2 <= min(a_max, g tan theta_max)` | 2.5 m/s² (tilt bound 5.66) | clipping |
| `\|alpha\| <= alpha_max` | 3.0 rad/s² | clipping |
| `\|a_k - a_{k-1}\|_2 <= j_max dt` | 8.0 m/s³ | clipping |

Clipping order is not arbitrary: the acceleration disc is applied **before** the
slew projection. The slew step projects onto a disc centred on the already
projected `a_{k-1}`, which lies inside the acceleration disc, so the result
stays on a segment between two points of that disc and cannot violate it. The
other order would break the slew bound.

`a_prev` — the acceleration currently being flown — anchors the first slew
constraint. Without it MPPI is free to demand a step change on the very node the
controller executes next.

## 4. Cost

```
J = w_goal   * sum_k |p_k - p_goal|
  + w_term_pos * |p_N - p_goal|          terminal position
  + w_term_vel * |v_N|^2                 terminal velocity: arrive stopped
  + w_obs    * sum_k max(0, d_infl - clearance_k)^2
  + sum_k |nu_k - nu_{k-1}|^2_{R_dnu}
  + w_yaw    * sum_k wrap(psi_k - psi_des_k)^2      (optional, default off)
```

Deliberately few terms. Hard limits live in the constraint set and safety lives
in the validator, so the cost only expresses preference. Piling on overlapping
penalties is what makes an MPPI planner untunable.

**Temperature is dimensionless.** `cost_normalise` divides `(S - S_min)` by the
batch standard deviation before the softmax. Without it the temperature is
absolute, and against an O(100) cost range a small lambda collapses
`exp(-dS/lambda)` onto the single best sample — the "weighted average" silently
becomes an argmin over a fresh random batch every tick, which re-decides
left-vs-right obstacle avoidance at 10 Hz. That failure looks like a working
planner that dithers. This is the specific defect that made the previous
`haa_mppi.py` (lambda = 0.02 against `w_terminal = 100`) unreliable.

## 5. Safety

```
r_safe = r_Q + r_perc + r_track + d_clr = 0.31 + 0.18 + 0.05 + 0.05 = 0.59 m
```

`r_perc = 0.18 m` is measured: 41 pillar samples over 0.66–3.17 m, radial
\|error\| p95 0.157, p99 0.182. `r_track` is the robust invariant tube the
low-level controller induces — because the nominal trajectory clears the
inflated set, the true trajectory clears the real obstacles.

**Occupied, unknown and out-of-map are all unsafe.** Unknown space is not free
space: the stereo pair has not seen it.

### The update is not assumed safe

Standard MPPI takes the weighted mean of the samples. That step is where a
sampling planner loses its safety argument: **the collision-free set is not
convex**, so the mean of a hundred collision-free sequences can pass straight
through the obstacle they all avoided — half dodge left, half dodge right, and
the average splits the difference through the middle.

Everything else survives averaging. `X` is affine in `U`, so the velocity,
heading-rate, acceleration and slew constraints are all convex in `U` and are
preserved exactly by a convex combination. Collision is the sole nonconvex
constraint and therefore the only thing the update can break.

So the update is a **candidate**, and this chain decides what flies:

| # | step | status |
|---|---|---|
| 1 | roll out the weighted update, validate the full swept path | `WEIGHTED` |
| 2 | if invalid, backtrack `U_old + beta dU`, `beta = 1, 1/2, 1/4...` | `DAMPED` |
| 3 | if no beta works, lowest-cost sample that passes swept validation | `BEST_SAMPLE` |
| 4 | if no sample survives, continue the previously validated plan | `PREVIOUS` |
| 5 | if there is no previous plan, brake to hover and validate that | `BRAKING` |
| 6 | if even braking is unsafe | `FAILED` |

`beta -> 0` recovers the previous nominal, which was validated last cycle, so
the backtracking searches a segment with a known-good end.

### Swept, not nodal

At `dt = 0.1` and `v_max = 1.5` a node spacing is up to 0.15 m — three cells. A
trajectory can put both endpoints of a segment in free space with the segment
passing straight through a 0.10 m obstacle. Sampled rollouts are rejected on
nodes only (cheap, hundreds of them); **the accepted trajectory always gets the
full swept check** at `sweep_step = res/2`.

The validator shares nothing with the obstacle cost. A validator that reused the
cost's notion of "near an obstacle" would inherit the cost's tuning, and
re-tuning `w_obs` for better goal-seeking would silently move the safety
boundary.

## 6. Running

```bash
python3 test_planar.py                    # 9 sections, ~41 s, all pass
python3 demo_planar.py --scenario slalom  # ASCII sim + 3D reference dump
python3 demo_planar.py --list             # open pillar wall gap narrow slalom corridor partial
python3 demo_planar.py --capture live.npz # snapshot the real /grid_map (needs ROS)
python3 demo_planar.py --grid live.npz    # replay it offline

# ROS, visualisation only -- publishes NO setpoints
ROS_NAMESPACE=rogx2 python3 planar_planner_node.py _dry_run:=true
```

Published topics (all under `~`, i.e. `/rogx2/planar_planner_node/`):

| topic | type | shows |
|---|---|---|
| `nominal_path` | `nav_msgs/Path` | the chosen plan |
| `rollouts` | `MarkerArray` | what MPPI explored |
| `inflated` | `OccupancyGrid` | the validator's real unsafe set |
| `goal_marker` | `Marker` | goal + tolerance |
| `status` | `String` | status, valid count, beta, cost, solve time |

`inflated` is the important one for debugging. `/grid_map` alone shows neither
the unknown-is-unsafe rule nor the 0.59 m inflation, so a path that looks
needlessly timid against the raw grid is usually hugging this instead.

## 7. Measured performance (Jetson Xavier NX, MODE_15W_6CORE)

Idle:

| K | N=20 solve | rate |
|---|---|---|
| 64 | 15 ms | 65 Hz |
| 128 | 16 ms | 60 Hz |
| 256 | 26 ms | 39 Hz |
| 512 | 46 ms | 22 Hz |
| 1024 | 69 ms | 14 Hz |

With the full perception stack running (ZED + `depth_to_grid` + `vicon_bridge`
+ `foxglove_bridge`, loadavg ~13 on 6 cores) the same solve takes roughly four
times as long, with a heavy tail:

| K | N | p50 | p95 | over 100 ms |
|---|---|---|---|---|
| 128 | 15 | 31–40 ms | 91–218 ms | 4–16% |
| 128 | 20 | 50–75 ms | 118–240 ms | ~15% |

The K-sweep comes out non-monotonic under load (K=32 measuring slower than
K=128), which is proof that **CPU contention, not sample count, is the limit**.
Sizing off the idle number is how a planner that "runs at 39 Hz on the bench"
misses every deadline in flight.

Missed deadlines degrade gracefully: `~plan_timeout` (0.5 s) keeps flying the
last plan while the 20 Hz publisher samples it by elapsed time, so a late solve
does not create a gap in the setpoint stream.

### Horizon vs surviving samples

The binding constraint on `N` is not solve time, it is how many samples survive:

| N | dt | horizon | valid % |
|---|---|---|---|
| 15 | 0.10 | 1.5 s | 57% |
| **20** | **0.10** | **2.0 s** | **36%** ← default |
| 25 | 0.10 | 2.5 s | 23% |
| 30 | 0.10 | 3.0 s | 14% |
| 20 | 0.15 | 3.0 s | 7% |
| 20 | 0.20 | 4.0 s | 1% |

A longer rollout has more chances to touch the unsafe set. Raising `dt` is
nearly free in CPU and costs no integration accuracy, but it destroys the valid
fraction fastest, so it is the wrong lever.

### Speed

| | |
|---|---|
| configured `v_max` | 1.5 m/s |
| achieved, open arena | 1.30 m/s (87%) |
| achieved, pillar | 1.20 m/s (80%) |
| achieved, slalom | 1.16 m/s (77%) |

The gap is `w_term_vel = 2.0` making every plan arrive stopped: with a 2 s
horizon and a goal within ~3 m the planner never commits to full speed. Lower
`~w_term_vel` or extend the horizon to close it.

## 8. Known limitations

**Local minima.** The goal term is Euclidean distance, as specified. The slalom
scenario is cleared in ~26 s where the open arena takes 6 s — it works, but is
slow through tight geometry, and two of three seeds needed ~30 s.
`mpc_controller/src/mpc_controller/cost_to_go.py` already implements the
geodesic field that fixes this; wiring it in as the goal term is the natural
next step.

**Unknown space dominates.** On the live grid, `/grid_map` is 70% unknown. After
`unknown -> unsafe` and inflation by `r_safe = 0.59 m`, **99.2% of the arena is
unflyable** and the planner correctly refuses to move more than ~0.3 m. This is
not a planner defect, it is the honest consequence of a 0.31 m quadrotor with
0.18 m of depth uncertainty in a barely-observed map. Levers, most trustworthy
first:

1. Observe more map before translating — yaw in place; free space grows fast.
2. Raise `depth_to_grid`'s `~near_free_radius` (currently 1.0 m), the parameter
   intended for exactly this.
3. Set `~d_clr` to 0 — buys 0.05 m, safe.
4. Reduce `r_perc` — only with new measurements; 0.18 came from a real p99.

**Not yet flown.** Everything here is verified offline and against the live grid
in `dry_run`. No setpoint has been published to the vehicle.

## 9. Key parameters

| param | default | note |
|---|---|---|
| `~num_samples` | 128 | measured under live load, not idle |
| `~horizon` | 20 | 2.0 s; see the valid-% table |
| `~dt` | 0.1 | |
| `~temperature` | 1.0 | dimensionless (costs are std-normalised) |
| `~v_max` / `~a_max` / `~j_max` | 1.5 / 2.5 / 8.0 | |
| `~tilt_max` | 0.5236 | 30°, bounds `a` via `g tan` |
| `~r_quad` / `~r_perc` / `~r_track` / `~d_clr` | 0.31 / 0.18 / 0.05 / 0.05 | sums to `r_safe` |
| `~w_goal` / `~w_term_pos` / `~w_term_vel` / `~w_obs` | 1 / 10 / 2 / 20 | |
| `~w_yaw` | 0.0 | off; couples heading to translation |
| `~unknown_unsafe` | true | do not turn this off casually |
| `~dry_run` | false | true = plan and visualise, publish nothing |
