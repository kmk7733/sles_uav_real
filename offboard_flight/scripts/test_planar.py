#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Offline checks for the planar MPPI planner. No ROS, no vehicle, no pytest.

    python3 test_planar.py

Same shape as test_haa.py: every check prints PASS/FAIL and the script exits
non-zero if any failed.

The interesting tests are sections 5-7. Sections 5 and 7 drive the safety chain
with stub validators rather than real maps, because the point is to prove that
each branch of the chain is reachable and does what it claims -- arranging a
real occupancy grid that forces "the weighted update is unsafe but sample 47 is
fine" is possible but fragile, and a fragile test of a safety mechanism is worse
than none. Section 8 then runs the whole thing against real grids and asserts
the invariant that actually matters: whatever comes out passes an INDEPENDENT
swept-path check.
"""

import time

import numpy as np

from planar_types import (PlanarState, PlanarReferenceSequence,
                          FixedAltitudeReference, PlannerStatus,
                          wrap_angle, S_POS)
from planar_dynamics import PlanarDynamics, PlanarLimits, G
from planar_map import PlanarOccupancy, FreeSpace
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarMPPI, PlanarCostWeights, build_planner
import planar_scenarios as sc

fail = []


def check(name, cond, detail=""):
    print("  %-52s %s %s" % (name, "PASS" if bool(cond) else "FAIL", detail))
    if not bool(cond):
        fail.append(name)


def section(title):
    print("\n=== %s ===" % title)


# Maps come from planar_scenarios so the demo and the tests exercise exactly the
# same geometry -- 144 x 104 @ 0.05 m, matching the live /grid_map.

def fly(planner, start, goal, steps=100, r_safe=None, occ=None):
    """Close the loop: replan, step one node along the plan, repeat.

    Returns (trajectory (n,2), status counts, violation count). The violation
    count is checked against an INDEPENDENT validator, never the planner's own.
    """
    audit = (PlanarSafetyValidator(occ, r_safe)
             if occ is not None and r_safe is not None else None)
    s = start
    a_prev = np.zeros(2)
    traj = [[s.x, s.y]]
    counts = {}
    violations = 0
    for _ in range(steps):
        res = planner.plan(s, np.asarray(goal), a_prev=a_prev)
        counts[res.status] = counts.get(res.status, 0) + 1
        if res.reference is None:
            break
        if audit is not None and not audit.path_safe(res.reference.p):
            violations += 1
        pt = res.reference.node(1)
        s = PlanarState(pt.p[0], pt.p[1], pt.v[0], pt.v[1], pt.psi, pt.psi_dot)
        a_prev = res.reference.a[0]
        traj.append([s.x, s.y])
    return np.array(traj), counts, violations


# ====================================================================== 1
section("1. planar dynamics and yaw wrapping")

dyn = PlanarDynamics(PlanarLimits(), dt=0.1)

# Constant acceleration integrates EXACTLY under this scheme: summing the
# per-step half-dt^2 terms telescopes to 0.5 T^2 a. That is the whole reason for
# using it instead of explicit Euler, so it is worth asserting to machine
# precision rather than to a tolerance.
xi0 = np.array([0.0, 0.0, 0.5, -0.2, 0.3, 0.4])
a = np.array([0.7, -0.3])
al = 0.25
N = 20
U = np.tile(np.array([a[0], a[1], al]), (N, 1))
X = dyn.rollout(xi0, U)[0]
T = N * dyn.dt

p_exact = xi0[0:2] + T * xi0[2:4] + 0.5 * T * T * a
v_exact = xi0[2:4] + T * a
check("position matches p0 + T v0 + 0.5 T^2 a exactly",
      np.abs(X[-1, 0:2] - p_exact).max() < 1e-12,
      "err=%.2e" % np.abs(X[-1, 0:2] - p_exact).max())
check("velocity matches v0 + T a exactly",
      np.abs(X[-1, 2:4] - v_exact).max() < 1e-12,
      "err=%.2e" % np.abs(X[-1, 2:4] - v_exact).max())

psi_exact = wrap_angle(xi0[4] + T * xi0[5] + 0.5 * T * T * al)
check("heading matches psi0 + T w0 + 0.5 T^2 alpha exactly",
      abs(wrap_angle(X[-1, 4] - psi_exact)) < 1e-12,
      "err=%.2e" % abs(wrap_angle(X[-1, 4] - psi_exact)))
check("heading rate matches w0 + T alpha exactly",
      abs(X[-1, 5] - (xi0[5] + T * al)) < 1e-12)

# yaw wrapping
check("wrap_angle(3pi) == pi", abs(abs(wrap_angle(3 * np.pi)) - np.pi) < 1e-12)
check("wrap_angle(-3pi) == pi", abs(abs(wrap_angle(-3 * np.pi)) - np.pi) < 1e-12)
check("wrap_angle(0.5) == 0.5", abs(wrap_angle(0.5) - 0.5) < 1e-15)

spin = np.zeros((60, 3))
spin[:, 2] = 0.0
xs = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 1.0])          # psi=3.0, omega=1 rad/s
Xw = dyn.rollout(xs, spin)[0]
check("psi stays wrapped to (-pi, pi] while spinning",
      np.abs(Xw[:, 4]).max() <= np.pi + 1e-12,
      "max |psi| = %.6f" % np.abs(Xw[:, 4]).max())
check("psi actually crossed the branch cut",
      Xw[:, 4].min() < -2.0 and Xw[:, 4].max() > 2.0)

# zero input is a fixed point in position only if velocity is zero
Xz = dyn.rollout(np.zeros(6), np.zeros((10, 3)))[0]
check("zero input from rest is a fixed point", np.abs(Xz).max() < 1e-15)


# ====================================================================== 2
section("2. constraints: velocity, acceleration, slew")

lim = PlanarLimits(v_max=1.5, a_max=2.5, omega_max=1.5, alpha_max=3.0,
                   tilt_max=np.radians(30.0), j_max=8.0)
d2 = PlanarDynamics(lim, dt=0.1)

check("a_max_eff = min(a_max, g tan tilt_max)",
      abs(lim.a_max_eff - min(2.5, G * np.tan(np.radians(30.0)))) < 1e-12,
      "%.4f (tilt bound %.4f)" % (lim.a_max_eff, lim.a_max_tilt))

tilt_bound = PlanarLimits(a_max=9.0, tilt_max=np.radians(10.0))
check("tilt bound binds when a_max is large",
      abs(tilt_bound.a_max_eff - G * np.tan(np.radians(10.0))) < 1e-12,
      "%.4f m/s^2" % tilt_bound.a_max_eff)

rng = np.random.RandomState(0)
Uw = rng.randn(64, 25, 3) * np.array([20.0, 20.0, 30.0])   # wildly out of bounds
a_prev = np.array([0.4, -0.6])
Uc = d2.clip_inputs(Uw, a_prev=a_prev)

an = np.linalg.norm(Uc[..., 0:2], axis=-1)
check("clipping enforces |a| <= a_max_eff",
      an.max() <= lim.a_max_eff + 1e-9, "max |a| = %.6f" % an.max())
check("clipping enforces |alpha| <= alpha_max",
      np.abs(Uc[..., 2]).max() <= lim.alpha_max + 1e-9,
      "max |alpha| = %.6f" % np.abs(Uc[..., 2]).max())

chain = np.concatenate([np.broadcast_to(a_prev, (64, 1, 2)), Uc[..., 0:2]],
                       axis=1)
slew = np.linalg.norm(np.diff(chain, axis=1), axis=-1)
check("clipping enforces slew <= j_max dt (incl. vs a_prev)",
      slew.max() <= lim.j_max * d2.dt + 1e-9,
      "max slew = %.6f, limit %.6f" % (slew.max(), lim.j_max * d2.dt))
check("slew projection did not undo the acceleration disc",
      an.max() <= lim.a_max_eff + 1e-9)
check("inputs_ok accepts its own clipped output",
      d2.inputs_ok(Uc, a_prev=a_prev).all())
check("inputs_ok rejects the unclipped input", not d2.inputs_ok(Uw).any())

# velocity is a STATE: cannot be clipped, must be rejected
Ufast = np.zeros((1, 30, 3))
Ufast[:, :, 0] = lim.a_max_eff
Xfast = d2.rollout(np.zeros(6), Ufast)
sp = np.linalg.norm(Xfast[0, :, 2:4], axis=-1).max()
check("sustained max accel breaks v_max", sp > lim.v_max, "peak %.3f m/s" % sp)
check("states_ok rejects the over-speed rollout", not d2.states_ok(Xfast)[0])

Uslow = np.zeros((1, 30, 3))
Uslow[:, :, 0] = 0.2
check("states_ok accepts an in-limit rollout",
      d2.states_ok(d2.rollout(np.zeros(6), Uslow))[0])

Uspin = np.zeros((1, 30, 3))
Uspin[:, :, 2] = lim.alpha_max
check("states_ok rejects an over-rate heading rollout",
      not d2.states_ok(d2.rollout(np.zeros(6), Uspin))[0])


# ====================================================================== 3
section("3. map, obstacle and between-node collision detection")

occ = sc.pillar(px=1.5, py=0.0, radius=0.25)
print("  %s" % occ.describe())

check("clearance is 0 inside the pillar", occ.clearance(1.5, 0.0) < 1e-9)
c_edge = float(occ.clearance(2.2, 0.0))
check("clearance grows away from the pillar", 0.3 < c_edge < 0.55,
      "%.3f m at 0.7 m from centre (r=0.25)" % c_edge)
check("clearance is 0 outside the grid", float(occ.clearance(99.0, 99.0)) == 0.0)

allunknown = PlanarOccupancy.from_values(
    sc.blank(fill=sc.UNKNOWN), sc.RES, sc.ORIGIN, unknown_unsafe=True)
check("unknown counts as unsafe", float(allunknown.clearance(0.0, 0.0)) == 0.0)
allunknown.clear_disc(0.0, 0.0, 0.4)
check("clear_disc frees the footprint and drops the EDT cache",
      float(allunknown.clearance(0.0, 0.0)) > 0.3,
      "%.3f m" % float(allunknown.clearance(0.0, 0.0)))

# --- the between-node test: both endpoints clear, segment crosses a wall
thin_wall = sc.wall(x=1.0, thickness=0.05)
v_thin = PlanarSafetyValidator(thin_wall, r_safe=0.01, sweep_step=0.02)
seg = np.array([[0.8, 0.0], [1.2, 0.0]])
check("both nodes of the straddling segment are clear",
      bool((v_thin.clearance(seg) >= v_thin.r_safe).all()),
      "clearances %s" % np.round(v_thin.clearance(seg), 3).tolist())
check("nodes_safe MISSES the wall between the nodes",
      v_thin.nodes_safe(seg[None, :, :])[0])
check("paths_safe CATCHES the wall between the nodes",
      not v_thin.path_safe(seg))
idx, mn = v_thin.first_violation(seg)
check("first_violation locates the bad segment", idx == 0,
      "segment %d, min clearance %.3f m" % (idx, mn))

clear_seg = np.array([[0.2, 0.0], [0.7, 0.0]])
check("paths_safe accepts a genuinely clear segment",
      v_thin.path_safe(clear_seg))

check("safe_radius = r_Q + r_perc + r_track + d_clr",
      abs(safe_radius(0.31, 0.18, 0.05, 0.05) - 0.59) < 1e-12,
      "%.3f m" % safe_radius(0.31, 0.18, 0.05, 0.05))


# ====================================================================== 4
section("4. warm start and deterministic sampling")

def make(seed=0, occ_=None, K=192, N=20, **kw):
    o = occ_ if occ_ is not None else sc.pillar(px=1.5, py=0.0, radius=0.25)
    return build_planner(o, r_safe=0.35, horizon=N, num_samples=K,
                         seed=seed, **kw)

p1 = make(seed=7)
p2 = make(seed=7)
p3 = make(seed=8)
st = PlanarState(-1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
goal = np.array([3.0, 0.0])

r1 = p1.plan(st, goal)
r2 = p2.plan(st, goal)
r3 = p3.plan(st, goal)
check("same seed gives bit-identical plans",
      r1.U is not None and r2.U is not None
      and np.array_equal(r1.U, r2.U))
check("different seed gives a different plan",
      r3.U is None or r1.U is None or not np.array_equal(r1.U, r3.U))

pw = make(seed=1)
pw.U_nom = np.arange(pw.N * 3, dtype=np.float64).reshape(pw.N, 3)
before = pw.U_nom.copy()
pw.warm_start(1)
check("warm_start shifts the nominal forward by one step",
      np.array_equal(pw.U_nom[:-1], before[1:]))
check("warm_start pads the tail with zero input (= hover)",
      np.abs(pw.U_nom[-1]).max() == 0.0)

pw.warm_start(pw.N + 5)
check("over-long warm_start resets rather than going ragged",
      pw.U_nom.shape == (pw.N, 3) and np.abs(pw.U_nom).max() == 0.0)

p4 = make(seed=3)
r = p4.plan(st, goal)
check("accepted sequence becomes the next nominal",
      r.U is not None and np.array_equal(p4.U_nom, r.U))

p5 = make(seed=4)
traj, counts, _ = fly(p5, PlanarState(-1.5, 0.0, 0.0, 0.0, 0.0, 0.0), goal,
                      steps=40)
check("40 consecutive warm-started solves all return a plan",
      sum(counts.values()) == 40 and PlannerStatus.FAILED not in counts,
      "statuses: %s" % counts)
d0 = np.linalg.norm(np.array([-1.5, 0.0]) - goal)
d_end = np.linalg.norm(traj[-1] - goal)
check("closed-loop replanning makes progress toward the goal",
      d_end < d0 - 1.0, "%.2f -> %.2f m over 4 s" % (d0, d_end))


# ====================================================================== 5
section("5. nonconvexity and the damped / best-sample fallbacks")

# 5a. Demonstrate the nonconvexity itself, with no MPPI involved: two input
# sequences that each go safely around a pillar, whose average does not.
occ_p = sc.pillar(px=1.2, py=0.0, radius=0.30)
val_p = PlanarSafetyValidator(occ_p, r_safe=0.20)
dyn_p = PlanarDynamics(PlanarLimits(v_max=3.0, a_max=4.0), dt=0.1)

Nn = 20
U_left = np.zeros((Nn, 3))
U_right = np.zeros((Nn, 3))
U_left[:, 0] = 1.2
U_right[:, 0] = 1.2
U_left[:8, 1] = 3.0
U_left[8:16, 1] = -3.0
U_right[:8, 1] = -3.0
U_right[8:16, 1] = 3.0

x_start = np.array([-0.6, 0.0, 0.6, 0.0, 0.0, 0.0])
X_l = dyn_p.rollout(x_start, U_left)[0]
X_r = dyn_p.rollout(x_start, U_right)[0]
X_m = dyn_p.rollout(x_start, 0.5 * (U_left + U_right))[0]

check("left detour is safe", val_p.validate_states(X_l),
      "max |y| = %.2f" % np.abs(X_l[:, 1]).max())
check("right detour is safe", val_p.validate_states(X_r),
      "max |y| = %.2f" % np.abs(X_r[:, 1]).max())
check("their AVERAGE is unsafe (collision-free set is nonconvex)",
      not val_p.validate_states(X_m),
      "average runs through y=%.2f at the pillar" % np.abs(X_m[:, 1]).max())


class StubValidator(object):
    """Validator whose swept verdict is a supplied predicate.

    Lets the chain be driven branch by branch without hunting for an occupancy
    grid that happens to trigger the branch under test.
    """

    def __init__(self, r_safe=0.3, nodes=True, swept=None, clear=5.0):
        self.r_safe = r_safe
        self._nodes = nodes
        self._swept = swept if swept is not None else (lambda P: True)
        self._clear = clear

    def clearance(self, P):
        return np.full(np.asarray(P).shape[:-1], self._clear, dtype=np.float64)

    def nodes_safe(self, P):
        P = np.asarray(P)
        out = np.full(P.shape[0], bool(self._nodes))
        return out

    def paths_safe(self, P):
        P = np.asarray(P)
        return np.array([bool(self._swept(p)) for p in P])

    def path_safe(self, P):
        return bool(self._swept(np.asarray(P)[:, :2]))

    def validate_states(self, X):
        return bool(self._swept(np.asarray(X)[:, S_POS]))

    def describe(self):
        return "StubValidator"


def stub_planner(validator, seed=0, K=192, N=20, **kw):
    return PlanarMPPI(PlanarDynamics(PlanarLimits(), 0.1), validator,
                      horizon=N, num_samples=K, seed=seed, **kw)


# 5b. Damped update: accept only trajectories that stay within `reach` of the
# start. With U_nom = 0 and v0 = 0, displacement scales exactly with beta, so
# the backtracking must land on a smaller beta rather than fail.
reach = 0.30
gate_reach = StubValidator(
    swept=lambda P: bool(np.linalg.norm(P - P[0], axis=-1).max() <= reach))
pd_ = stub_planner(gate_reach, seed=11)
rd = pd_.plan(PlanarState(0, 0, 0, 0, 0, 0), np.array([8.0, 0.0]))
check("reach-limited gate forces a damped update",
      rd.status == PlannerStatus.DAMPED, "status=%s beta=%.4g" % (rd.status, rd.beta))
check("damped beta is strictly less than 1", 0.0 < rd.beta < 1.0,
      "beta=%.5g" % rd.beta)
check("accepted damped trajectory satisfies the gate",
      rd.X is not None and np.linalg.norm(rd.X[:, S_POS] - rd.X[0, S_POS],
                                          axis=-1).max() <= reach + 1e-9)

# 5c. Best-sample: accept only trajectories with real lateral excursion. The
# nominal is straight and so is every damped blend of it, so beta backtracking
# cannot succeed at any beta -- only an individual sample can.
gate_lat = StubValidator(
    swept=lambda P: bool(np.abs(P[:, 1] - P[0, 1]).max() >= 0.25))
pb = stub_planner(gate_lat, seed=12, K=384)
rb = pb.plan(PlanarState(0, 0, 0, 0, 0, 0), np.array([5.0, 0.0]))
check("lateral-only gate forces the best-sample fallback",
      rb.status == PlannerStatus.BEST_SAMPLE, "status=%s" % rb.status)
check("best-sample trajectory satisfies the gate",
      rb.X is not None and np.abs(rb.X[:, 1] - rb.X[0, 1]).max() >= 0.25)
check("result reports itself as degraded", rb.degraded)


# ====================================================================== 6
section("6. fixed-altitude reference lifting")

pl = make(seed=21)
rl = pl.plan(PlanarState(-1.5, 0.0, 0.0, 0.0, 0.2, 0.0), goal)
ref = rl.reference
check("planner returns a PlanarReferenceSequence",
      isinstance(ref, PlanarReferenceSequence))
check("reference has N+1 nodes", ref.n_nodes == pl.N + 1,
      "%d nodes, horizon %d" % (ref.n_nodes, ref.horizon))
check("acceleration is padded to N+1 by holding the last input",
      np.array_equal(ref.a[-1], ref.a[-2]))
check("reference duration matches N dt",
      abs(ref.duration - pl.N * pl.dyn.dt) < 1e-12,
      "%.2f s" % ref.duration)

z0 = 1.0
lifted = ref.lift(z0)
check("lift returns a FixedAltitudeReference",
      isinstance(lifted, FixedAltitudeReference))
check("lifted position is 3D at constant z0",
      lifted.p.shape[1] == 3 and np.abs(lifted.p[:, 2] - z0).max() == 0.0)
check("lifted xy is unchanged", np.array_equal(lifted.p[:, :2], ref.p))
check("lifted vz is identically zero", np.abs(lifted.v[:, 2]).max() == 0.0)
check("lifted az is identically zero", np.abs(lifted.a[:, 2]).max() == 0.0)
check("lifted vxy / axy are unchanged",
      np.array_equal(lifted.v[:, :2], ref.v)
      and np.array_equal(lifted.a[:, :2], ref.a))
check("psi and psi_dot survive the lift",
      np.array_equal(lifted.psi, ref.psi)
      and np.array_equal(lifted.psi_dot, ref.psi_dot))

s0 = lifted.sample(0.0)
check("sample(0) returns node 0", np.abs(s0.p - lifted.p[0]).max() < 1e-12)
mid = lifted.sample(1.5 * lifted.dt)
expect = 0.5 * (lifted.p[1] + lifted.p[2])
check("sample interpolates linearly between nodes",
      np.abs(mid.p - expect).max() < 1e-12)
late = lifted.sample(1e6)
check("sample clamps past the horizon instead of extrapolating",
      np.abs(late.p - lifted.p[-1]).max() < 1e-12)
check("sampled point keeps the fixed altitude", abs(late.p[2] - z0) < 1e-12)

wrap_ref = PlanarReferenceSequence(
    p=np.zeros((2, 2)), v=np.zeros((2, 2)), a=np.zeros((2, 2)),
    psi=np.array([3.0, -3.0]), psi_dot=np.zeros(2), dt=0.1)
mid_psi = wrap_ref.sample(0.05).psi
check("psi interpolation takes the short way round the branch cut",
      abs(abs(mid_psi) - np.pi) < 0.15, "psi=%.4f" % mid_psi)


# ====================================================================== 7
section("7. previous-plan and braking fallbacks")

# Braking: no previous plan, every sample rejected at the node check, but the
# braking trajectory passes the swept check.
gate_brake = StubValidator(nodes=False, swept=lambda P: True)
pbrk = stub_planner(gate_brake, seed=31)
moving = PlanarState(0.0, 0.0, 1.2, 0.4, 0.0, 0.8)
rk = pbrk.plan(moving, np.array([5.0, 0.0]))
check("no valid sample and no previous plan -> BRAKING",
      rk.status == PlannerStatus.BRAKING, "status=%s" % rk.status)
check("braking trajectory ends stopped",
      rk.X is not None and np.linalg.norm(rk.X[-1, 2:4]) < 1e-6,
      "final speed %.2e m/s" % np.linalg.norm(rk.X[-1, 2:4]))
check("braking trajectory ends with zero heading rate",
      abs(rk.X[-1, 5]) < 1e-6, "final omega %.2e rad/s" % abs(rk.X[-1, 5]))
check("braking respects the acceleration and slew limits",
      pbrk.dyn.inputs_ok(rk.U))
check("braking respects the state limits", pbrk.dyn.states_ok(rk.X))

# Previous plan: solve once cleanly, then reject everything.
gate_prev = StubValidator(nodes=True, swept=lambda P: True)
pprev = stub_planner(gate_prev, seed=32)
r_first = pprev.plan(PlanarState(0, 0, 0.3, 0.0, 0, 0), np.array([4.0, 0.0]))
check("first solve succeeds and stores a plan", r_first.ok,
      "status=%s" % r_first.status)
ref_before = pprev._last_ref.p.copy()

gate_prev._nodes = False                     # no sample survives now
r_prev = pprev.plan(PlanarState(0, 0, 0.3, 0.0, 0, 0), np.array([4.0, 0.0]))
check("no valid sample but a stored plan -> PREVIOUS",
      r_prev.status == PlannerStatus.PREVIOUS, "status=%s" % r_prev.status)
check("PREVIOUS returns the stored plan shifted by one node",
      np.abs(r_prev.reference.p[:-1] - ref_before[1:]).max() < 1e-12)
check("PREVIOUS still carries a usable reference",
      r_prev.reference is not None and r_prev.ok)
check("PREVIOUS is flagged as degraded", r_prev.degraded)

# Everything unsafe, including braking -> FAILED.
gate_dead = StubValidator(nodes=False, swept=lambda P: False)
pdead = stub_planner(gate_dead, seed=33)
r_dead = pdead.plan(PlanarState(0, 0, 1.0, 0.0, 0, 0), np.array([4.0, 0.0]))
check("nothing valid anywhere -> FAILED",
      r_dead.status == PlannerStatus.FAILED, "status=%s" % r_dead.status)
check("FAILED carries no reference", r_dead.reference is None)
check("FAILED is not .ok", not r_dead.ok)
check("FAILED explains itself", len(r_dead.reason) > 0, r_dead.reason)


# ====================================================================== 8
section("8. end-to-end invariant on real grids")

# The invariant that matters: for ANY returned plan, an independently
# constructed validator must agree the swept path is clear. If this ever fails,
# the safety chain has a hole in it.
#
# These runs are expected to come back entirely WEIGHTED, and that is the
# HEALTHY result, not a gap in coverage: colliding samples are given zero weight
# and so never enter the average, which on maps like these is enough to keep the
# average safe on its own. The fallbacks are the backstop for when it is not,
# and section 5/7 prove each branch is reachable and correct. Asserting a
# fallback must fire here would be asserting the planner must nearly fail.

r_safe = 0.35
GOAL = np.array([2.5, 0.0])
START = PlanarState(-2.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Step budgets are per scenario because the planner is deliberately cautious in
# tight geometry: the slalom is cleared in ~22 s of flight where the open arena
# takes 5 s. That is a real property worth knowing (see the tuning notes in
# planar_mppi), not something to hide behind a loose tolerance.
BUDGET = {"open": 60, "pillar": 110, "gap": 110, "wall": 60, "slalom": 260}

runs = {}
total_plans = 0
total_violations = 0
all_status = {}
for name in ("open", "pillar", "gap", "wall", "slalom"):
    o = sc.SCENARIOS[name]()
    per_seed = []
    for seed in range(2):
        planner = build_planner(o, r_safe=r_safe, horizon=20, num_samples=192,
                                seed=seed)
        traj, counts, viol = fly(planner, START, GOAL, steps=BUDGET[name],
                                 r_safe=r_safe, occ=o)
        per_seed.append(traj)
        total_plans += sum(counts.values())
        total_violations += viol
        for k, n in counts.items():
            all_status[k] = all_status.get(k, 0) + n
    runs[name] = per_seed

print("  %d plans over %d scenarios x 2 seeds" % (total_plans, len(runs)))
for k in sorted(all_status):
    print("      %-12s %d" % (k, all_status[k]))

check("no returned plan ever fails an independent swept check",
      total_violations == 0, "%d violations" % total_violations)
check("no scenario ever ends in FAILED",
      PlannerStatus.FAILED not in all_status,
      "statuses: %s" % ", ".join(sorted(all_status)))


def reached(trajs, goal, tol=0.35):
    return all(np.linalg.norm(t[-1] - goal) < tol for t in trajs)


def max_lateral(trajs):
    return max(float(np.abs(t[:, 1]).max()) for t in trajs)


def min_x_gap(trajs, x):
    return min(float((x - t[:, 0]).min()) for t in trajs)


check("open arena: reaches the goal", reached(runs["open"], GOAL),
      "final gap %.2f m" % max(np.linalg.norm(t[-1] - GOAL)
                               for t in runs["open"]))
check("open arena: goes essentially straight",
      max_lateral(runs["open"]) < 0.35,
      "max |y| = %.2f m" % max_lateral(runs["open"]))

check("pillar: reaches the goal", reached(runs["pillar"], GOAL),
      "final gap %.2f m" % max(np.linalg.norm(t[-1] - GOAL)
                               for t in runs["pillar"]))
check("pillar: detours laterally around it rather than through it",
      max_lateral(runs["pillar"]) > 0.5,
      "max |y| = %.2f m (pillar r=0.30 at x=0.5)" % max_lateral(runs["pillar"]))

check("gap in wall: threads the gap and reaches the goal",
      reached(runs["gap"], GOAL),
      "max |y| = %.2f m (gap at y=0.8)" % max_lateral(runs["gap"]))

check("slalom: reaches the goal through staggered pillars",
      reached(runs["slalom"], GOAL, tol=0.5),
      "max |y| = %.2f m, needed %d steps (%.0f s)"
      % (max_lateral(runs["slalom"]), BUDGET["slalom"],
         BUDGET["slalom"] * 0.1))

# The solid wall has no way through. The correct behaviour is to stop short of
# it and stay stopped -- never to cross, and never to FAILED.
gap_to_wall = min_x_gap(runs["wall"], 0.5)
check("solid wall: never crosses it", gap_to_wall > 0.0,
      "closest approach %.2f m short of the wall" % gap_to_wall)
check("solid wall: keeps r_safe clear of it", gap_to_wall >= r_safe,
      "%.2f m vs r_safe %.2f m" % (gap_to_wall, r_safe))
check("solid wall: does not give up (no FAILED)",
      PlannerStatus.FAILED not in all_status)

# open space with no map at all
fp = build_planner(FreeSpace(), r_safe=0.35, horizon=20, num_samples=128,
                   seed=5)
rfp = fp.plan(PlanarState(0, 0, 0, 0, 0, 0), np.array([3.0, 0.0]))
check("FreeSpace lets the planner run unobstructed",
      rfp.status == PlannerStatus.WEIGHTED, "status=%s" % rfp.status)


# ====================================================================== 9
section("9. solve time on this machine")

o = sc.pillar(px=0.5, py=0.0, radius=0.30)
print("  %6s %5s %12s %10s" % ("K", "N", "solve [ms]", "status"))
for K in (128, 256, 512):
    for Nh in (15, 20):
        pp = build_planner(o, r_safe=0.35, horizon=Nh, num_samples=K, seed=0)
        s = PlanarState(-2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pp.plan(s, np.array([2.5, 0.0]))                 # warm the caches
        t = time.time()
        rr = pp.plan(s, np.array([2.5, 0.0]))
        ms = (time.time() - t) * 1000.0
        print("  %6d %5d %12.1f %10s %s"
              % (K, Nh, ms, rr.status, "" if ms < 100 else "<-- over 10 Hz budget"))


print("\n" + "=" * 66)
if fail:
    print("FAILED (%d): %s" % (len(fail), ", ".join(fail)))
    raise SystemExit(1)
print("all checks passed")
