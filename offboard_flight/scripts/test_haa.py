#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Offline checks for the HAA dynamics + MPPI. No ROS, no vehicle."""

import numpy as np
import time

from haa_dynamics import (QuadrotorDynamics, ThrustRateQuadrotor,
                          AttitudeStabilizedQuadrotor,
                          hover_state, G, ROGX_MASS, ROGX_INERTIA)
from haa_obstacles import InflatedGrid, effective_radius
from haa_mppi import HAAMPPI

fail = []


def check(name, cond, detail=""):
    print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        fail.append(name)


print("=== 1. dynamics ===")
dyn = QuadrotorDynamics()
print("  m = %.3f kg, hover thrust = %.3f N" % (dyn.m, dyn.hover_thrust))
print("  J =\n%s" % np.array_str(dyn.J, precision=5))

# hover must be a fixed point
x = hover_state(0, 0, 1.0)
X = dyn.rollout(x, np.tile(dyn.hover_control(), (1, 60, 1)), 0.05)
drift = np.abs(X[0, -1, 2] - 1.0)
check("hover holds altitude over 3 s", drift < 1e-6, "drift=%.2e m" % drift)

# zero thrust must free-fall at g
X = dyn.rollout(x, np.zeros((1, 20, 4)), 0.05)
vz = X[0, -1, 5]
check("zero thrust free-falls at g", abs(vz - (-G * 1.0)) < 1e-6,
      "vz=%.4f expected %.4f" % (vz, -G))

# a positive Mx must spin up roll in the +x sense
U = np.zeros((1, 10, 4)); U[:, :, 0] = dyn.hover_thrust; U[:, :, 1] = 0.01
X = dyn.rollout(x, U, 0.02)
check("positive Mx rolls positive", X[0, -1, 6] > 0, "roll=%.5f" % X[0, -1, 6])

# inertia asymmetry must show up: same moment about x and z -> different rates
Ux = np.zeros((1, 10, 4)); Ux[:, :, 0] = dyn.hover_thrust; Ux[:, :, 1] = 0.01
Uz = np.zeros((1, 10, 4)); Uz[:, :, 0] = dyn.hover_thrust; Uz[:, :, 3] = 0.01
wx = dyn.rollout(x, Ux, 0.02)[0, -1, 9]
wz = dyn.rollout(x, Uz, 0.02)[0, -1, 11]
check("Jxx < Jzz gives wx > wz for equal moment", wx > wz,
      "wx=%.4f wz=%.4f" % (wx, wz))

# feasibility must reject an over-tilted rollout
Ubad = np.zeros((1, 40, 4)); Ubad[:, :, 0] = dyn.hover_thrust; Ubad[:, :, 1] = 0.5
Xbad = dyn.rollout(x, Ubad, 0.05)
check("over-tilt rejected by feasible()", not dyn.feasible(Xbad)[0])
check("hover accepted by feasible()",
      dyn.feasible(dyn.rollout(x, np.tile(dyn.hover_control(), (1, 40, 1)), 0.05))[0])


print("\n=== 2. obstacle inflation ===")
r_eff = effective_radius(0.31, 0.05, 0.0)
check("r_eff = r_Q + r_track + r_perc", abs(r_eff - 0.36) < 1e-12, "%.3f m" % r_eff)


class FakeGridMsg(object):
    class _H:
        frame_id = "map"
        stamp = 0

    class _I:
        resolution = 0.05
        width = 200
        height = 200

        class origin:
            class position:
                x = -5.0
                y = -5.0

    def __init__(self, data):
        self.header = self._H()
        self.info = self._I()
        self.data = data


# single occupied cell at world (0,0)
g = np.zeros((200, 200), dtype=np.int16)
g[100, 100] = 100
msg = FakeGridMsg(g.reshape(-1).tolist())

grid = InflatedGrid(unknown_is_obstacle=False)
grid.update(msg, r_eff)
n_before, n_after = 1, int(grid.blocked.sum())
expected = np.pi * (r_eff / 0.05) ** 2
check("one cell inflates to ~disk of r_eff",
      abs(n_after - expected) / expected < 0.15,
      "%d cells, disk~%.0f" % (n_after, expected))
check("cell at the obstacle collides", bool(grid.collides(0.0, 0.0)))
check("point just inside r_eff collides", bool(grid.collides(0.30, 0.0)))
check("point well outside r_eff is free", not bool(grid.collides(0.60, 0.0)))
check("off-grid counts as blocked", bool(grid.collides(99.0, 99.0)))

empty = InflatedGrid()
check("no map -> everything blocked (fails safe)", bool(empty.collides(0.0, 0.0)))


print("\n=== 3. MPPI ===")


class Free(object):
    ready = True
    def collides(self, x, y, **kw):
        return np.zeros(np.asarray(x).shape, dtype=bool)


ctbr = ThrustRateQuadrotor(dyn)
mppi = HAAMPPI(ctbr, Free(), dt=0.1, z_hold=1.0)
x0 = hover_state(0, 0, 1.0)

t = time.time()
res = mppi.plan(x0, np.array([2.0, 0.0]))
solve = (time.time() - t) * 1000.0
check("solve returns feasible in free space", res.feasible, res.reason)
check("some samples survived", res.n_valid > 0, "%d/%d" % (res.n_valid, res.n_samples))
print("  solve time: %.0f ms  (10 Hz budget = 100 ms)" % solve)

if res.feasible:
    check("nominal keeps altitude near z_hold",
          abs(res.X[:, 2] - 1.0).max() < 0.5,
          "max |dz| = %.3f m" % abs(res.X[:, 2] - 1.0).max())
    d0 = np.linalg.norm(res.X[0, :2] - np.array([2.0, 0.0]))
    dN = np.linalg.norm(res.X[-1, :2] - np.array([2.0, 0.0]))
    check("nominal makes progress toward the goal", dN < d0,
          "%.3f -> %.3f m" % (d0, dN))

# repeated solves: warm start must not blow up
ok = True
for _ in range(10):
    r = mppi.plan(x0, np.array([2.0, 0.0]))
    ok &= r.feasible
check("10 consecutive warm-started solves stay feasible", ok)

# fully blocked -> must declare infeasible
class Blocked(object):
    ready = True
    def collides(self, x, y, **kw):
        return np.ones(np.asarray(x).shape, dtype=bool)

m2 = HAAMPPI(ctbr, Blocked(), horizon=15, num_samples=200, dt=0.1, z_hold=1.0)
check("all-blocked declares infeasible", not m2.plan(x0, np.array([1.0, 0.0])).feasible)


print("\n" + "=" * 60)
if fail:
    print("FAILED: %s" % ", ".join(fail))
    raise SystemExit(1)
print("all checks passed")


print("\n=== 4. CTBR model ===")
ctbr = ThrustRateQuadrotor(dyn)

# hover must be a fixed point in this parameterisation too
Xh = ctbr.rollout(hover_state(0,0,1.0), np.tile(ctbr.hover_control(), (1,30,1)), 0.1)
check("CTBR hover holds altitude", abs(Xh[0,-1,2]-1.0) < 1e-6,
      "drift=%.2e" % abs(Xh[0,-1,2]-1.0))

# commanded body rate must be tracked, not integrated from a moment
U = np.tile(ctbr.hover_control(), (1,20,1)); U[:,:,1] = 0.5
Xr = ctbr.rollout(hover_state(0,0,1.0), U, 0.1)
check("body-rate command is tracked", abs(Xr[0,-1,9]-0.5) < 1e-3,
      "wx=%.4f cmd=0.5" % Xr[0,-1,9])

# J must NOT change the trajectory under CTBR
d2 = QuadrotorDynamics(inertia=ROGX_INERTIA*7.0)
Xj = ThrustRateQuadrotor(d2).rollout(hover_state(0,0,1.0), U, 0.1)
check("J has no effect under CTBR (7x inertia)",
      np.abs(Xj - Xr).max() < 1e-12, "max diff %.2e" % np.abs(Xj-Xr).max())

V = np.tile(ctbr.hover_control(), (2000, 15, 1))
V += np.random.RandomState(1).randn(2000,15,4) * np.array([3.0,0.35,0.35,0.2])
Xs = ctbr.rollout(hover_state(0,0,1.0), ctbr.clip_control(V), 0.1)
frac = ctbr.feasible(Xs).mean()
check("CTBR keeps enough survivors for MPPI", frac > 0.5, "%.1f%%" % (100*frac))
print("  max |roll| %.3f rad  max |v| %.2f m/s  max |w| %.2f rad/s"
      % (np.abs(Xs[...,6]).max(), np.linalg.norm(Xs[...,3:6],axis=-1).max(),
         np.abs(Xs[...,9:12]).max()))

print("\n  %6s %5s %10s %10s" % ("K","H","solve[ms]","valid%"))
for K in (128, 256, 512):
    for H in (15, 20):
        mm = HAAMPPI(ctbr, Free(), horizon=H, num_samples=K, dt=0.1, z_hold=1.0)
        t=time.time(); r=mm.plan(hover_state(0,0,1.0), np.array([2.0,0.0]))
        ms=(time.time()-t)*1000
        print("  %6d %5d %10.0f %10.1f %s" % (K,H,ms,
              100.0*r.n_valid/max(r.n_samples,1), "" if ms<100 else "<-- over budget"))
