#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Offline check of the vicon/world <-> FCU ENU alignment."""
import numpy as np
from haa_frames import WorldToFcu, wrap_pi, yaw_from_quat

fail = []
def check(name, cond, detail=""):
    print("  %-48s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond: fail.append(name)

# ground truth: FCU frame is the world frame rotated 35 deg and shifted
TRUE_DYAW = np.radians(35.0)
TRUE_T = np.array([2.881, -0.319, 0.05])
def Rz(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def w2f(p, yaw):
    return Rz(TRUE_DYAW).dot(p) + TRUE_T, wrap_pi(yaw + TRUE_DYAW)

al = WorldToFcu(alpha=0.3, min_updates=5)
check("not ready before any pair", not al.ready)

rng = np.random.RandomState(0)
for i in range(200):
    pw = rng.uniform(-3, 3, 3); yw = rng.uniform(-np.pi, np.pi)
    pf, yf = w2f(pw, yw)
    # a little sensor noise on the FCU side
    pf = pf + rng.randn(3) * 0.003
    al.update(pw, yw, 100.0 + i * 0.05, pf, yf, 100.0 + i * 0.05)

check("ready after enough pairs", al.ready, "n=%d" % al.n_updates)
check("recovers dyaw", abs(wrap_pi(al.dyaw - TRUE_DYAW)) < np.radians(0.5),
      "%.2f deg (true 35.00)" % np.degrees(al.dyaw))
check("recovers translation", np.linalg.norm(al.t - TRUE_T) < 0.01,
      "err %.4f m" % np.linalg.norm(al.t - TRUE_T))

# round trip
pw = np.array([-2.881, 0.319, 1.0])
pf, yf = al.to_fcu(pw, 0.4)
back, yb = al.to_world(pf, yf)
check("to_fcu -> to_world round trips", np.linalg.norm(back - pw) < 1e-6,
      "err %.2e" % np.linalg.norm(back - pw))
check("yaw round trips", abs(wrap_pi(yb - 0.4)) < 1e-9)

# a raw (unconverted) world coord would be badly wrong -- that's the whole point
check("conversion actually matters", np.linalg.norm(pf - pw) > 1.0,
      "%.2f m apart" % np.linalg.norm(pf - pw))

# stale pair must be rejected
n0 = al.n_updates
al.update([0,0,0], 0.0, 500.0, [0,0,0], 0.0, 501.0)
check("stale pair rejected", al.n_updates == n0)

# quaternion helper
check("yaw_from_quat(identity)=0", abs(yaw_from_quat(0,0,0,1)) < 1e-12)
check("yaw_from_quat(90deg)", abs(yaw_from_quat(0,0,np.sin(np.pi/4),np.cos(np.pi/4)) - np.pi/2) < 1e-9)

fresh = WorldToFcu()
try:
    fresh.to_fcu([0,0,0]); check("unready to_fcu raises", False)
except RuntimeError:
    check("unready to_fcu raises", True)

print("\n  " + al.describe())
print("\n" + "="*62)
print("FAILED: %s" % ", ".join(fail) if fail else "all checks passed")
raise SystemExit(1 if fail else 0)
