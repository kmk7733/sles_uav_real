#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Which constraint kills the samples, and at what noise level do they live?"""

import numpy as np
from haa_dynamics import QuadrotorDynamics, hover_state, RPY, OM, V

dyn = QuadrotorDynamics()
x0 = hover_state(0, 0, 1.0)
rng = np.random.RandomState(0)

print("limits: tilt=%.3f rad (%.0f deg)  omega=%.1f rad/s  vel=%.1f m/s"
      % (dyn.tilt_max, np.degrees(dyn.tilt_max), dyn.omega_max, dyn.vel_max))
print("J diag = %s  -> M/J for M=0.05: %.2f rad/s^2"
      % (np.round(np.diag(dyn.J), 4).tolist(), 0.05 / dyn.J[0, 0]))

print("\n--- which constraint binds (N=40, dt=0.1, sig_f=2.0, sig_M=0.05) ---")
K, N, dt = 2000, 40, 0.1
U = np.tile(dyn.hover_control(), (K, N, 1))
U += rng.randn(K, N, 4) * np.array([2.0, 0.05, 0.05, 0.05])
X = dyn.rollout(x0, dyn.clip_control(U), dt)

tilt_ok = (np.abs(X[..., RPY][..., 0]) <= dyn.tilt_max).all(axis=1) & \
          (np.abs(X[..., RPY][..., 1]) <= dyn.tilt_max).all(axis=1)
om_ok = (np.linalg.norm(X[..., OM], axis=-1) <= dyn.omega_max).all(axis=1)
vel_ok = (np.linalg.norm(X[..., V], axis=-1) <= dyn.vel_max).all(axis=1)
fin_ok = np.isfinite(X).all(axis=(1, 2))

for name, m in [("finite", fin_ok), ("tilt", tilt_ok),
                ("omega", om_ok), ("vel", vel_ok)]:
    print("  %-8s pass %5.1f%%" % (name, 100.0 * m.mean()))
print("  ALL      pass %5.1f%%" % (100.0 * (tilt_ok & om_ok & vel_ok & fin_ok).mean()))
print("  max |roll| reached: %.2f rad   max |v|: %.2f m/s"
      % (np.abs(X[..., RPY][..., 0]).max(), np.linalg.norm(X[..., V], axis=-1).max()))

print("\n--- survival vs sigma_moment (iid noise, N=40) ---")
for sM in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    U = np.tile(dyn.hover_control(), (K, N, 1))
    U += rng.randn(K, N, 4) * np.array([2.0, sM, sM, sM])
    X = dyn.rollout(x0, dyn.clip_control(U), dt)
    print("  sigma_M=%.4f -> %5.1f%% feasible" % (sM, 100.0 * dyn.feasible(X).mean()))

print("\n--- survival vs horizon (sigma_M=0.005) ---")
for n in [5, 10, 20, 40]:
    U = np.tile(dyn.hover_control(), (K, n, 1))
    U += rng.randn(K, n, 4) * np.array([2.0, 0.005, 0.005, 0.005])
    X = dyn.rollout(x0, dyn.clip_control(U), dt)
    print("  N=%2d -> %5.1f%% feasible" % (n, 100.0 * dyn.feasible(X).mean()))

print("\n--- smoothed (knot-interpolated) noise, N=40 ---")
for knots in [3, 5, 8]:
    for sM in [0.05, 0.02, 0.01]:
        kn = rng.randn(K, knots, 4) * np.array([2.0, sM, sM, sM])
        idx = np.linspace(0, knots - 1, N)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, knots - 1)
        w = (idx - lo)[None, :, None]
        U = np.tile(dyn.hover_control(), (K, N, 1)) + kn[:, lo] * (1 - w) + kn[:, hi] * w
        X = dyn.rollout(x0, dyn.clip_control(U), dt)
        print("  knots=%d sigma_M=%.3f -> %5.1f%% feasible"
              % (knots, sM, 100.0 * dyn.feasible(X).mean()))
