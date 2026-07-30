#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Where does the solve time actually go?"""

import numpy as np
import os
import time

from haa_dynamics import QuadrotorDynamics, AttitudeStabilizedQuadrotor, hover_state

print("cores: %d   loadavg: %s" % (os.cpu_count(), open("/proc/loadavg").read().split()[:3]))

dyn = QuadrotorDynamics()
x0 = hover_state(0, 0, 1.0)
rng = np.random.RandomState(0)


def timeit(fn, n=5):
    fn()
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n * 1000.0


print("\n--- rollout cost vs K, at N=20, substeps=1 ---")
st1 = AttitudeStabilizedQuadrotor(dyn, substeps=1)
for K in (64, 256, 1024, 4096):
    V = rng.randn(K, 20, 4) * 0.5
    ms = timeit(lambda: st1.rollout(x0, V, 0.1))
    print("  K=%5d  %7.1f ms   (%.4f ms per sample)" % (K, ms, ms / K))

print("\n--- rollout cost vs N, at K=1024, substeps=1 ---")
for N in (10, 20, 40):
    V = rng.randn(1024, N, 4) * 0.5
    ms = timeit(lambda: st1.rollout(x0, V, 0.1))
    print("  N=%3d  %7.1f ms   (%.2f ms per step)" % (N, ms, ms / N))

print("\n--- substeps multiplier (K=1024, N=20) ---")
for ss in (1, 2, 3):
    st = AttitudeStabilizedQuadrotor(dyn, substeps=ss)
    V = rng.randn(1024, 20, 4) * 0.5
    print("  substeps=%d  %7.1f ms" % (ss, timeit(lambda: st.rollout(x0, V, 0.1))))

print("\n--- raw base step_vec vs stabilized step_vec (K=1024) ---")
X = np.tile(x0, (1024, 1))
U = np.tile(dyn.hover_control(), (1024, 1))
print("  base.step_vec       %6.3f ms" % timeit(lambda: dyn.step_vec(X, U, 0.1), 50))
V1 = np.zeros((1024, 4))
print("  stabilized.step_vec %6.3f ms" % timeit(lambda: st1.step_vec(X, V1, 0.1), 50))

print("\n--- inner-loop stability check: does substeps=1 hold at k_rate=12? ---")
for ss in (1, 2):
    st = AttitudeStabilizedQuadrotor(dyn, substeps=ss)
    V = np.tile(st.hover_control(), (500, 25, 1))
    V += np.random.RandomState(3).randn(500, 25, 4) * np.array([1.2, 1.2, 0.4, 0.2])
    Xs = st.rollout(x0, st.clip_control(V), 0.1)
    print("  substeps=%d  feasible %.1f%%  max|roll| %.3f  max|w| %.2f"
          % (ss, 100.0 * st.feasible(Xs).mean(), np.abs(Xs[..., 6]).max(),
             np.abs(Xs[..., 9:12]).max()))
