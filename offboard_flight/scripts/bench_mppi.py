#!/usr/bin/env python3
"""Offline timing of the CPU MPPI planner, same params as mpc.launch.

Measures plan() solve time on a synthetic 224x204 @ 0.05 m grid (the size
depth_to_grid actually publishes), so it is directly comparable to the CNN
inference benchmark.
"""
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/home/rogx/catkin_ws/src/mpc_controller/src")

from mpc_controller.mppi import MPPI, MPPIPlanner
from mpc_controller.quadrotor import PlanarHolonomic
from mpc_controller.obstacle_map import OccupancyGridMap
from mpc_controller.cost_to_go import compute_cost_to_go

# ---- params from mpc.launch
N, K, M = 30, 256, 1
DT = 0.1
A_MAX, V_CAP, V_MAX = 0.7, 0.8, 1.5
ROBOT_R = 0.31

CELL = 0.05
W, H = 224, 204

rng = np.random.default_rng(0)
occupied = np.zeros((H, W), dtype=bool)
occupied[0, :] = occupied[-1, :] = True
occupied[:, 0] = occupied[:, -1] = True
for _ in range(12):                       # scattered obstacles
    cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
    occupied[cy - 6:cy + 6, cx - 6:cx + 6] = True

occ = OccupancyGridMap(occupied, CELL, -1.1, -5.1)

planar = PlanarHolonomic()
mppi = MPPI(num_nodes=N, num_rollouts=K,
            channel_scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            sigma=0.7, temperature=0.3, use_noise_ramp=False,
            noise_beta=0.8, cost_norm=True)
ctrl_lo = np.array([-A_MAX, -A_MAX, -3.0], dtype=np.float32)
ctrl_hi = np.array([+A_MAX, +A_MAX, +3.0], dtype=np.float32)
planner = MPPIPlanner(
    mppi, planar, occ, ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi,
    robot_radius=ROBOT_R, safe_margin=0.0,
    state_lims=[None, None, None, (-V_MAX, V_MAX), (-V_MAX, V_MAX), (-3.0, 3.0)],
    near_obs_w=3.0, z_track_w=0.0, v_cap=V_CAP, v_cap_w=50.0,
    smoothing_w=1.0, dt=DT, mppi_iters=M,
    consistency_w=200.0, path_consistency_w=100.0,
    near_obs_soft=True, near_obs_falloff=0.30,
    revalidate_mean=True, retry_keep_warm=True)

goal = np.array([2.5, 0.0, 0.0], dtype=np.float32)
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# cost-to-go field (refreshed every ctg_period=0.5 s in the node)
t0 = time.perf_counter()
field = compute_cost_to_go(occ, goal[:2], ROBOT_R)
ctg_ms = (time.perf_counter() - t0) * 1e3
planner.set_cost_to_go(field)

for _ in range(5):
    planner.plan(x0, goal)

lat = []
for _ in range(200):
    t0 = time.perf_counter()
    planner.plan(x0, goal)
    lat.append((time.perf_counter() - t0) * 1e3)

# EDT cost (rebuilt on every new grid message)
occ2 = OccupancyGridMap(occupied, CELL, -1.1, -5.1)
t0 = time.perf_counter()
occ2._ensure_edt()
edt_ms = (time.perf_counter() - t0) * 1e3

lat.sort()
n = len(lat)
print(f"grid {W}x{H} @ {CELL} m | K={K} rollouts, N={N} horizon, M={M} iters")
print(f"  MPPI plan()      mean {statistics.mean(lat):7.2f} ms | "
      f"p50 {lat[n // 2]:7.2f} | p90 {lat[int(n * .9)]:7.2f} | p99 {lat[int(n * .99)]:7.2f}")
print(f"  cost_to_go()     {ctg_ms:7.2f} ms  (every 0.5 s)")
print(f"  EDT build        {edt_ms:7.2f} ms  (every new grid msg)")
print(f"  rollout-steps    {K * N * M:,} (vs CNN 12.78 M MAC)")
