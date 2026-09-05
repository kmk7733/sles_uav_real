#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time the planner on whatever machine it is sitting on.

    python -m planner.bench                 # the HAA, the way it is flown
    python -m planner.bench --supervisor    # add the DeSimplex probe cost
    python -m planner.bench --json out.json

STANDALONE ON PURPOSE. It imports nothing outside this package, reads no
config.yaml, and needs only numpy and scipy -- so it runs on the vehicle where
`planner/` has been copied on its own and there is no simulator to build a
world with. That is also what makes it a check of the claim in
`planner/__init__.py`: if this file runs, the package really did travel.

EVERY NUMBER BELOW IS PINNED, and pinned to `config.yaml` as shipped rather
than to something convenient. A benchmark whose K, N, dt, sigma, limits, radius
or grid differ from the flown ones is not measuring the planner, and two
machines cannot be compared through it. If config.yaml moves, this file has to
be updated deliberately -- `tests/test_planner_constants.py` is where that is
caught on the simulator side.

WHAT THE ANSWER IS FOR. `mppi.dt` is 0.1 s, so one plan tick has a 100 ms
budget and the interesting question is not milliseconds but how much of that
budget is left. The breakdown matters as much as the total: on the reference
machine 59% of a solve is the geodesic field rebuild, which is a grid-wide
Dijkstra rather than anything MPPI does, and is the first thing to attack if
the budget is tight.
"""

import argparse
import json
import platform
import sys
import time

import numpy as np

from planner.dynamics import PlanarLimits
from planner.grid import PlanarOccupancy
from planner.mppi import PlanarCostWeights
from planner.haa.capped import CappedDynamics
from planner.haa.cost import FrontierMPPI
from planner.haa.geodesic import CostToGo
from planner.safety import PlanarSafetyValidator

# --------------------------------------------------------------- the config
# config.yaml as shipped. `limits_flown` = limits (-) safety, i.e. X_bar, which
# is what `build_planner` hands the solver -- NOT the physical `limits:` block.
V_MAX, OMEGA_MAX = 0.3100, 0.4887        # 0.35 - z_vel, 0.5236 - z_omega
A_MAX, ALPHA_MAX = 3.5000, 3.8500        # a_max_eff - a_reserve, alpha likewise
J_MAX, TILT_MAX = 5.500, 0.610865        # limits.j_max, 35 deg
K, N, DT = 192, 30, 0.10                 # mppi.num_samples / horizon / dt
TEMPERATURE = 1.0
SIGMA = (0.305629, 0.305629, 0.481810)   # derived: Config.mppi_sigma
R_EFF, SWEEP_STEP = 0.4600, 0.025
W_GOAL, W_TERM_POS, W_TERM_VEL = 1.0, 10.0, 2.0
W_OBS, D_INFLUENCE, R_DNU = 20.0, 0.6000, (1.0, 1.0, 0.2)
W_FRONTIER, C_OCCUPIED, C_UNKNOWN = 5.0, 2.0, -4.0

# The arena the numbers were made on: 5 x 7 m at 0.05 m cells = 100 x 140.
INTERIOR = (-2.5, -3.5, 2.5, 3.5)
RES = 0.05

# Reference, so a run on another machine is a RATIO and not a bare number.
# AMD Ryzen 9 7900X (Zen 4), numpy 1.26.4, python 3.10, this same file.
#
# SINGLE-THREAD-BOUND, MEASURED: 1, 4 and 8 BLAS threads all gave 11.3 ms.
# There is no large BLAS call to parallelise -- the work is elementwise numpy
# over (K, N, .) arrays plus a python loop over N -- so a machine with more
# cores does not get a faster solve, and the ratio below is a SINGLE-CORE
# ratio. Do not expect Xavier's eight to help.
REFERENCE = {
    "machine": "AMD Ryzen 9 7900X",
    "plan_ms": 11.57,
    "costtogo_ms": 2.86,
    "mppi_only_ms": 7.22,
    "in_s_haa_ms": 12.01,
}


def build_world(n_pillars=5, seed=7):
    """A clutter room of the shipped geometry. Deterministic in `seed`."""
    x0, y0, x1, y1 = INTERIOR
    w = int(round((x1 - x0) / RES))
    h = int(round((y1 - y0) / RES))
    yy, xx = np.mgrid[0:h, 0:w]
    wx = x0 + (xx + 0.5) * RES
    wy = y0 + (yy + 0.5) * RES
    grid = np.zeros((h, w), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True   # walls
    rng = np.random.RandomState(seed)
    placed = []
    while len(placed) < n_pillars:
        c = np.array([rng.uniform(x0 + 0.8, x1 - 0.8),
                      rng.uniform(y0 + 0.8, y1 - 0.8)])
        if all(np.hypot(*(c - q)) > 1.3 for q in placed):
            placed.append(c)
            grid |= ((wx - c[0]) ** 2 + (wy - c[1]) ** 2) <= 0.22 ** 2
    return PlanarOccupancy(grid, RES, (x0, y0)), placed


def build_planner(occ, use_geodesic=True):
    lim = PlanarLimits(v_max=V_MAX, a_max=A_MAX, omega_max=OMEGA_MAX,
                       alpha_max=ALPHA_MAX, tilt_max=TILT_MAX, j_max=J_MAX)
    dyn = CappedDynamics(lim, dt=DT)     # mppi.cap_velocity is true
    val = PlanarSafetyValidator(occ, R_EFF, sweep_step=SWEEP_STEP)
    w = PlanarCostWeights(w_goal=W_GOAL, w_term_pos=W_TERM_POS,
                          w_term_vel=W_TERM_VEL, w_obs=W_OBS,
                          d_influence=D_INFLUENCE, R_dnu=R_DNU)
    return FrontierMPPI(dyn, val, weights=w, horizon=N, num_samples=K,
                        temperature=TEMPERATURE, sigma=SIGMA, seed=0,
                        w_frontier=W_FRONTIER, c_occupied=C_OCCUPIED,
                        c_unknown=C_UNKNOWN, use_geodesic=use_geodesic)


def timeit(fn, reps, warm=3):
    """Median and p95 in ms. Warm-up first: the first call pays for numpy's
    lazy allocation and for scipy's EDT import, neither of which recurs."""
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    a = np.array(out)
    return float(np.median(a)), float(np.percentile(a, 95))


def find_open_start(val, occ, rng):
    """A start with real clearance, so the solve is a solve and not the cheap
    rejection of a state already inside the inflated set."""
    x0, y0, x1, y1 = INTERIOR
    best, bc = None, -1.0
    for _ in range(3000):
        q = np.array([rng.uniform(x0 + 0.6, x1 - 0.6),
                      rng.uniform(y0 + 0.6, y1 - 0.6)])
        c = float(val.clearance(q))
        if c > bc:
            best, bc = q, c
    return best, bc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--reps', type=int, default=40,
                    help='timed calls per measurement (default 40)')
    ap.add_argument('--supervisor', action='store_true',
                    help='also time one S_HAA probe, the DeSimplex per-tick cost')
    ap.add_argument('--json', metavar='PATH', help='write the numbers here too')
    args = ap.parse_args(argv)

    print('machine   %s  %s' % (platform.processor() or platform.machine(),
                                platform.system()))
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name') or line.startswith('Model'):
                    print('cpu       %s' % line.split(':', 1)[1].strip())
                    break
    except (IOError, OSError):
        pass
    print('python    %s   numpy %s' % (platform.python_version(), np.__version__))
    print('config    K=%d  N=%d  dt=%.2f  ->  budget %.0f ms/tick'
          % (K, N, DT, 1000 * DT))

    occ, _ = build_world()
    p = build_planner(occ)
    rng = np.random.RandomState(0)
    start, clear = find_open_start(p.validator, occ, rng)
    goal = np.array([-start[0], -start[1]])
    xi = np.array([start[0], start[1], 0.20, 0.0, 0.3, 0.0])
    print('world     %dx%d cells (%d)   start clearance %.2f m (r_eff %.2f)'
          % (occ.unsafe.shape[1], occ.unsafe.shape[0], occ.unsafe.size,
             clear, R_EFF))
    print()

    res = {}
    total, total95 = timeit(lambda: p.plan(xi, goal), args.reps)
    res['plan_ms'] = total
    print('HAA plan()              %7.2f ms   p95 %6.2f   -> %5.1f Hz'
          % (total, total95, 1000.0 / total))

    ctg, _ = timeit(lambda: CostToGo(p.validator.occ, goal, p.validator.r_safe,
                                     unknown_free=p.unknown_free), args.reps)
    res['costtogo_ms'] = ctg
    print('  CostToGo rebuild      %7.2f ms   (%4.1f%% -- a grid-wide Dijkstra,'
          ' not MPPI)' % (ctg, 100.0 * ctg / total))

    p2 = build_planner(occ, use_geodesic=False)
    mppi, _ = timeit(lambda: p2.plan(xi, goal), args.reps)
    res['mppi_only_ms'] = mppi
    print('  MPPI proper           %7.2f ms   (%4.1f%%)'
          % (mppi, 100.0 * mppi / total))

    U = np.random.RandomState(0).randn(K, N, 3) * np.asarray(SIGMA)
    Uc = p.dyn.clip_inputs(U, a_prev=np.zeros(2))
    X = p.dyn.rollout(xi, Uc.copy())
    for label, fn in (
            ('clip_inputs', lambda: p.dyn.clip_inputs(U, a_prev=np.zeros(2))),
            ('rollout (capped)', lambda: p.dyn.rollout(xi, Uc.copy())),
            ('cost J', lambda: p._cost(X, Uc, goal, np.zeros(2))),
            ('inputs_ok', lambda: p.dyn.inputs_ok(Uc, a_prev=np.zeros(2))),
            ('states_ok', lambda: p.dyn.states_ok(X))):
        m, _ = timeit(fn, args.reps)
        res[label.split()[0] + '_ms'] = m
        print('      %-20s %7.2f ms' % (label, m))

    if args.supervisor:
        from planner.dynamics import PlanarDynamics
        from planner.supervisor import DeSimplexSupervisor
        # dyn_full is the HPA envelope, UNTIGHTENED: R_Nr asks what the full
        # authority can recover. hpa: v_max 0.5, a_max 5.0, omega 0.5236.
        full = PlanarLimits(v_max=0.5, a_max=5.0, omega_max=0.5236,
                            alpha_max=4.0, tilt_max=np.radians(45.0), j_max=8.0)
        x_true = PlanarLimits(v_max=0.35, a_max=3.5, omega_max=0.5236,
                              alpha_max=3.85, tilt_max=TILT_MAX, j_max=J_MAX)
        sup = DeSimplexSupervisor(build_planner(occ), build_planner(occ),
                                  PlanarDynamics(full, dt=DT), p.validator,
                                  goal, lim_x=x_true)
        m, _ = timeit(lambda: sup.in_s_haa(xi), max(8, args.reps // 4))
        res['in_s_haa_ms'] = m
        print()
        print('S_HAA probe             %7.2f ms   (one full solve; a DeSimplex'
              ' tick costs 1..n_r+1)' % m)

    print()
    ref = REFERENCE['plan_ms']
    print('vs %s: %.2f ms -> %.2f ms, %.1fx'
          % (REFERENCE['machine'], ref, total, total / ref))
    print()
    used = 100.0 * total / (1000.0 * DT)
    verdict = ('FITS, %.0f%% of budget spare' % (100.0 - used) if used < 100.0
               else 'DOES NOT FIT -- %.1fx over budget' % (used / 100.0))
    print('one HAA tick uses %.1f%% of the %.0f ms budget: %s'
          % (used, 1000 * DT, verdict))
    if res.get('in_s_haa_ms'):
        ds = total + res['in_s_haa_ms']
        print('a DeSimplex tick is at least %.1f ms (HAA + one probe), %.0f%% '
              'of budget' % (ds, 100.0 * ds / (1000.0 * DT)))

    if args.json:
        res['cpu'] = platform.processor() or platform.machine()
        res['numpy'] = np.__version__
        with open(args.json, 'w') as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print('\n-> %s' % args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
