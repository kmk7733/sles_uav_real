#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How does the solve time split into K-dependent and K-independent work?

    python3 sweep_kn.py                                  # full grid
    python3 sweep_kn.py --reps 40 --geo on --Ns 30 --Ks 128,96,64

RUN IT WITH THE STACK UP. Idle numbers are a different measurement: the same
config measures 97 ms idle and 127-194 ms with the ZED, the mapper and
foxglove running, and -- the part that matters -- the FIXED cost goes 52 ms
to 87 ms while the per-sample cost barely moves.

Reuses planner/bench.py's world, limits and weights so a number here is
comparable with one from there. The only things varied are K and N.

WHY THIS EXISTS. "Lower num_samples until it fits" assumes the cost is
proportional to K. It is not: the geodesic field is a grid-wide Dijkstra
rebuilt once per solve and does not care how many samples are drawn, and it
was 28.6% of an 86.7 ms tick on this machine. Anything that big and fixed
puts a floor under the budget that no K can get below, so the split has to be
measured before a K is chosen.
"""

import argparse
import sys

import os

import numpy as np


def _find_src(start):
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "planner", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise ImportError("cannot find planner/ above %s" % start)


sys.path.insert(0, _find_src(__file__))

from planner import bench
from planner.dynamics import PlanarLimits
from planner.mppi import PlanarCostWeights
from planner.haa.capped import CappedDynamics
from planner.haa.cost import FrontierMPPI
from planner.safety import PlanarSafetyValidator
from planner.types import PlanarState


def make(occ, K, N, geodesic):
    lim = PlanarLimits(v_max=bench.V_MAX, a_max=bench.A_MAX,
                       omega_max=bench.OMEGA_MAX, alpha_max=bench.ALPHA_MAX,
                       tilt_max=bench.TILT_MAX, j_max=bench.J_MAX)
    w = PlanarCostWeights(w_goal=bench.W_GOAL, w_term_pos=bench.W_TERM_POS,
                          w_term_vel=bench.W_TERM_VEL, w_obs=bench.W_OBS,
                          d_influence=bench.D_INFLUENCE, R_dnu=bench.R_DNU)
    # SIGMA FOLLOWS N. It is min(clipping, rejection) and the rejection
    # ceiling is v_max/(dt*sqrt(N)), so a sweep over N that held sigma fixed
    # would be measuring the wrong planner at every point but one.
    spread = bench.DT * np.sqrt(N)
    a = min(0.48 * bench.A_MAX, 0.54 * bench.V_MAX / spread)
    al = min(0.5 * bench.ALPHA_MAX, 0.54 * bench.OMEGA_MAX / spread)
    return FrontierMPPI(
        CappedDynamics(lim, dt=bench.DT),
        PlanarSafetyValidator(occ, bench.R_EFF, sweep_step=bench.SWEEP_STEP),
        weights=w, horizon=N, num_samples=K, temperature=bench.TEMPERATURE,
        sigma=(a, a, al), seed=0, w_frontier=bench.W_FRONTIER,
        c_occupied=bench.C_OCCUPIED, c_unknown=bench.C_UNKNOWN,
        use_geodesic=geodesic), (a, a, al)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    # UNDER LOAD, REPS ARE THE MEASUREMENT. On an idle box 10 reps give a
    # monotone table; with the ZED and the mapper running, contention makes
    # K=160 measure SLOWER than K=192 and the table stops meaning anything.
    # Narrow the grid and raise the reps instead of reading noise as signal.
    ap.add_argument("--Ks", default="192,160,128,96,64")
    ap.add_argument("--Ns", default="30,25,20")
    ap.add_argument("--geo", default="on,off")
    ap.add_argument("--budget", type=float, default=100.0,
                    help="tick budget in ms (1000/plan_rate)")
    args = ap.parse_args()

    occ, _ = bench.build_world()
    val = PlanarSafetyValidator(occ, bench.R_EFF, sweep_step=bench.SWEEP_STEP)
    rng = np.random.RandomState(0)
    start, clr = bench.find_open_start(val, occ, rng)
    goal = np.array([bench.INTERIOR[2] - 0.7, bench.INTERIOR[3] - 0.7])
    st = PlanarState(start[0], start[1], 0.0, 0.0, 0.0, 0.0)

    print("world %dx%d cells, start clearance %.2f m, budget %.0f ms"
          % (occ.unsafe.shape[1], occ.unsafe.shape[0], clr, args.budget))
    print("")
    print("  geo    N    K   sigma_a   solve p50   p95   valid   verdict")
    print("  ---  ---  ---   -------   ---------  ----   -----   -------")

    rows = []
    for geodesic in [g == "on" for g in args.geo.split(",")]:
        for N in [int(v) for v in args.Ns.split(",")]:
            for K in [int(v) for v in args.Ks.split(",")]:
                pl, sig = make(occ, K, N, geodesic)
                res = [None]

                def one():
                    res[0] = pl.plan(st, goal, a_prev=np.zeros(2))

                p50, p95 = bench.timeit(one, args.reps)
                r = res[0]
                frac = (100.0 * r.n_valid / r.n_samples) if r.n_samples else 0.0
                ok = "FITS" if p95 <= args.budget else ""
                print("  %-3s  %3d  %3d    %.4f    %6.1f  %6.1f   %4.0f%%   %s"
                      % ("on" if geodesic else "OFF", N, K, sig[0], p50, p95,
                         frac, ok))
                rows.append((geodesic, N, K, p50, p95, frac))

    # the fixed cost, read off the K axis rather than assumed
    print("")
    for geodesic in sorted(set(r[0] for r in rows)):
        Nbig = max(r[1] for r in rows)
        sel = [r for r in rows if r[0] is geodesic and r[1] == Nbig]
        if len(sel) < 2:
            continue
        Ks = np.array([r[2] for r in sel], float)
        ts = np.array([r[3] for r in sel], float)
        slope, intercept = np.polyfit(Ks, ts, 1)
        pred = slope * Ks + ts * 0 + intercept
        ss_tot = float(((ts - ts.mean()) ** 2).sum())
        r2 = 1.0 - float(((ts - pred) ** 2).sum()) / ss_tot if ss_tot else 0.0
        order = np.argsort(Ks)
        mono = bool(np.all(np.diff(ts[order]) >= 0))

        # A FIT IS ONLY A MEASUREMENT IF THE DATA HAS A TREND. Under load this
        # table goes non-monotone -- K=96 measuring FASTER than K=64 -- and
        # least squares will still hand back a confident-looking slope and
        # intercept. One was quoted from an R2 of 0.21 in this session and the
        # "geodesic costs 52 ms" claim built on it was wrong. So the numbers
        # are printed only when they mean something.
        head = "  geodesic %-3s at N=%d:" % ("on" if geodesic else "OFF", Nbig)
        if r2 < 0.8 or not mono:
            print("%s  NO USABLE FIT (R2 %+.2f%s)" % (
                head, r2, ", not monotone in K" if not mono else ""))
            print("      the solve did not fall with K here -- that IS the")
            print("      result: K is not the lever. Compare configurations")
            print("      DIRECTLY at matched K instead of through this line.")
        else:
            print("%s  %.4f ms per sample  +  %.1f ms FIXED   (R2 %+.3f)"
                  % (head, slope, intercept, r2))
            print("      -> even at K=1 the tick costs about %.0f ms"
                  % intercept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
