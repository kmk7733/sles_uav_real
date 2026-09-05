#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Andert's inverse measurement model: the profile and the line walk.

    python tests/test_inverse_sensor.py

Was `planar_sim/perception/inverse_sensor.py:_run_tests`, moved out so the
module that every episode imports carries only the code that flies. Nothing in
the checks changed.

WHAT IS PINNED. Eqs. 8-10 -- P(l) is continuous and unimodal at l_p, so the
per-cell maximum is the endpoint or the peak and needs no sampling; sigma_l
follows from the disparity sigma through the convex z = fB/d; and the exact
cell-interval line walk puts obstacle log-odds in the cells the ray actually
crosses, with the per-frame maximum across pixels (eq. 11).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception.inverse_sensor import (
    ProfileParams,
    StereoParams,
    _TAIL_SIGMAS,
    build_frame_grid,
    depth_sigma_to_line_sigma,
    disparity_to_depth_sigma,
    inverse_measurement_logodds,
    inverse_measurement_probability,
    peak_probability,
    validate_profile_params,
)


def _run_tests() -> None:
    ok = [0]

    def check(name, cond, detail=""):
        print("  %-56s %s   %s" % (name, "PASS" if cond else "FAIL", detail))
        assert cond, name
        ok[0] += 1

    fx = fy = 350.0
    cx, cy = 320.0, 180.0
    b = 0.12
    cell = 0.05
    gw, gh = 120, 160
    plane_y, plane_tol = 1.0, 0.15
    prof = ProfileParams(p_min=0.35, eta=0.025)
    S = lambda sd: StereoParams(fx, fy, cx, cy, b, sd, 0.3, 9.0)

    def depth_image(h, w, z):
        """Fronto-parallel wall: optical-axis Z is constant across the image."""
        return np.full((h, w), float(z))

    def frame(z, sd=0.25, hw=(9, 41), yaw=0.0, cam=(3.0, 1.0, 1.0)):
        d = depth_image(hw[0], hw[1], z)
        st = StereoParams(fx, fy, (hw[1] - 1) / 2.0, (hw[0] - 1) / 2.0,
                          b, sd, 0.3, 9.0)
        return build_frame_grid(d, (cam[0], cam[1], cam[2], 0.0, 0.0, yaw),
                                st, cell, gw, gh, plane_y, plane_tol, prof), st

    # ---------------------------------------------------------- the profile
    print("\nPROFILE (eqs. 8-10)")
    sl = 0.05
    lp = 3.0
    check("P is continuous at l_p",
          abs(float(inverse_measurement_probability(lp - 1e-9, lp, sl, 0.35, 0.025))
              - float(inverse_measurement_probability(lp + 1e-9, lp, sl, 0.35, 0.025)))
          < 1e-7)
    peak = float(inverse_measurement_probability(lp, lp, sl, 0.35, 0.025))
    check("peak equals 0.5 + eta/(sqrt(2 pi) sigma_l)",
          abs(peak - float(peak_probability(sl, 0.025))) < 1e-12,
          "P_peak %.4f" % peak)
    ls = np.linspace(0.1, 6.0, 4001)
    P = inverse_measurement_probability(ls, lp, sl, 0.35, 0.025)
    check("unimodal: the maximum is at l_p",
          abs(ls[int(np.argmax(P))] - lp) < 2e-3)
    check("free evidence before the surface is logit(p_min)",
          abs(float(inverse_measurement_logodds(1.0, lp, sl, 0.35, 0.025))
              - np.log(0.35 / 0.65)) < 1e-9)
    check("behind the surface the contribution decays to zero",
          abs(float(inverse_measurement_logodds(lp + 8 * sl, lp, sl,
                                                0.35, 0.025))) < 1e-3,
          "L = %.2e" % float(inverse_measurement_logodds(lp + 8 * sl, lp, sl,
                                                         0.35, 0.025)))

    # ------------------------------------------------------ uncertainty chain
    print("\nUNCERTAINTY CHAIN")
    for z in (1.0, 3.0, 6.0):
        got = float(disparity_to_depth_sigma(z, fx, b, 0.25))
        want = z * z * 0.25 / (fx * b)
        check("sigma_Z = Z^2 sigma_s/(fx b) at %.0f m" % z,
              abs(got - want) < 1e-12, "%.5f m" % got)
    s1 = float(disparity_to_depth_sigma(2.0, fx, b, 0.25))
    s2 = float(disparity_to_depth_sigma(6.0, fx, b, 0.25))
    check("and it scales as Z^2", abs(s2 / s1 - 9.0) < 1e-9,
          "%.5f -> %.5f m" % (s1, s2))
    obl = 1.35
    check("sigma_l = (l_p/Z) sigma_Z applies the obliquity",
          abs(float(depth_sigma_to_line_sigma(s1, 2.0 * obl, 2.0))
              - s1 * obl) < 1e-12)

    # ------------------------------------------------------------- flat wall
    print("\nFLAT WALL")
    z_wall = 2.0
    (L, seen), st = frame(z_wall)
    cam_i, cam_j = int(3.0 / cell), int(1.0 / cell)
    col = L[:, cam_i]
    j_wall = int((1.0 + z_wall) / cell)
    check("free evidence in front of the wall",
          (col[cam_j + 1:j_wall - 2] < 0).all(),
          "min %.3f, max %.3f" % (col[cam_j + 1:j_wall - 2].min(),
                                  col[cam_j + 1:j_wall - 2].max()))
    check("and it equals logit(p_min)",
          abs(col[cam_j + 5] - np.log(0.35 / 0.65)) < 1e-6,
          "%.4f vs %.4f" % (col[cam_j + 5], np.log(0.35 / 0.65)))
    peak_j = int(np.argmax(col))
    check("occupancy peaks at the measured wall",
          abs(peak_j - j_wall) <= 1,
          "peak at cell %d (%.2f m), wall at %d" % (peak_j, peak_j * cell, j_wall))
    check("the peak is positive (occupied evidence)", col[peak_j] > 0,
          "L = %+.3f" % col[peak_j])
    behind = col[j_wall + 6:]
    check("cells behind the wall get no evidence",
          np.allclose(behind, 0.0, atol=1e-6),
          "max |L| = %.2e" % np.abs(behind).max())
    check("and are not marked observed",
          not seen[j_wall + 8:, cam_i].any())

    # -------------------------------------------- distance-dependent width
    print("\nDISTANCE-DEPENDENT UNCERTAINTY")
    # Both ranges are chosen ABOVE the sigma_l floor (0.5 cell = 0.025 m), or
    # the near one would be floor-dominated and would measure the floor rather
    # than the stereo scaling.
    #
    # THE WIDTH IS NOT PROPORTIONAL TO sigma, and asserting that it is would be
    # asserting a falsehood. The above-unknown region runs from where P first
    # exceeds 0.5 to the 3-sigma truncation, and the first of those is
    # sigma * sqrt(2 ln(1/q)) with q = (0.5 - p_min)/(C + 0.5 - p_min) and
    # C = eta/(sqrt(2 pi) sigma) -- so as sigma grows the amplitude C falls and
    # the leading half-width SHRINKS in units of sigma. The test therefore
    # compares each width against what the profile predicts for its own sigma.
    def predicted_cells(sigma):
        c = prof.eta / (np.sqrt(2.0 * np.pi) * sigma)
        q = (0.5 - prof.p_min) / (c + 0.5 - prof.p_min)
        before = sigma * np.sqrt(2.0 * np.log(1.0 / q)) if q < 1.0 else 0.0
        return (before + _TAIL_SIGMAS * sigma) / cell

    widths, sigmas = [], []
    for z in (3.0, 6.0):
        (Lz, _), _ = frame(z)
        widths.append(int((Lz[:, cam_i] > 1e-6).sum()))
        sigmas.append(float(disparity_to_depth_sigma(z, fx, b, 0.25)))
    check("a farther wall gives a wider occupancy peak",
          widths[1] > widths[0], "%d cells at 3 m -> %d at 6 m"
          % (widths[0], widths[1]))
    for z, wmeas, sg in zip((3.0, 6.0), widths, sigmas):
        wpred = predicted_cells(sg)
        check("  the %.0f m peak is as wide as the profile predicts" % z,
              abs(wmeas - wpred) <= 2.0,
              "%d cells measured, %.1f predicted at sigma_l %.4f m"
              % (wmeas, wpred, sg))
    check("sigma_l itself scales as Z^2 between the two ranges",
          abs(sigmas[1] / sigmas[0] - 4.0) < 1e-6,
          "%.4f -> %.4f m, x%.2f" % (sigmas[0], sigmas[1], sigmas[1] / sigmas[0]))

    # ------------------------------------------------ zero disparity noise
    print("\nZERO DISPARITY UNCERTAINTY")
    (L0, _), _ = frame(z_wall, sd=0.0)
    col0 = L0[:, cam_i]
    check("sigma_disp_px = 0 does not divide by zero",
          np.isfinite(col0).all())
    hot = np.flatnonzero(col0 > 0)
    # 3*sigma_floor is 1.5 cells, so the peak occupies a 2-3 cell neighbourhood
    # of l_p -- the tightest this grid can represent. Not one cell: that would
    # be sub-cell knowledge the map does not have.
    check("the peak collapses to the tightest the grid can hold",
          hot.size <= 3 and abs(float(hot.mean()) - j_wall) <= 1.5,
          "cells above zero: %s, wall at %d" % (hot.tolist(), j_wall))
    check("and everything outside that neighbourhood is free or unknown",
          (col0[hot.max() + 1:] == 0).all() and (col0[cam_j + 1:hot.min()] < 0).all())
    check("free evidence before it survives",
          abs(col0[cam_j + 5] - np.log(0.35 / 0.65)) < 1e-6)

    # ------------------------------------------------------- invalid depth
    print("\nINVALID DEPTH")
    d = depth_image(9, 41, z_wall)
    d[:] = np.nan
    st = StereoParams(fx, fy, 20.0, 4.0, b, 0.25, 0.3, 9.0)
    Ln, sn = build_frame_grid(d, (3.0, 1.0, 1.0, 0.0, 0.0, 0.0), st,
                              cell, gw, gh, plane_y, plane_tol, prof)
    check("an all-NaN image leaves the map untouched",
          not Ln.any() and not sn.any())
    d2i = depth_image(9, 41, z_wall)
    d2i[:, :20] = np.nan
    Lp, sp = build_frame_grid(d2i, (3.0, 1.0, 1.0, 0.0, 0.0, 0.0), st,
                              cell, gw, gh, plane_y, plane_tol, prof)
    check("NaN pixels contribute no free space of their own",
          sp.sum() < seen.sum(), "%d cells observed vs %d with a full image"
          % (int(sp.sum()), int(seen.sum())))

    # ------------------------------------------------------------ known pose
    print("\nKNOWN POSE")
    (La, _), _ = frame(z_wall, cam=(3.0, 1.0, 1.0), yaw=0.0)
    (Lb, _), _ = frame(z_wall, cam=(3.0, 1.0, 2.0), yaw=0.0)
    ja = int(np.argmax(La[:, cam_i]))
    jb = int(np.argmax(Lb[:, cam_i]))
    check("moving the camera 1 m moves the wall 1 m, no extra blur",
          abs((jb - ja) * cell - 1.0) < 1.5 * cell,
          "peak cell %d -> %d" % (ja, jb))
    wa = int((La[:, cam_i] > 0.05).sum())
    wb = int((Lb[:, cam_i] > 0.05).sum())
    check("and the peak is no wider after the move",
          abs(wa - wb) <= 1, "%d vs %d cells" % (wa, wb))

    # ------------------------------------------ overlapping pixels combine by max
    print("\nOVERLAPPING PIXELS")
    (L1, _), _ = frame(z_wall, hw=(9, 5))
    (L2, _), _ = frame(z_wall, hw=(9, 81))
    p1, p2 = L1[:, cam_i].max(), L2[:, cam_i].max()
    check("many pixels on one wall do not stack the log-odds",
          abs(p1 - p2) < 1e-6, "%.4f with 5 columns, %.4f with 81" % (p1, p2))

    # ------------------------------------------------------- grid resolution
    print("\nGRID RESOLUTION")
    misses = 0
    for frac in np.linspace(0.0, 0.95, 20):
        z = 2.0 + frac * cell
        (Lf, _), _ = frame(z, sd=0.0)
        if Lf[:, cam_i].max() <= 0:
            misses += 1
    check("the peak is never stepped over, wherever it falls in a cell",
          misses == 0, "%d/20 endpoint offsets lost the peak" % misses)

    # ------------------------------------------------- rays that leave the grid
    print("\nRAYS THAT LEAVE THE GRID")
    # The walk kills a ray once it is past an edge AND stepping further past
    # it, because the grid is convex and it can never come back. The half of
    # that condition worth a test is the OTHER half: a ray may START outside
    # the grid and be heading in, and killing it on position alone would throw
    # away measurements that do land in the map. The camera being outside the
    # mapped rectangle is not hypothetical -- the origin shift does not
    # guarantee otherwise, and `bag_grid_map` sizes its grid from a trajectory
    # bounding box.
    z_in = 2.5
    (L_out, seen_out), _ = frame(z_in, cam=(3.0, 1.0, -1.0))
    j_wall_out = int((-1.0 + z_in) / cell)
    col_out = L_out[:, int(3.0 / cell)]
    check("a camera OUTSIDE the grid, looking in, still maps",
          seen_out.any(), "%d cells observed" % int(seen_out.sum()))
    check("and its wall lands where the geometry says",
          abs(int(np.argmax(col_out)) - j_wall_out) <= 1,
          "peak at %d, wall at %d" % (int(np.argmax(col_out)), j_wall_out))
    check("with free evidence in front of it",
          (col_out[2:j_wall_out - 2] < 0).all(),
          "max %.3f" % col_out[2:j_wall_out - 2].max())

    # Looking OUT of the grid: everything the ray can reach is mapped, and the
    # early exit removes only steps taken outside the map.
    (L_edge, seen_edge), _ = frame(6.0, cam=(3.0, 1.0, 7.5))
    check("a camera near the far edge, looking out, maps what is still inside",
          seen_edge.any(), "%d cells observed" % int(seen_edge.sum()))
    check("and claims nothing it could not see",
          not seen_edge[:int(7.0 / cell), :].any(),
          "nothing observed behind the camera")

    # ------------------------------------------------------------ validation
    print("\nPARAMETER VALIDATION")
    st = S(0.25)
    sigma_min = validate_profile_params(prof, st, cell)
    check("a sane eta validates", sigma_min > 0,
          "tightest sigma_l %.4f m, peak P %.4f"
          % (sigma_min, float(peak_probability(sigma_min, prof.eta))))
    try:
        validate_profile_params(ProfileParams(0.35, 0.1), st, cell)
        raised = False
    except ValueError:
        raised = True
    check("an eta that would push the peak past 1 is rejected", raised,
          "the vendored K = 0.1 is one such value at a 0.05 m cell")

    print("\n%d checks passed" % ok[0])


if __name__ == "__main__":
    _run_tests()
