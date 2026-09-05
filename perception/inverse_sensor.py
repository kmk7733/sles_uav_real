#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Andert's inverse measurement model, drawn into a 2D grid by exact line walk.

    python tests/test_inverse_sensor.py             # the tests for this module

Andert, "Drawing Stereo Disparity Images into Occupancy Grids: Measurement
Model and Fast Implementation" (IROS 2009), eqs. 8-11.

    stereo depth (optical-axis Z, NaN where invalid)
        -> back-project with K, rotate by the KNOWN camera pose
        -> height band filter                       (planar grid)
        -> disparity uncertainty -> sigma_Z -> sigma_l
        -> continuous profile P(l), log-odds L(l)   (eqs. 8-10)
        -> exact cell-interval line walk, per-cell max over the interval
        -> per-frame max across pixels              (eq. 11, obstacle priority)
        -> temporal log-odds accumulation

WHAT IS AND IS NOT MODELLED
The camera pose is KNOWN EXACTLY. The only probabilistic term is the stereo
disparity uncertainty `sigma_disp_px`, the same number the sensor draws its
noise with. There is no translational or angular pose uncertainty and no
section-VII grid blur: that post-processing exists in the paper to smear a grid
built from an UNCERTAIN pose, and applying it here would inflate the map for an
error this simulator does not commit. Grid discretisation is deterministic and
is handled by taking the exact maximum over each cell's line interval, not by
being called an uncertainty and blurred away.

WHY THE PER-CELL MAXIMUM IS EXACT AND CHEAP
P(l) is continuous at l = l_p -- the two branches of the base term meet there,
both giving 0.5 + eta/(sqrt(2 pi) sigma_l) -- and it is unimodal: rising below
l_p, falling above. L = logit(P) is monotone in P. So the maximum of L over a
cell's interval [a, b] is attained at clip(l_p, a, b) and costs ONE profile
evaluation per cell. No dense sampling, no lookup table, and the narrow
occupied peak cannot be stepped over however coarse the grid is.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# RELATIVE, AND IT HAS TO STAY RELATIVE. This module and renderer.py are
# deployed verbatim to the vehicle as ~/catkin_ws/src/perception/ (see
# VENDOR.md), where the package is not called planar_sim.perception. A
# relative import resolves identically under either name, so the two copies
# stay byte-identical and a fix made here is a fix made on the aircraft.
# Making it absolute again breaks the deployment, silently, at import time.
from .renderer import (
    NATIVE_B, NATIVE_F, CameraModel, camera_to_renderer)

__all__ = [
    "StereoParams",
    "ProfileParams",
    "disparity_to_depth_sigma",
    "depth_sigma_to_line_sigma",
    "inverse_measurement_probability",
    "inverse_measurement_logodds",
    "validate_profile_params",
    "build_frame_grid",
]

# Keeps the logit finite. Numerical only: it does not redefine the model, and
# `validate_profile_params` refuses parameters that would push the peak into
# the clip in the first place.
_P_EPS = 1e-6

# Below this the Gaussian contributes less than exp(-4.5) = 1.1% of its
# amplitude, so the line is not drawn past it. The free-space run before the
# surface and the peak are both inside; everything beyond is P = 0.5, L = 0,
# i.e. no evidence, which is what "unknown behind the obstacle" means.
_TAIL_SIGMAS = 3.0

# The NEAR side of the peak, where the walk enters instead of leaves.
# More than this many sigma IN FRONT of the measured surface the Gaussian has
# died and P(l) is exactly its base, p_min -- so the cell's value is the
# constant logit(p_min) and the exp/log/where that produce it are waste. At
# 6 sigma the term dropped is exp(-18) = 1.5e-8 of the amplitude, i.e. about
# 4e-8 of log-odds against a clip of 10: five orders of magnitude more
# conservative than the FAR-side cut this model already takes at
# _TAIL_SIGMAS = 3 (exp(-4.5) = 1.1e-2).
#
# It matters because it is where the time goes. Most cells on a ray are far in
# front of the surface -- at 2.4 m range and sigma_l 0.034 m, the peak window
# is ~6 cells of a ~50-cell walk -- so the full profile was being evaluated for
# roughly ninety per cent of cells in order to return the same number every
# time. Profiled on the aircraft: the profile evaluation was 34% of the frame.
_NEAR_SIGMAS = 6.0

# THE sigma_l -> 0 LIMIT. A grid cannot represent a profile narrower than a
# cell, so sigma_l is floored at half a cell. At sigma_disp_px = 0 the
# measurement then becomes a sharply localised occupied peak: positive evidence
# confined to the two or three cells within 3*sigma_floor = 1.5 cells of l_p,
# logit(p_min) free evidence in every cell before it, and exactly 0 after. That
# is a deterministic limit rather than a division by zero, and it is the
# tightest an occupancy grid of this resolution can represent -- claiming a
# single cell would be claiming sub-cell knowledge the grid does not have. The
# floor also bounds the peak amplitude eta/(sqrt(2 pi) sigma_l), which is what
# `validate_profile_params` checks against.
_SIGMA_L_FLOOR_CELLS = 0.5


class StereoParams(CameraModel):
    """The stereo calibration BOTH the sensor and the mapper read.

    One object, because the mapper's uncertainty and the sensor's noise are the
    same physical quantity: if `sigma_disp_px` here disagreed with the value
    `simulate_stereo_depth` drew with, the map would be confidently wrong in a
    way no test on either module alone could see.

    A stereo rig IS a camera plus a baseline plus a matcher error, so this
    extends `CameraModel` rather than restating its intrinsics. That also makes
    it the object `render_depth` takes, which is the point: the image the
    renderer draws and the geometry the mapper back-projects with are then the
    same numbers by construction, not by two `from_config` calls agreeing.
    """

    __slots__ = ("baseline_m", "sigma_disp_px")

    def __init__(self, fx, fy, cx, cy, baseline_m, sigma_disp_px,
                 z_min_m, z_max_m, width=None, height=None):
        # The raster is optional so a caller that only wants the back-projection
        # geometry -- the unit tests below, a recorded image of known size --
        # need not invent one. It defaults to the raster those intrinsics
        # describe, i.e. a principal point at the centre.
        width = int(round(2.0 * float(cx))) if width is None else width
        height = int(round(2.0 * float(cy))) if height is None else height
        super(StereoParams, self).__init__(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
            z_min_m=z_min_m, z_max_m=z_max_m)
        self.baseline_m = float(baseline_m)
        self.sigma_disp_px = float(sigma_disp_px)
        if not np.isfinite(self.baseline_m) or self.baseline_m <= 0.0:
            raise ValueError("baseline_m must be finite and positive, got %r"
                             % self.baseline_m)
        if not np.isfinite(self.sigma_disp_px) or self.sigma_disp_px < 0.0:
            raise ValueError("sigma_disp_px must be finite and non-negative, "
                             "got %r" % self.sigma_disp_px)

    @staticmethod
    def default_sigma_disp_px(render_scale=1.0):
        """The disparity uncertainty when `mapping.sigma_disp_px` is null [px].

        0.5 px at the vendored HD720 reference (F = 700), rescaled to this
        render's focal length. IT MUST SCALE WITH F, or the modelled stereo
        error changes with the render resolution: sigma_Z = Z^2 sigma_s/(fx b)
        with fx proportional to scale, so halving the raster while holding
        sigma_s in pixels doubles sigma_Z and smears every surface twice as
        deep. Render resolution is OUR choice; the sensor's range accuracy is
        not, so this keeps sigma_Z the same number at every scale.

        The 700 is the vendored HD720 reference focal length the 0.5 px was
        quoted at, kept as a literal: it is a property of that published
        figure, not of the camera this simulator renders, so it must not
        follow `NATIVE_F` if the modelled raster ever changes.
        """
        return 0.5 * (NATIVE_F * float(render_scale) / 700.0)

    @classmethod
    def from_config(cls, cfg):
        """Read the one place these live: `mapping:` plus the camera raster."""
        m = cfg.mapping
        scale = float(m.render_scale)
        sd = m.get("sigma_disp_px")
        sd = (cls.default_sigma_disp_px(scale) if sd is None else float(sd))
        # `depth_noise` is the one switch. False means an ideal sensor, i.e.
        # sigma_disp_px = 0. It does NOT disable the correspondence mask:
        # occlusion is geometry, not noise, and a camera pair cannot see past a
        # near surface however good it is.
        if not bool(m.get("depth_noise", True)):
            sd = 0.0
        return cls.from_scale(scale, z_min_m=float(m.z_min),
                              z_max_m=float(m.z_max),
                              baseline_m=NATIVE_B, sigma_disp_px=sd)

    def __repr__(self):
        return ("StereoParams(%dx%d, fx=%.1f, cx=%.1f, cy=%.1f, B=%.3f, "
                "sigma_disp_px=%.3f, z=[%.2f, %.2f])"
                % (self.width, self.height, self.fx, self.cx, self.cy,
                   self.baseline_m, self.sigma_disp_px,
                   self.z_min_m, self.z_max_m))


class ProfileParams(object):
    """The two constants of Andert's inverse measurement model.

    p_min   free-space probability in front of the measured surface, 0 < p < 0.5
    eta     significance of ONE depth measurement; it sets the peak height
            through eta / (sqrt(2 pi) sigma_l), so a far measurement -- which
            has a wide sigma_l -- votes less, which is the point.
    """

    __slots__ = ("p_min", "eta")

    def __init__(self, p_min=0.35, eta=0.025):
        self.p_min = float(p_min)
        self.eta = float(eta)
        if not (0.0 < self.p_min < 0.5):
            raise ValueError("need 0 < p_min < 0.5, got %r" % self.p_min)
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive, got %r"
                             % self.eta)

    @classmethod
    def from_config(cls, cfg):
        m = cfg.mapping
        return cls(p_min=float(m.get("p_min", 0.35)),
                   eta=float(m.get("eta", 0.025)))

    def __repr__(self):
        return "ProfileParams(p_min=%.3f, eta=%.4f)" % (self.p_min, self.eta)


# ------------------------------------------------------- uncertainty chain

def disparity_to_depth_sigma(z, fx, baseline_m, sigma_disp_px):
    """sigma_Z = Z^2 sigma_s / (fx b), the first-order propagation of Z = fx b/s.

    Exactly the relation the stereo sensor's noise was drawn under, read the
    other way round: the sensor turns sigma_s into a depth perturbation, this
    turns it into the width the map should believe that depth to within.
    """
    z = np.asarray(z, dtype=np.float64)
    return z * z * float(sigma_disp_px) / (float(fx) * float(baseline_m))


def depth_sigma_to_line_sigma(sigma_z, ell_p, z):
    """sigma_l = (l_p / Z) sigma_Z.

    The profile is parameterised by Euclidean distance ALONG THE MEASUREMENT
    LINE, not by optical-axis depth, and the two differ by the ray's obliquity
    l_p / Z = sqrt(1 + x_n^2 + y_n^2). At the corner of an 85 deg horizontal
    field of view that is 1.35, so treating them as the same would understate
    the peak width by a third exactly where the geometry is worst.
    """
    sigma_z = np.asarray(sigma_z, dtype=np.float64)
    return sigma_z * (np.asarray(ell_p, dtype=np.float64)
                      / np.asarray(z, dtype=np.float64))


# ------------------------------------------------------------- the profile

def inverse_measurement_probability(ell, ell_p, sigma_l, p_min=0.35,
                                    eta=0.025):
    """P(l), Andert eqs. 8-10.

        P_base(l) = p_min for 0 < l <= l_p, 0.5 beyond
        P(l)      = P_base + (eta/(sqrt(2 pi) sigma_l) + 1/2 - P_base)
                             exp(-(l - l_p)^2 / (2 sigma_l^2))

    Continuous at l_p: both branches evaluate to 0.5 + eta/(sqrt(2 pi) sigma_l)
    there. Unimodal, rising below l_p and falling above, which is what makes
    the per-cell maximum a single clip-and-evaluate.

    Beyond the surface P -> 0.5, i.e. log-odds 0, i.e. NO EVIDENCE. Space
    behind an obstacle is not claimed free and is not claimed occupied.
    """
    ell = np.asarray(ell, dtype=np.float64)
    ell_p = np.asarray(ell_p, dtype=np.float64)
    sigma_l = np.asarray(sigma_l, dtype=np.float64)
    base = np.where(ell <= ell_p, p_min, 0.5)
    amp = eta / (np.sqrt(2.0 * np.pi) * sigma_l) + 0.5 - base
    gauss = np.exp(-0.5 * ((ell - ell_p) / sigma_l) ** 2)
    return base + amp * gauss


def inverse_measurement_logodds(ell, ell_p, sigma_l, p_min=0.35, eta=0.025):
    """L(l) = log(P/(1-P)). Negative is free, positive is occupied, 0 unknown."""
    p = inverse_measurement_probability(ell, ell_p, sigma_l, p_min, eta)
    # np.minimum(np.maximum(...)) RATHER THAN np.clip, and it is not taste.
    # numpy's `clip` is a Python wrapper -- fromnumeric.clip -> _wrapfunc ->
    # _methods._clip -> _clip_dep_is_scalar_nan, twice -- and this is called
    # once per cell of every ray of every frame. Profiled on the aircraft
    # against a real captured frame, the two clip sites in the walk were 19%
    # of the whole frame time, of which _clip_dep_is_scalar_nan alone was 8%.
    # The two forms agree bit for bit, NaN propagation included.
    p = np.minimum(np.maximum(p, _P_EPS), 1.0 - _P_EPS)
    return np.log(p / (1.0 - p))


def peak_probability(sigma_l, eta):
    """The value P reaches at l = l_p. Exceeding 1 means the model is invalid."""
    return 0.5 + eta / (np.sqrt(2.0 * np.pi) * np.asarray(sigma_l,
                                                          dtype=np.float64))


def validate_profile_params(profile: ProfileParams, stereo: StereoParams,
                            cell_m: float, p_peak_max: float = 0.995) -> float:
    """Refuse an eta whose peak would run off the top of the probability scale.

    The peak is 0.5 + eta/(sqrt(2 pi) sigma_l) and sigma_l is SMALLEST at the
    near end of the range, so that is where the model breaks first. Returns the
    smallest sigma_l the configuration can produce, which is also the floor the
    line walk will apply.

    Without this the clip in `inverse_measurement_logodds` would quietly turn
    every near measurement into the same saturated vote, and the model would
    stop being the paper's -- it would be a constant.
    """
    floor = _SIGMA_L_FLOOR_CELLS * float(cell_m)
    z0 = stereo.z_min_m
    sigma_z0 = disparity_to_depth_sigma(z0, stereo.fx, stereo.baseline_m,
                                        stereo.sigma_disp_px)
    sigma_l0 = float(max(sigma_z0, 0.0))          # obliquity >= 1, so this bounds
    sigma_min = max(sigma_l0, floor)
    p_peak = float(peak_probability(sigma_min, profile.eta))
    if p_peak > p_peak_max:
        raise ValueError(
            "eta = %.4f gives a peak probability of %.4f at the tightest "
            "sigma_l this configuration can produce (%.4f m, from z_min %.2f m "
            "and a %.3f m cell). The model requires P < 1; raise the cell size, "
            "raise sigma_disp_px, or use eta < %.4f."
            % (profile.eta, p_peak, sigma_min, z0, cell_m,
               (p_peak_max - 0.5) * np.sqrt(2.0 * np.pi) * sigma_min))
    return sigma_min


# ------------------------------------------------------------- line drawing

# Fraction of rays still alive below which `_walk_lines` compacts. 0.0
# disables compaction entirely, which is what `tests/test_perception_fast.py`
# uses to prove the compacted walk returns the same grid as the plain one.
_COMPACT_AT = 0.5


def _walk_lines(x0, z0, dx, dz, ell_p, sigma_l, ell_end, xz_norm,
                cell, grid_w, grid_h, p_min, eta, max_steps=None):
    """Amanatides-Woo grid traversal, vectorised over rays. Returns (L, hit).

    Walks the EXACT cells the projected measurement line crosses and, for each,
    the exact interval of line distances [a, b] it spends inside. The cell's
    contribution is L(clip(l_p, a, b)) -- the maximum of the profile over that
    interval, because the profile is unimodal.

    Every ray advances by ONE CELL per iteration, so the loop runs for as many
    iterations as the longest ray has cells and each iteration is one vectorised
    pass over all rays. Nothing is sampled twice and nothing is missed, which a
    fixed-step sampler cannot promise in either direction.

    `ell` is 3-D Euclidean distance along the measurement line; the walk itself
    is in the grid plane, where distance is `ell * xz_norm`. Dividing back is
    what keeps the profile parameterised the way the uncertainty was derived.
    """
    n = x0.shape[0]
    # -inf, NOT zero. `np.maximum.at` into a zero-initialised grid would clamp
    # every negative contribution away, and the negative contributions are the
    # free space -- the map would gain obstacles and never gain clear ground.
    # Untouched cells are restored to 0 (no evidence) from the `hit` mask at
    # the end, which is why that mask is tracked rather than inferred.
    L = np.full((grid_h, grid_w), -np.inf, dtype=np.float64)
    hit = np.zeros((grid_h, grid_w), dtype=bool)
    if n == 0:
        L[:] = 0.0
        return L, hit

    # Walk parameter is 2-D distance; convert the profile's 3-D distances once.
    t_p = ell_p * xz_norm
    t_end = ell_end * xz_norm

    ix = np.floor(x0 / cell).astype(np.int64)
    iz = np.floor(z0 / cell).astype(np.int64)
    step_x = np.where(dx >= 0.0, 1, -1).astype(np.int64)
    step_z = np.where(dz >= 0.0, 1, -1).astype(np.int64)

    big = np.float64(np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_delta_x = np.where(dx != 0.0, cell / np.abs(dx), big)
        t_delta_z = np.where(dz != 0.0, cell / np.abs(dz), big)
        bx = np.where(dx >= 0.0, (ix + 1) * cell, ix * cell)
        bz = np.where(dz >= 0.0, (iz + 1) * cell, iz * cell)
        t_max_x = np.where(dx != 0.0, (bx - x0) / dx, big)
        t_max_z = np.where(dz != 0.0, (bz - z0) / dz, big)

    # THE WALK STOPS WHERE THE GRID DOES, and this is decided ONCE rather than
    # tested every iteration.
    #
    # Nothing outside the grid is ever accumulated, but the loop used to carry
    # such a ray all the way to `t_end` anyway -- with a 101 deg field of view,
    # a 6 m range gate and a 7.2 x 5.2 m grid, most of the fan exits within a
    # metre or two and then walked another eighty cells outside the map. That
    # is pure cost, and it is the DOMINANT cost: the loop's price is per
    # ITERATION, not per ray. Measured on the aircraft, halving the image rows
    # moved the frame time by 0% and halving `rays_per_col` by 12%, because
    # each pass pays ~15 array ops and an unbuffered `np.maximum.at` however
    # few rays are left.
    #
    # Which is also why the test belongs HERE and not in the loop: an
    # equivalent per-iteration check ("index past an edge and stepping further
    # past it") was measured at 218 -> 248 ms, SLOWER, because it added nine
    # array ops to every one of ~360 passes to save a few of them. The ray/box
    # intersection is O(1) per ray, computed once, and costs the loop nothing.
    #
    # Standard slab clip against x in [0, grid_w*cell], z in [0, grid_h*cell].
    # A ray parallel to an axis and outside that slab never intersects; one
    # parallel and inside is unconstrained on that axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        gx, gz = grid_w * cell, grid_h * cell
        tx1 = np.where(dx != 0.0, (0.0 - x0) / dx, np.where(x0 >= 0.0, -big, big))
        tx2 = np.where(dx != 0.0, (gx - x0) / dx, np.where(x0 <= gx, big, -big))
        tz1 = np.where(dz != 0.0, (0.0 - z0) / dz, np.where(z0 >= 0.0, -big, big))
        tz2 = np.where(dz != 0.0, (gz - z0) / dz, np.where(z0 <= gz, big, -big))
        t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(tz1, tz2))
        t_out_box = np.minimum(np.maximum(tx1, tx2), np.maximum(tz1, tz2))
    # ONE CELL OF SLACK, deliberately. `t_out_box` is algebraically the same
    # boundary crossing the walk reaches by accumulating `t_delta`, but it is
    # not the same FLOATING-POINT number, and the last in-grid cell's interval
    # end feeds `clip(ell_p, a, b)`. Leaving a cell of room means the truncation
    # can only ever fall on cells that are already outside the grid, so the
    # accumulated grid is unchanged to the last bit -- verified over 298 ticks
    # of a recorded flight.
    t_end = np.minimum(t_end, t_out_box + cell)

    # logit(p_min), the value of every cell more than _NEAR_SIGMAS in front of
    # the surface. Computed once rather than derived per cell from a profile
    # that is flat there.
    logit_p_min = float(np.log(p_min / (1.0 - p_min)))

    t_cur = np.zeros(n, dtype=np.float64)
    # A ray that misses the grid entirely (t_out_box <= max(t_in, 0)) is dead
    # before the first pass rather than after eighty of them.
    alive = (t_end > 0.0) & (t_out_box > np.maximum(t_in, 0.0))
    if max_steps is None:
        # Every iteration crosses one cell boundary, so the bound is the number
        # of x-boundaries plus z-boundaries the longest ray can cross.
        span = float(np.nanmax(t_end)) if n else 0.0
        max_steps = int(np.ceil(2.0 * span / cell)) + 4

    # DEAD RAYS ARE DROPPED, NOT MASKED. `max_steps` is set by the LONGEST ray
    # while the median one is a third of that, so a loop that keeps every ray
    # in every pass spends most of its time on rays that already terminated:
    # measured on a `clutter` frame, 97142 rays x 41 steps = 3.98M ray-steps of
    # which 63% were dead. Compacting is exact -- a ray with t_cur >= t_end
    # contributes nothing to any later iteration, which is precisely what
    # `alive` already encoded -- and it shrinks every temporary in the pass,
    # so the saving is on memory traffic rather than on branches.
    #
    # The threshold exists because compaction costs a fancy-index copy of ~11
    # arrays. Doing it every iteration would pay that on passes where almost
    # nothing died; at half it pays once per halving, which is O(log) copies.
    ell_p = np.broadcast_to(np.asarray(ell_p, dtype=np.float64), (n,))
    sigma_l = np.broadcast_to(np.asarray(sigma_l, dtype=np.float64), (n,))
    xz_norm = np.broadcast_to(np.asarray(xz_norm, dtype=np.float64), (n,))

    for _ in range(int(max_steps)):
        n_alive = int(alive.sum())
        if n_alive == 0:
            break
        if n_alive < _COMPACT_AT * alive.shape[0]:
            k = np.flatnonzero(alive)
            (ix, iz, step_x, step_z, t_delta_x, t_delta_z, t_max_x, t_max_z,
             t_cur, t_end, ell_p, sigma_l, xz_norm) = (
                ix[k], iz[k], step_x[k], step_z[k], t_delta_x[k],
                t_delta_z[k], t_max_x[k], t_max_z[k], t_cur[k], t_end[k],
                ell_p[k], sigma_l[k], xz_norm[k])
            alive = np.ones(k.shape[0], dtype=bool)

        t_out = np.minimum(np.minimum(t_max_x, t_max_z), t_end)
        # The cell's interval in 3-D line distance.
        a = t_cur / xz_norm
        b = t_out / xz_norm
        # See the note in inverse_measurement_logodds: np.clip's wrapper is
        # too expensive to call once per cell. a <= b by construction here
        # (b is t_out/xz_norm and t_out >= t_cur), so the nesting order is the
        # same clamp.
        ell_star = np.minimum(np.maximum(ell_p, a), b)

        # THE FULL PROFILE IS EVALUATED ONLY NEAR THE PEAK. Everywhere else in
        # front of the surface it is the constant logit(p_min) -- see
        # _NEAR_SIGMAS. The subset is small (a ~6-cell window on a ~50-cell
        # walk), so this replaces eight full-width array ops with one compare,
        # one fill, and eight ops on a tenth of the rays.
        vals = np.full(ell_star.shape, logit_p_min, dtype=np.float64)
        near = ell_star > ell_p - _NEAR_SIGMAS * sigma_l
        if near.any():
            vals[near] = inverse_measurement_logodds(
                ell_star[near], ell_p[near], sigma_l[near], p_min, eta)

        ok = (alive & (ix >= 0) & (ix < grid_w) & (iz >= 0) & (iz < grid_h)
              & (t_out > t_cur))
        if ok.any():
            np.maximum.at(L, (iz[ok], ix[ok]), vals[ok])
            hit[iz[ok], ix[ok]] = True

        # Advance one cell along whichever axis boundary comes first.
        # The two masks are each used twice and the in-place forms write
        # through their own buffers, which is four array ops and four
        # temporaries per pass that used to be paid for nothing. At ~43 numpy
        # calls per pass and ~175 passes a frame, the dispatch overhead on
        # these small arrays IS the cost -- not the arithmetic.
        take_x = t_max_x <= t_max_z
        m_x = alive & take_x
        m_z = alive & ~take_x
        t_cur = np.where(alive, t_out, t_cur)
        np.add(ix, step_x, out=ix, where=m_x)
        np.add(iz, step_z, out=iz, where=m_z)
        np.add(t_max_x, t_delta_x, out=t_max_x, where=m_x)
        np.add(t_max_z, t_delta_z, out=t_max_z, where=m_z)
        alive &= t_cur < t_end

    L[~hit] = 0.0
    return L, hit


# ------------------------------------------------------------ frame builder

def build_frame_grid(depth, pose, stereo: StereoParams, cell, grid_w, grid_h,
                     plane_y, plane_tol, profile: Optional[ProfileParams] = None,
                     rotation=None, rays_per_col=0) -> Tuple[np.ndarray, np.ndarray]:
    """One frame of log-odds evidence, plus the mask of cells it touched.

    `depth` is optical-axis Z from the stereo sensor, NaN where invalid.
    `pose` is the renderer's (px, py, pz, roll, pitch, yaw) in GRID coordinates
    (already origin-shifted), py being the height. `rotation` overrides the
    camera-to-world matrix; by default it is the vendored renderer's, so the
    map and the depth image cannot disagree about which way the camera points.

    Returns (frame_logodds, touched). `touched` is the observation mask -- and
    it is returned rather than inferred from `frame != 0`, because a cell whose
    interval max lands exactly on P = 0.5 legitimately contributes 0.0 and
    would otherwise read as never seen.
    """
    profile = ProfileParams() if profile is None else profile
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be 2-D, got %d-D" % depth.ndim)
    px, py, pz, roll, pitch, yaw = (float(v) for v in pose)

    if rotation is None:
        rotation = camera_to_renderer(roll, pitch, yaw)
    T = np.asarray(rotation, dtype=np.float64)

    h, w = depth.shape
    # INVALID PIXELS PRODUCE NO LINE AT ALL. No endpoint is fabricated at
    # z_max, no ray is marked free, nothing is replaced by zero. A pixel the
    # stereo sensor could not measure leaves the map exactly as it was.
    valid = np.isfinite(depth) & (depth > 0.0)
    valid &= (depth >= stereo.z_min_m) & (depth <= stereo.z_max_m)
    if not valid.any():
        return (np.zeros((grid_h, grid_w), dtype=np.float64),
                np.zeros((grid_h, grid_w), dtype=bool))

    v_idx, u_idx = np.indices((h, w))
    x_n = (u_idx - stereo.cx) / stereo.fx
    y_n = (v_idx - stereo.cy) / stereo.fy
    obliquity = np.sqrt(1.0 + x_n * x_n + y_n * y_n)      # l_p / Z

    # p^C = Z K^-1 [u v 1]^T ; direction is that, normalised.
    rx_c, ry_c, rz_c = x_n / obliquity, y_n / obliquity, 1.0 / obliquity
    rx_w = T[0, 0] * rx_c + T[0, 1] * ry_c + T[0, 2] * rz_c
    ry_w = T[1, 0] * rx_c + T[1, 1] * ry_c + T[1, 2] * rz_c
    rz_w = T[2, 0] * rx_c + T[2, 1] * ry_c + T[2, 2] * rz_c

    # Invalid pixels are carried through the geometry with a placeholder so the
    # arrays stay rectangular, and are discarded by `valid` before anything is
    # drawn. Without the placeholder the NaNs propagate into sigma_l and raise
    # a divide warning for pixels that were never going to contribute.
    z_safe = np.where(valid, depth, 1.0)
    ell_p = z_safe * obliquity                             # 3-D distance
    # HEIGHT BAND, preserved from the existing mapper: an endpoint contributes
    # only if it lands in the horizontal slab the planar grid represents.
    y_end = py + ell_p * ry_w
    in_band = np.abs(y_end - plane_y) < plane_tol

    xz_norm = np.sqrt(rx_w * rx_w + rz_w * rz_w)
    valid &= in_band & (xz_norm > 1e-3)

    # THIN THE BAND. A column of the image is ONE horizontal bearing -- the
    # bearing depends on u alone -- so every row the band keeps in that column
    # redraws the same line into the same cells. Measured on a `clutter` frame
    # at z0 = 1.0 m: the band keeps 170 of 360 rows per column, 97142 rays for
    # a 14484-cell grid (6.7 rays per cell), and across the busiest column the
    # depths differ by 8.2 mm against a 50 mm cell. The redundancy is the whole
    # cost of the mapper: 331 ms per frame, against the ~100 ms that fusing at
    # map_hz 10 in real time allows.
    #
    # WHY NOT JUST TAKE THE MIDDLE ROWS. Because the band is only centred on cy
    # while the camera is level. It sits at the plane height by construction,
    # so a ray through the principal row stays in the slab -- but tilt moves
    # the horizon off cy by `tilt * fy` rows, which at the 14 deg measured
    # during a full-authority brake is 86 rows, far outside any fixed window.
    # Keeping the PHYSICAL band test and thinning what survives is right at
    # every attitude; a fixed row window is right only in hover.
    #
    # This is a SUBSAMPLE, not a merge: every ray kept is the ray the full
    # version would have walked, with its own depth, obliquity and sigma_l.
    # Nothing is averaged, so the only question is whether the discarded rays
    # carried information the kept ones do not -- see tests/test_row_thin.py.
    if rays_per_col and valid.any():
        rank = np.cumsum(valid, axis=0) - 1          # index within the column
        cnt = valid.sum(axis=0)                       # rays this column has
        stride = np.maximum(1, np.ceil(cnt / float(rays_per_col))).astype(int)
        valid &= (rank % stride[None, :] == 0)

    if not valid.any():
        return (np.zeros((grid_h, grid_w), dtype=np.float64),
                np.zeros((grid_h, grid_w), dtype=bool))

    sigma_z = disparity_to_depth_sigma(z_safe, stereo.fx, stereo.baseline_m,
                                       stereo.sigma_disp_px)
    sigma_l = depth_sigma_to_line_sigma(sigma_z, ell_p, z_safe)
    sigma_l = np.maximum(sigma_l, _SIGMA_L_FLOOR_CELLS * cell)

    # Draw only as far as the profile carries information.
    ell_end = np.minimum(ell_p + _TAIL_SIGMAS * sigma_l,
                         stereo.z_max_m * obliquity)

    sel = valid
    return _walk_lines(
        np.full(int(sel.sum()), px), np.full(int(sel.sum()), pz),
        (rx_w / xz_norm)[sel], (rz_w / xz_norm)[sel],
        ell_p[sel], sigma_l[sel], ell_end[sel], xz_norm[sel],
        float(cell), int(grid_w), int(grid_h), profile.p_min, profile.eta)
