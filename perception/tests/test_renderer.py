#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The simulator's depth camera: intrinsics, frame conventions, ray casting.

    python tests/test_renderer.py

Was `planar_sim/perception/renderer.py:_main`, moved out so the module that
every episode imports carries only the code that flies. Nothing in the checks
changed.

WHAT IS PINNED. The intrinsics scale with the raster and stay consistent with
`StereoParams`; `camera_to_renderer` composes the two frame conventions in the
right order; `render_depth` returns optical-axis Z and honours the near/far
clip; and the pose is the renderer's 6-tuple, never the vendored 3-tuple that
filled altitude from a module global.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception.renderer import (
    CameraModel,
    NATIVE_F,
    NATIVE_H,
    NATIVE_W,
    camera_to_renderer,
    render_depth,
)


def _wall(z, half=8.0):
    """One fronto-parallel segment at renderer z, spanning +-half in x."""
    return [((-half, z), (half, z))]


def _main():
    ok = [0]

    def check(name, cond, detail=""):
        print("  %-58s %s   %s" % (name, "PASS" if cond else "FAIL", detail))
        assert cond, name
        ok[0] += 1

    print("camera model")
    cam = CameraModel.from_scale(1.0, z_min_m=0.05, z_max_m=60.0)
    check("scale 1.0 is the native raster",
          (cam.width, cam.height, cam.fx) == (NATIVE_W, NATIVE_H, NATIVE_F),
          repr(cam))
    half = CameraModel.from_scale(0.5, z_min_m=0.05, z_max_m=60.0)
    check("halving the raster holds the field of view",
          abs(half.hfov_rad - cam.hfov_rad) < 1e-12,
          "%.2f deg at both scales" % np.degrees(cam.hfov_rad))
    # The bug this class exists to make impossible: two roundings of one raster.
    worst = max(range(20, 101),
                key=lambda p: abs(CameraModel.from_scale(p / 100.0, 0.05, 60.0).cx
                                  - NATIVE_W * (p / 100.0) / 2.0))
    c = CameraModel.from_scale(worst / 100.0, 0.05, 60.0)
    check("the principal point is the centre of the raster actually built",
          c.cx == c.width / 2.0 and c.cy == c.height / 2.0,
          "worst scale %.2f: cx %.1f vs %.1f if rounded separately"
          % (worst / 100.0, c.cx, NATIVE_W * (worst / 100.0) / 2.0))
    n_bad = 0
    for kw in (dict(width=1), dict(fx=0.0), dict(fy=-1.0), dict(cx=np.nan),
               dict(z_min_m=0.0), dict(z_max_m=0.01)):
        args = dict(width=64, height=36, fx=35.0, fy=35.0, cx=32.0, cy=18.0,
                    z_min_m=0.3, z_max_m=6.0)
        args.update(kw)
        try:
            CameraModel(**args)
        except ValueError:
            n_bad += 1
        else:
            print("    NOT REJECTED: %r" % kw)
    check("malformed intrinsics raise ValueError", n_bad == 6, "%d/6" % n_bad)

    print("\nframe convention")
    T = camera_to_renderer(0.0, 0.0, 0.0)
    check("at yaw 0 the camera z axis is renderer +z",
          np.allclose(T[:, 2], [0, 0, 1], atol=1e-6), str(T[:, 2]))
    check("and the camera x axis is renderer +x",
          np.allclose(T[:, 0], [1, 0, 0], atol=1e-6), str(T[:, 0]))
    check("and the camera y axis (down) is renderer -y",
          np.allclose(T[:, 1], [0, -1, 0], atol=1e-6), str(T[:, 1]))
    Th = camera_to_renderer(0.0, 0.0, 0.5 * np.pi)
    check("a yaw of pi/2 swings the camera x axis onto renderer +z",
          np.allclose(Th[:, 0], [0, 0, 1], atol=1e-6), str(Th[:, 0]))
    check("the rotation is orthonormal",
          np.allclose(T.T @ T, np.eye(3), atol=1e-6))
    # Every consumer indexes T as float32; widening it silently would change
    # every rendered depth, so the dtype is asserted, not assumed.
    check("and stays float32, as the vendored version was",
          T.dtype == np.float32, str(T.dtype))

    print("\nray casting")
    cam = CameraModel.from_scale(1.0, z_min_m=0.05, z_max_m=60.0)
    z_wall = 3.0
    img = render_depth(_wall(z_wall), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), cam, 4.0)
    check("the depth image has the camera's raster",
          img.shape == (cam.height, cam.width), str(img.shape))
    row = img[cam.height // 2]
    fin = np.isfinite(row)
    x_n = (np.arange(cam.width)[fin] - cam.cx) / cam.fx
    spread = float(row[fin].max() - row[fin].min())
    check("it returns optical-axis Z, not ray length",
          spread < 1e-5,
          "row spread %.2e m; ray length would spread %.3f m"
          % (spread, z_wall * (np.sqrt(1 + x_n.max() ** 2) - 1.0)))
    check("and the constant equals the wall distance",
          abs(float(np.median(row[fin])) - z_wall) < 1e-5,
          "%.6f m vs %.1f m" % (float(np.median(row[fin])), z_wall))
    check("a miss is +inf, never z_max or NaN",
          not np.isnan(img).any() and np.isinf(render_depth(
              [], (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
              CameraModel(64, 36, 35.0, 35.0, 32.0, 18.0, 0.3, 6.0), 1.0
          )[:18]).all(),
          "rays above the horizon with no wall hit nothing")
    near = render_depth(_wall(0.5), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                        CameraModel(64, 36, 35.0, 35.0, 32.0, 18.0, 1.0, 6.0),
                        4.0)
    check("a hit nearer than z_min is dropped, not clamped",
          np.isinf(near[:18]).all(), "the wall at 0.5 m with z_min 1.0 m")
    short = CameraModel.from_scale(1.0, z_min_m=0.05, z_max_m=6.0)
    far = render_depth(_wall(9.0), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), short, 4.0)
    check("and a hit past z_max likewise",
          np.isinf(far[short.height // 2]).all(),
          "the wall at 9 m with z_max %.0f m" % short.z_max_m)

    # The wall is 4 m tall and the camera is at 1 m, so a level ray must hit
    # it; raise the obstacle height cut below the camera and the wall vanishes.
    low = render_depth(_wall(z_wall), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0), cam, 0.5)
    check("obstacle_height bounds the extrusion",
          np.isinf(low[cam.height // 2]).all() and fin.any(),
          "a 0.5 m wall is invisible to a camera at 1.0 m looking level")

    print("\nposes")
    yawed = render_depth(_wall(z_wall), (0.0, 1.0, 0.0, 0.0, 0.0, 0.5 * np.pi),
                         cam, 4.0)
    check("yaw turns the camera away from the wall",
          np.isfinite(yawed).mean() < np.isfinite(img).mean(),
          "%.0f%% -> %.0f%% finite" % (100 * np.isfinite(img).mean(),
                                       100 * np.isfinite(yawed).mean()))
    n_bad = 0
    for bad in ((0.0, 0.0, 0.0), (0.0, 0.0), tuple(range(7))):
        try:
            render_depth([], bad, cam, 1.0)
        except ValueError:
            n_bad += 1
        else:
            print("    NOT REJECTED: len=%d" % len(bad))
    check("only the 6-tuple pose is accepted", n_bad == 3,
          "the vendored 3-tuple filled altitude from a global; it is gone")

    print("\n%d checks passed" % ok[0])


if __name__ == "__main__":
    _main()
