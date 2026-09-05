#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The simulator's depth camera: intrinsics, frame conventions, ray casting.

    python tests/test_renderer.py                    # the tests for this module

Brought in from `vendor/perception/depth_to_grid_2d.py` (sles_uav_sim), which
was the companion code to Andert, "Drawing Stereo Disparity Images into
Occupancy Grids" (IROS 2009). Only three things in that file were still on a
live call path -- the ray caster, the camera-to-renderer rotation, and the
camera constants -- and they are here, unchanged in arithmetic. What was left
behind was its own mapping stack (`log_odds`, `build_frame_grid`, `fuse_frames`,
`postprocess_blur`) and its demo plotting, all superseded by
`perception/inverse_sensor.py`; keeping them would have kept matplotlib and
scipy on the import path of every episode that renders a frame.

WHY THE PARAMETERS ARE ARGUMENTS NOW
The vendored file kept its geometry in module-level globals and `AndertMapper`
reached in and rewrote them (`_configure_module`) before every episode. That was
the right call while the file was vendored-unmodified, and the wrong shape to
keep once it is ours: it made the camera a piece of process-wide mutable state,
so two mappers at different `render_scale` could not coexist, a test that
changed a global corrupted every later import, and the raster lived in one place
(`d2.WIDTH`) while the intrinsics describing that same raster lived in another
(`StereoParams`) with nothing keeping them equal. `CameraModel` is that one
place, and `StereoParams` extends it rather than restating it.

FRAMES
Two conventions meet here and neither is planar_sim's.

  camera   OpenCV: x right, y down, z forward along the optical axis.
  renderer y up, z forward, x right. The ground plane is y = 0 and obstacles
           are 2D segments in the x-z plane extruded from y = 0 to
           `obstacle_height`.

planar_sim's own world is z up with psi measured from +x. The conversion
(origin shift and `psi - pi/2`) is applied in exactly one place,
`AndertMapper._renderer_pose`, so this convention never escapes the perception
package.

FLOAT32 IS DELIBERATE
`camera_to_renderer` returns float32, as the vendored version did, which makes
the whole ray cast float32. Widening it would change every depth by ~1e-7
relative -- far below `sigma_disp_px`, but enough to make recorded episodes
non-reproducible. The move is a move; changing the numerics is a separate
decision.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "NATIVE_W", "NATIVE_H", "NATIVE_F", "NATIVE_B",
    "CameraModel",
    "camera_to_renderer",
    "render_depth",
]


# THE CAMERA'S OWN RASTER. 640 x 360 is what the ZED 2i delivers to this
# system, so it is the reference `mapping.render_scale` is a fraction OF, and
# scale 1.0 is the real sensor rather than an upsampled ideal.
#
# The vendored module described the same camera in HD720 (1280 x 720, F 700),
# and F is scaled with the width so the field of view is identical:
# 2*atan(1280/(2*700)) == 2*atan(640/(2*350)) == 84.9 deg. Changing the raster
# changes the angular sampling, never what is in frame.
NATIVE_W, NATIVE_H = 640, 360
NATIVE_F = 350.0

# Stereo baseline [m]. A hardware fact about the ZED 2i, not a tuning knob, so
# it sits with the other camera constants instead of in config.yaml.
NATIVE_B = 0.12


class CameraModel(object):
    """A pinhole camera's raster, intrinsics and usable depth range.

    `StereoParams` extends this with the baseline and the matcher's disparity
    error. Both the ray caster and the inverse sensor read the same object, so
    the image the renderer produces and the geometry the mapper back-projects
    with cannot drift apart.
    """

    __slots__ = ("width", "height", "fx", "fy", "cx", "cy",
                 "z_min_m", "z_max_m")

    def __init__(self, width, height, fx, fy, cx, cy, z_min_m, z_max_m):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)
        if self.width < 2 or self.height < 2:
            raise ValueError("raster must be at least 2x2, got %dx%d"
                             % (self.width, self.height))
        for name in ("fx", "fy"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError("%s must be finite and positive, got %r"
                                 % (name, v))
        for name in ("cx", "cy"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError("%s must be finite" % name)
        if not (0.0 < self.z_min_m < self.z_max_m):
            raise ValueError("need 0 < z_min_m < z_max_m, got %r and %r"
                             % (self.z_min_m, self.z_max_m))

    @staticmethod
    def raster_for_scale(render_scale) -> Tuple[int, int]:
        """The rendered raster at `mapping.render_scale`, rounded once.

        THE ROUNDING HAPPENS HERE AND NOWHERE ELSE. It used to happen twice --
        `int(round(NATIVE_W * scale))` for the renderer's width and
        `NATIVE_W * scale / 2` for the principal point -- which disagree by up
        to half a pixel at any scale that is not a power of two, putting the
        mapper's optical centre off the image the renderer actually drew.
        """
        scale = float(render_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("render_scale must be finite and positive, "
                             "got %r" % render_scale)
        return (max(16, int(round(NATIVE_W * scale))),
                max(9, int(round(NATIVE_H * scale))))

    @classmethod
    def from_scale(cls, render_scale, z_min_m, z_max_m, **kw):
        """The native camera rendered at a fraction of its raster.

        The focal length scales with the width, holding the field of view; the
        principal point is the centre of the raster that was actually built.
        """
        w, h = cls.raster_for_scale(render_scale)
        return cls(width=w, height=h,
                   fx=NATIVE_F * float(render_scale),
                   fy=NATIVE_F * float(render_scale),
                   cx=w / 2.0, cy=h / 2.0,
                   z_min_m=z_min_m, z_max_m=z_max_m, **kw)

    @property
    def hfov_rad(self) -> float:
        return float(2.0 * np.arctan(self.width / (2.0 * self.fx)))

    def __repr__(self):
        return ("CameraModel(%dx%d, fx=%.1f, cx=%.1f, cy=%.1f, z=[%.2f, %.2f])"
                % (self.width, self.height, self.fx, self.cx, self.cy,
                   self.z_min_m, self.z_max_m))


def camera_to_renderer(roll, pitch, yaw) -> np.ndarray:
    """3x3 matrix taking a camera-frame ray direction to renderer-world.

    Composition (leftmost is applied last):

        camera frame  --M_cb-->  body-quad frame (z up, y forward, x right)
                      --R_zyx--> quad-world frame (ZYX intrinsic Euler)
                      --M_rq-->  renderer-world frame (y up, z forward)

    Its FIRST COLUMN is the camera's x axis expressed in renderer coordinates,
    which is what `right_camera_pose` offsets the second camera along -- so the
    stereo baseline rotates with the vehicle instead of being pinned to world
    x. Worked through for roll = pitch = 0:

        camera x -> renderer ( cos yaw, 0, sin yaw)
        camera z -> renderer (-sin yaw, 0, cos yaw)

    float32 on purpose; see the module docstring.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [    -sp,                 cp * sr,                 cp * cr],
    ], dtype=np.float32)
    # Camera-to-body axis swap: body has z up, y forward, x right.
    # x_c -> +x_body ; z_c -> +y_body ; y_c -> -z_body
    M_cb = np.array([[1, 0,  0],
                     [0, 0,  1],
                     [0, -1, 0]], dtype=np.float32)
    # Quad-world (z up, y forward) to renderer-world (y up, z forward).
    M_rq = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]], dtype=np.float32)
    return (M_rq @ R @ M_cb).astype(np.float32)


_RAY_CACHE = {}


def _camera_rays(cam: CameraModel):
    """Unit ray directions in the CAMERA frame, (rx, ry, rz) float32.

    Cached on the intrinsics because nothing here depends on the pose, and
    `render_depth` is called 15 times a second for the whole of a run. The
    arrays are returned, not copied: callers must treat them as read-only,
    which `render_depth` does -- it only ever reads them into new temporaries.
    """
    key = (cam.width, cam.height, float(cam.fx), float(cam.fy),
           float(cam.cx), float(cam.cy))
    hit = _RAY_CACHE.get(key)
    if hit is None:
        u = np.arange(cam.width, dtype=np.float32)
        v = np.arange(cam.height, dtype=np.float32)
        U, V = np.meshgrid(u, v)
        x_dir = (U - np.float32(cam.cx)) / np.float32(cam.fx)
        y_dir = (V - np.float32(cam.cy)) / np.float32(cam.fy)
        z_dir = np.ones_like(x_dir)
        norm = np.sqrt(x_dir * x_dir + y_dir * y_dir + z_dir * z_dir)
        hit = (x_dir / norm, y_dir / norm, z_dir / norm)
        _RAY_CACHE[key] = hit
    return hit


def render_depth(segments: Sequence, pose, cam: CameraModel,
                 obstacle_height: float) -> np.ndarray:
    """Cast one ray per pixel and return optical-axis Z-depth [m].

    `pose` is the renderer's 6-tuple (px, py, pz, roll, pitch, yaw) with py the
    camera altitude above the ground plane -- ALREADY IN RENDERER COORDINATES.
    `AndertMapper._renderer_pose` is what builds it; nothing here converts.

    `segments` are 2D top-down line segments ((x1, z1), (x2, z2)) in renderer
    metres, extruded vertically from y = 0 to `obstacle_height`, plus an
    implicit ground plane at y = 0.

    Returns (cam.height, cam.width) float32. A pixel with no hit, or a hit
    outside [z_min_m, z_max_m], is +inf -- NOT NaN and NOT z_max. Converting
    that to a distance is what `simulated_stereo_depth` does, deliberately, by
    marking it invalid rather than claiming open space.

    Cost is O(len(segments) * width * height): every segment is intersected
    against every ray with no acceleration structure. It is the dominant term
    in the perception update -- see `mapping.render_scale` if that matters.

    THE LOOP IS MEMORY-BOUND, NOT ARITHMETIC-BOUND, and three changes made it
    5.9x faster (273.5 -> 46.8 ms at 640x360 with 44 segments) without moving
    a single hit:

      * float32 throughout. `np.arange` gives int64 and dividing promotes to
        float64, so every one of the ~8 temporaries per segment was 1.8 MB
        instead of 0.9. Halving the traffic is most of the win.
      * the CAMERA-frame ray field is cached. `meshgrid` and the normalisation
        depend only on (width, height, fx, fy, cx, cy) -- not on the pose --
        so rebuilding them 15 times a second was pure waste.
      * `np.minimum(..., out=t_min)` instead of `t_min = np.minimum(...)`,
        and one `errstate` around the whole loop rather than one per segment.

    Verified against the float64 version on a `clutter` room: the finite/inf
    mask is IDENTICAL and the largest depth difference is 2.7e-7 m, which is
    float32 rounding and ~5 orders of magnitude below `sigma_disp_px`'s effect.
    `tests/test_renderer_fast.py` pins both properties.
    """
    px, py, pz, roll, pitch, yaw = _normalise_pose(pose)
    T = camera_to_renderer(roll, pitch, yaw).astype(np.float32)
    oh = np.float32(obstacle_height)
    px32, py32, pz32 = np.float32(px), np.float32(py), np.float32(pz)

    rx_c, ry_c, rz_c = _camera_rays(cam)

    # Rotate camera-frame rays into renderer-world frame.
    rx_w = T[0, 0] * rx_c + T[0, 1] * ry_c + T[0, 2] * rz_c
    ry_w = T[1, 0] * rx_c + T[1, 1] * ry_c + T[1, 2] * rz_c
    rz_w = T[2, 0] * rx_c + T[2, 1] * ry_c + T[2, 2] * rz_c

    t_min = np.full(rx_c.shape, np.inf, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Ground plane at world y = 0: py + t * ry_w = 0
        t_ground = -py32 / ry_w
        np.minimum(t_min, np.where((ry_w < -1e-9) & (t_ground > 0),
                                   t_ground, np.inf), out=t_min)

        # Walls -- 2D segments extruded along world y from 0 to
        # `obstacle_height`.
        for (p1, p2) in segments:
            p1x = np.float32(p1[0])
            p1z = np.float32(p1[1])
            sx = np.float32(p2[0]) - p1x
            sz = np.float32(p2[1]) - p1z
            ax = p1x - px32
            az = p1z - pz32
            denom = rx_w * sz - rz_w * sx
            t = (ax * sz - az * sx) / denom
            ss = (ax * rz_w - az * rx_w) / denom
            y_hit = py32 + t * ry_w
            valid = ((np.abs(denom) > 1e-12) & (t > 1e-6) &
                     (ss >= 0) & (ss <= 1) &
                     (y_hit >= 0) & (y_hit <= oh))
            np.minimum(t_min, np.where(valid, t, np.inf), out=t_min)

    # OPTICAL-AXIS Z, NOT RAY LENGTH. t is Euclidean distance along a unit ray;
    # multiplying by that ray's camera-frame z component projects it onto the
    # optical axis, which is what a depth camera reports and what
    # `simulated_stereo_depth` assumes when it forms d = fx*B/Z.
    z_c = t_min * rz_c
    z_c[z_c < cam.z_min_m] = np.inf
    z_c[z_c > cam.z_max_m] = np.inf
    return z_c


def _normalise_pose(pose) -> Tuple[float, float, float, float, float, float]:
    """(px, py, pz, roll, pitch, yaw) with py the camera altitude in renderer y.

    The vendored version also accepted a 3-tuple (px, pz, yaw) that filled the
    altitude from a module global. That global is gone: this simulator always
    knows the camera's height, and defaulting it silently is how a slice ends
    up rendered from the wrong plane.
    """
    if len(pose) != 6:
        raise ValueError("pose must be the renderer's 6-tuple "
                         "(px, py, pz, roll, pitch, yaw), got len=%d"
                         % len(pose))
    return tuple(float(v) for v in pose)
