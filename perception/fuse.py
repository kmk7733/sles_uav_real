# -*- coding: utf-8 -*-
"""One depth frame -> one Andert log-odds increment, folded into the grid.

THIS FUNCTION IS THE FLYING CODE, AND THE OFFLINE VALIDATION CALLS IT TOO.
`depth_to_grid_andert.py` (live, ROS) and `replay_bag.py` (offline, against a
recording) both go through `fuse_frame`. A replay that agrees with the
simulator's `experiments/bag_grid_map.py` while the NODE disagrees with both is
worse than no replay at all, so there is exactly one implementation and the two
callers differ only in where the depth image and the pose came from.

WHAT THIS MODULE OWNS
    decode_depth              32FC1 -> float32, NaN/inf preserved
    stereo_from_camera_info   the REAL intrinsics, never the simulator's
    fuse_frame                rotation, origin shift, build_frame_grid, integrate

THE ROTATION IS BUILT FROM A MATRIX AND NEVER FROM EULER ANGLES.
`inverse_sensor.build_frame_grid` will call `camera_to_renderer(roll, pitch,
yaw)` if no rotation is supplied, and that path is WRONG for a pose that came
from a body-frame quaternion: `camera_to_renderer` applies its ZYX composition
to a frame whose axes are (x right, y forward, z up), while a body ZYX
decomposition means (x forward, y left, z up). Working the two against each
other,

    what the renderer applies    Rz(psi) Rx(pitch) Ry(-roll)
    what the angles mean         Rz(psi) Ry(pitch) Rx(roll)

-- roll and pitch are EXCHANGED and a sign flips. At yaw alone the two agree
exactly, which is why nothing notices in level flight; at the 4.2 deg tilt of
the 20260730 recording they point the boresight 5.2 deg apart, which is 0.46 m
at 5 m. A TF quaternion goes to a rotation matrix in one step and the ambiguity
never arises, so that is what is done here. Do not "simplify" this back into
Euler angles.
"""

import numpy as np

from .inverse_sensor import ProfileParams, StereoParams, build_frame_grid

__all__ = ["P_WORLD_RENDER", "ProfileParams", "StereoParams",
           "decode_depth", "stereo_from_camera_info", "default_sigma_disp_px",
           "ray_spacing_limit", "band_rows_per_col", "fuse_frame"]


# World (x, y, z up) -> renderer (x, y up, z forward). The renderer's (x, z)
# plane is our horizontal plane and its y is our z; the segment/grid layout
# passes through unchanged and only the axis order moves.
P_WORLD_RENDER = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0]])


def decode_depth(msg):
    """sensor_msgs/Image 32FC1 -> (h, w) float32, honouring row stride.

    Verbatim from `experiments/bag_grid_map.py:_decode_depth`.

    NaN (no stereo match) and +-inf (out of range) are PRESERVED.
    `build_frame_grid` drops exactly those pixels and draws no line for them,
    which is the correct reading: a pixel the matcher could not measure is not
    evidence of free space and is not evidence of a surface. Substituting a
    number here would invent one or the other -- and substituting z_max in
    particular would say "nothing within range" exactly where a nearer surface
    is hiding something.
    """
    if msg.encoding != "32FC1":
        raise ValueError("expected 32FC1 depth, got %r "
                         "(is openni_depth_mode on?)" % (msg.encoding,))
    if msg.is_bigendian:
        raise ValueError("big-endian depth not handled")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == w * 4:
        return buf.view(np.float32).reshape(h, w)
    return buf.reshape(h, step)[:, :w * 4].copy().view(np.float32).reshape(h, w)


def default_sigma_disp_px(fx):
    """The simulator's disparity sigma, at THIS camera's focal length.

    `StereoParams.default_sigma_disp_px` is `0.5 * NATIVE_F * scale / 700`,
    where 700 is the HD720 reference focal length the 0.5 px figure was quoted
    at. Written against a real fx it is `0.5 * fx / 700`, which is what
    `bag_grid_map.build_stereo` uses -- so the two agree by construction and
    the cell-for-cell comparison against it is exact rather than approximate.

    It is a SIMULATOR-calibrated number and this camera runs
    `depth/quality: PERFORMANCE`. It is the starting point, not the answer; see
    the sigma sweep in the port plan.
    """
    return 0.5 * float(fx) / 700.0


def stereo_from_camera_info(K, width, height, decim=1, row_decim=None,
                            baseline_m=0.12, sigma_disp_px=None,
                            z_min_m=0.4, z_max_m=6.0):
    """StereoParams from the camera's OWN camera_info, decimated with the image.

    THE SIMULATOR'S CONSTANTS MUST NOT BE USED HERE. `NATIVE_F = 350` at
    640x360 is an 84.9 deg HFOV; this ZED 2i reports fx = 261.4, cx = 327.4,
    cy = 176.1 at the same raster -- 101.5 deg, and a principal point that is
    NOT the raster centre. Back-projecting with the simulated fx would squeeze
    the scene into an 85 deg wedge and bend every straight wall.

    THE TWO AXES ARE NOT THE SAME KIND OF THING, so they get separate strides.
    A COLUMN IS ONE BEARING: the horizontal direction of a ray depends on the
    image column alone, and the grid's angular resolution is exactly the column
    spacing -- thin columns and distant obstacles come back with gaps between
    the rays that would have found them. ROWS only decide which rays survive
    the height-band test, and every surviving row in one column redraws THE
    SAME LINE into the same cells (which is why `rays_per_col` exists at all).
    So `decim` strides columns and `row_decim` strides rows, and rows can be
    thinned much harder: it costs nothing but how finely the band is sampled.

    The band keeps roughly `2 * plane_tol * fy / Z` rows per column, so a row
    stride is safe while that stays above `rays_per_col` at the ranges that
    matter -- at plane_tol 0.25 and fy 65 that is 16 rows at 2 m and 8 at 4 m.

    `build_frame_grid` back-projects with fx and fy separately, so anisotropic
    intrinsics need nothing special; `NATIVE_B` does carry over, the measured
    baseline of this camera being 119.97 mm.
    """
    dx = int(decim)
    dy = dx if row_decim is None else int(row_decim)
    fx, fy = float(K[0]) / dx, float(K[4]) / dy
    cx, cy = float(K[2]) / dx, float(K[5]) / dy
    w, h = int(width) // dx, int(height) // dy
    if sigma_disp_px is None:
        # DISPARITY IS MEASURED ALONG A ROW, so its sigma scales with fx and
        # with fx alone. Deriving it from fy would make a row stride change the
        # modelled depth uncertainty, which is not a thing a row stride does.
        sigma_disp_px = default_sigma_disp_px(fx)
    return StereoParams(fx, fy, cx, cy,
                        baseline_m=float(baseline_m),
                        sigma_disp_px=float(sigma_disp_px),
                        z_min_m=float(z_min_m), z_max_m=float(z_max_m),
                        width=w, height=h)


def band_rows_per_col(fy, plane_tol, z):
    """How many image rows the height band keeps in one column, at range `z`.

    The check that a row stride has not gone too far: this must stay at or
    above `rays_per_col` over the range that matters, or the cap stops binding
    and the thinning starts costing coverage instead of time.
    """
    return 2.0 * float(plane_tol) * float(fy) / max(float(z), 1e-6)


def ray_spacing_limit(resolution, fx):
    """The range past which adjacent rays skip cells, `res * fx`.

    At range Z adjacent image columns are Z/fx metres apart, so beyond res*fx a
    distant wall is drawn as a dotted line rather than a wall. The simulator
    never meets this (fx 350, res 0.05 -> 17.5 m); a decimated real camera does
    (fx 130.7 at decim 2 -> 6.5 m). A caller that sets z_max above this is
    asking for perforated far walls.
    """
    return float(resolution) * float(fx)


def fuse_frame(depth, cam_xyz, R_world_opt, stereo, profile, grid,
               plane_height, plane_tol, rays_per_col=0):
    """Fold ONE depth image into `grid`. Returns (frame, touched).

    depth        (h, w) float32, optical-axis Z in metres, NaN/inf preserved
    cam_xyz      camera optical centre in WORLD coordinates, (x, y, z)
    R_world_opt  3x3 rotation, camera optical frame -> world. Straight from a
                 TF quaternion; see the module docstring on why not Euler.
    grid         AndertGrid -- supplies the geometry AND is updated in place

    The grid the renderer walks is zero-based (`andert.py`, note 1: "the
    renderer's grid starts at (0, 0) and has no origin offset"), so the
    horizontal coordinates are shifted by -min_x / -min_y here and the caller
    hands the real origin to nav_msgs. ALTITUDE IS NOT SHIFTED: `plane_height`
    is an absolute world height and the two have to be measured from the same
    zero.
    """
    # A decimated depth image is a STRIDED VIEW of the full raster, and every
    # numpy pass over it then pays for the stride: measured on the Xavier,
    # 134.3 -> 127.9 ms at decim 2. The arrays are bit-identical either way
    # (checked, frame and touched both), so this is layout and nothing else.
    # It also detaches the array from the ROS message buffer it was a view of.
    depth = np.ascontiguousarray(depth)
    R = np.asarray(P_WORLD_RENDER.dot(np.asarray(R_world_opt, dtype=float)),
                   dtype=np.float32)
    pose = (float(cam_xyz[0]) - grid.min_x,   # renderer x
            float(cam_xyz[2]),                # renderer y == world z, absolute
            float(cam_xyz[1]) - grid.min_y,   # renderer z
            0.0, 0.0, 0.0)                    # unread: rotation is supplied
    frame, touched = build_frame_grid(
        depth, pose, stereo, grid.res, grid.nx, grid.ny,
        float(plane_height), float(plane_tol), profile,
        rotation=R, rays_per_col=int(rays_per_col))
    grid.integrate(frame, touched)
    return frame, touched
