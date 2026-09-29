"""GPU evaluation of FrontierMPPI's K-sample batch (planner/haa/cuda/mppi_batch.cu).

`enable_cuda(planner)` attaches a backend that PlanarMPPI.plan uses for the
clip -> capped rollout -> reject step and for the per-node lookups of
FrontierMPPI._cost. Everything else -- the RNG stream, the cost sums and their
order, weighting, the damped update, swept validation and all fallbacks --
stays in the original NumPy code. Positions, velocities, inputs, validity,
distances and clearances are computed with the same double-precision operation
order (nvcc -fmad=false); only the heading wrap (sin/cos/atan2) may differ from
libm by an ulp. The yaw term (w_yaw > 0) needs no kernel support: it is a
function of the rolled-out X (heading, position, velocity) and is summed in
FrontierMPPI._cost_from_parts like every other term. Anything the kernel does
not reproduce exactly (other dynamics, no occupancy grid) falls back to the
NumPy path for that call.

Opt-in only: the simulator and existing nodes never enable it.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

from planner.dynamics import PlanarDynamics, _norm_last_axis, _TOL
from planner.haa.capped import CappedDynamics
from planner.haa.geodesic import REACH_PENALTY_M
from planner.types import S_POS

LIB = Path(__file__).resolve().parent / "cuda" / "libmppi_batch.so"
CLEAR, OCCUPIED, UNKNOWN = 0, 1, -1


class _Params(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int) for n in ("K", "N", "H", "W", "use_geo", "pad")] +
                [(n, ctypes.c_double) for n in ("dt", "half", "v_max", "om_max", "alpha_max", "a_max_eff",
                                                "dmax", "alpha_lim", "a_lim", "j_lim", "v_lim", "om_lim",
                                                "res", "ox", "oy", "r_safe", "gx", "gy", "penalty")] +
                [("xi0", ctypes.c_double * 6), ("prev_clip", ctypes.c_double * 2),
                 ("prev_raw", ctypes.c_double * 2)])


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class CudaBatch(object):
    """One CUDA library per process; shareable by several planner instances."""

    def __init__(self, lib_path=None):
        path = Path(lib_path or os.environ.get("MPPI_BATCH_LIB", LIB))
        self.lib = ctypes.CDLL(str(path))
        self.lib.mppi_params_size.restype = ctypes.c_int
        if self.lib.mppi_params_size() != ctypes.sizeof(_Params):
            raise RuntimeError("CUDA MppiParams layout differs from the Python mirror")
        self.lib.mppi_eval.restype = ctypes.c_int
        self.lib.mppi_frontier.restype = ctypes.c_int
        self.path = str(path)
        self.calls = self.fallbacks = 0

    def warmup(self):
        """Create the CUDA context now instead of inside the first plan."""
        if hasattr(self.lib, "mppi_warmup") and self.lib.mppi_warmup() != 0:
            raise RuntimeError("CUDA context creation failed")

    # ------------------------------------------------------------------
    @staticmethod
    def _supported(planner):
        occ = getattr(planner.validator, "occ", None)
        dyn = planner.dyn
        return (type(dyn) is CappedDynamics
                and getattr(dyn.step, "__func__", None) is PlanarDynamics.step
                and occ is not None and hasattr(occ, "unsafe") and hasattr(occ, "cell_to_world"))

    @staticmethod
    def _cpu(planner, xi0, U, a_prev):
        U = planner.dyn.clip_inputs(U, a_prev=a_prev)
        X = planner.dyn.rollout(xi0, U)
        valid = planner.dyn.states_ok(X)
        valid &= planner.dyn.inputs_ok(U, a_prev=a_prev)
        valid &= planner.validator.nodes_safe(X[..., S_POS])
        return U, X, valid, None

    @staticmethod
    def _clearance_table(occ):
        # clearance() depends only on the cell a point falls in; evaluate it
        # once per cell at the cell centre (maps back to the same cell).
        ix, iy = np.meshgrid(np.arange(occ.W), np.arange(occ.H))
        xc, yc = occ.cell_to_world(ix, iy)
        tab = np.asarray(occ.clearance(xc, yc), dtype=np.float64)
        cx, cy = occ.world_to_cell(xc, yc)
        if not (np.array_equal(cx, ix) and np.array_equal(cy, iy)):
            raise RuntimeError("cell centres do not map back to their cells")
        return np.ascontiguousarray(tab)

    @staticmethod
    def _frontier_classes(planner, occ):
        """first_blocking_class's classification grid (r_pass = r_safe)."""
        occupied = np.asarray(getattr(occ, "occupied", occ.unsafe))
        unknown = np.asarray(getattr(occ, "unknown", np.zeros_like(occupied, dtype=bool)))
        r_pass = planner.validator.r_safe
        if r_pass > 0.0:
            passable = planner._map_cost_cache.passable(occupied, occ.res, r_pass)
            occupied = occupied | (~passable & ~unknown)
        classes = np.zeros(occupied.shape, dtype=np.int8)
        classes[unknown] = UNKNOWN
        classes[occupied] = OCCUPIED
        return np.ascontiguousarray(classes)

    # ------------------------------------------------------------------
    def evaluate(self, planner, xi0, U, goal, a_prev):
        if not self._supported(planner):
            self.fallbacks += 1
            return self._cpu(planner, xi0, U, a_prev)
        self.calls += 1
        dyn, lim, val = planner.dyn, planner.dyn.lim, planner.validator
        occ = val.occ
        U = np.ascontiguousarray(U, dtype=np.float64)
        K, N = U.shape[0], U.shape[1]
        if a_prev is None:
            prev_raw = np.zeros(2)
            prev_clip = np.zeros((1, 2))
        else:
            prev_raw = np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2].copy()
            prev_clip = prev_raw[None, :].copy()
            pn = _norm_last_axis(prev_clip, keepdims=True)
            prev_clip *= np.minimum(1.0, lim.a_max_eff / np.maximum(pn, 1e-12))
        p = _Params()
        p.K, p.N, p.H, p.W = K, N, occ.H, occ.W
        p.dt = dyn.dt
        p.half = 0.5 * dyn.dt * dyn.dt
        p.v_max, p.om_max = float(lim.v_max), float(lim.omega_max)
        p.alpha_max, p.a_max_eff, p.dmax = lim.alpha_max, lim.a_max_eff, lim.j_max * dyn.dt
        p.alpha_lim = lim.alpha_max + _TOL
        p.a_lim = lim.a_max_eff + _TOL
        p.j_lim = lim.j_max * dyn.dt + _TOL
        p.v_lim = lim.v_max + _TOL
        p.om_lim = lim.omega_max + _TOL
        p.res, p.ox, p.oy = float(occ.res), float(occ.origin[0]), float(occ.origin[1])
        p.r_safe = float(val.r_safe)
        ctg = planner.ctg if (planner.use_geodesic and planner.ctg is not None) else None
        g = np.asarray(ctg.goal if ctg is not None else goal, dtype=np.float64).reshape(-1)[:2]
        p.gx, p.gy = float(g[0]), float(g[1])
        p.use_geo = 1 if ctg is not None else 0
        p.penalty = float(REACH_PENALTY_M)
        xi = np.asarray(xi0, dtype=np.float64).reshape(-1)
        for i in range(6):
            p.xi0[i] = xi[i]
        p.prev_clip[0], p.prev_clip[1] = prev_clip[0]
        p.prev_raw[0], p.prev_raw[1] = prev_raw
        if ctg is not None and (ctg.H, ctg.W, ctg.res, ctg.origin) != (occ.H, occ.W, p.res, (p.ox, p.oy)):
            self.fallbacks += 1
            return self._cpu(planner, xi0, U, a_prev)
        clear = self._clearance_table(occ)
        field = np.ascontiguousarray(ctg.field if ctg is not None else np.zeros((occ.H, occ.W)), dtype=np.float64)
        Uo = np.empty_like(U)
        X = np.empty((K, N + 1, 6))
        flags = np.empty(K, dtype=np.uint8)
        dgeo = np.empty((K, N + 1))
        cl = np.empty((K, N + 1))
        rc = self.lib.mppi_eval(ctypes.byref(p), _ptr(U), _ptr(clear), _ptr(field),
                                _ptr(Uo), _ptr(X), _ptr(flags), _ptr(dgeo), _ptr(cl))
        if rc != 0:
            raise RuntimeError("mppi_eval failed: %d" % rc)
        valid = flags == 7
        frontier = None
        if planner.w_frontier != 0.0:
            gf = np.asarray(goal, dtype=np.float64).reshape(2)
            P = X[:, -1, 0:2]
            d = gf[None, :] - P
            dist = np.linalg.norm(d, axis=1)
            step = 0.5 * float(occ.res)
            m = int(np.ceil(float(dist.max()) / step)) + 1 if dist.size else 2
            m = int(np.clip(m, 2, 256))
            t = np.ascontiguousarray(np.linspace(0.0, 1.0, m))
            classes = self._frontier_classes(planner, occ)
            out = np.empty(K, dtype=np.int32)
            rc = self.lib.mppi_frontier(ctypes.c_double(gf[0]), ctypes.c_double(gf[1]), m, _ptr(t),
                                        occ.H, occ.W, ctypes.c_double(p.res), ctypes.c_double(p.ox),
                                        ctypes.c_double(p.oy), _ptr(classes), _ptr(out))
            if rc != 0:
                raise RuntimeError("mppi_frontier failed: %d" % rc)
            frontier = out.astype(np.int64)
        U[...] = Uo    # rollout's in-place contract: the caller's U holds the applied inputs
        parts = dict(d=dgeo, clearance=cl if planner.w.w_obs > 0.0 else None, frontier=frontier)
        return U, X, valid, parts


_SHARED = None


def enable_cuda(*planners, lib_path=None, share_map_cache=True):
    """Attach one shared CUDA backend to each FrontierMPPI instance.

    share_map_cache: planners on the SAME validator/grid (DeSimplex's haa and
    probe) also share one MapCostCache. The cache compares the map contents on
    every call, so sharing only avoids recomputing an identical EDT/graph/field.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = CudaBatch(lib_path)
        _SHARED.warmup()
    for planner in planners:
        planner.batch_backend = _SHARED
    if share_map_cache and len(planners) > 1:
        cache = planners[0]._map_cost_cache
        for planner in planners[1:]:
            if planner.validator is planners[0].validator:
                planner._map_cost_cache = cache
    return _SHARED
