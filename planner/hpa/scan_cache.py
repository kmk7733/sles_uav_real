"""Reuse calibrated pixel rays without changing the approved scan arithmetic.

The model bundle remains byte-identical. Its verified planar_scan source is
compiled with exactly one substitution: the five statements that construct
calibration-dependent rays become a cache lookup. Depth, attitude, mount
rotation, height filtering, bearings, min pooling and normalization still run
through the original expressions with their original float64 intermediates.

This deliberately refuses an unfamiliar source block. A later model bundle
must be reviewed instead of silently applying an optimization to new code.
Only private function namespaces are rebound; NumPy and bundle modules are
never monkeypatched. The original preprocessing function remains available.
"""
import ast
from collections import OrderedDict
import threading
import types

import numpy as np


_RAY_BLOCK = """    v_i, u_i = np.indices((h, w))
    xn = (u_i - st.cx) / st.fx
    yn = (v_i - st.cy) / st.fy
    ob = np.sqrt(1.0 + xn * xn + yn * yn)          # ray length / optical-axis Z
    rc = np.stack([xn / ob, yn / ob, 1.0 / ob])    # unit ray, camera frame
"""
_CACHED_RAY_BLOCK = "    ob, rc = _hpa_cached_pixel_rays(h, w, st)\n"


class PixelRayCache:
    """A bounded, thread-safe cache of immutable float64 ray arrays.

    At 360x640, each calibration retains 7,372,800 bytes. The default two
    entries therefore retain at most about 14.1 MiB. Intrinsic values use their
    exact float64 bytes as a key, so even signed zero is not silently merged.
    FOV, depth and attitude do not affect pixel rays and are never cached here.
    """

    def __init__(self, max_entries=2):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("pixel ray cache capacity must be a positive integer")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, h, w, st):
        calibration = np.asarray([st.fx, st.fy, st.cx, st.cy], dtype=np.float64)
        key = (int(h), int(w), calibration.tobytes())
        with self._lock:
            if key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                return self._entries[key]
            self._misses += 1
            # These expressions intentionally match the approved source.
            v_i, u_i = np.indices((h, w))
            xn = (u_i - st.cx) / st.fx
            yn = (v_i - st.cy) / st.fy
            ob = np.sqrt(1.0 + xn * xn + yn * yn)
            rc = np.stack([xn / ob, yn / ob, 1.0 / ob])
            ob.setflags(write=False)
            rc.setflags(write=False)
            self._entries[key] = (ob, rc)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return ob, rc

    @property
    def metadata(self):
        with self._lock:
            return {"entries": len(self._entries), "max_entries": self.max_entries,
                    "hits": self._hits, "misses": self._misses,
                    "bytes": sum(ob.nbytes + rc.nbytes for ob, rc in self._entries.values()),
                    "dtype": "float64", "key": "height,width,fx,fy,cx,cy"}


def _rebind(function, namespace):
    """Keep a verified function's bytecode, changing only its private globals."""
    rebound = types.FunctionType(function.__code__, namespace, function.__name__,
                                 function.__defaults__, function.__closure__)
    rebound.__kwdefaults__ = function.__kwdefaults__
    return rebound


class CachedScanProjection:
    """Cached and original entry points for the same verified preprocessing."""

    def __init__(self, model, preprocessing, verified_model_source, max_entries=2):
        source = (verified_model_source.decode("utf-8")
                  if isinstance(verified_model_source, bytes) else verified_model_source)
        if source.count(_RAY_BLOCK) != 1:
            raise ValueError("approved planar_scan pixel-ray source block differs")
        original_tree = ast.parse(source)
        original_functions = [node for node in original_tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == "planar_scan"]
        if len(original_functions) != 1:
            raise ValueError("expected exactly one approved planar_scan function")
        original_function = original_functions[0]
        function_source = "".join(source.splitlines(keepends=True)[
            original_function.lineno - 1:original_function.end_lineno])
        if function_source.count(_RAY_BLOCK) != 1:
            raise ValueError("pixel-ray source block must belong to planar_scan")
        tree = ast.parse(function_source.replace(_RAY_BLOCK, _CACHED_RAY_BLOCK, 1))
        self.rays = PixelRayCache(max_entries)
        model_namespace = dict(vars(model), _hpa_cached_pixel_rays=self.rays.get)
        exec(compile(tree, model.__file__ + " [cached pixel rays]", "exec"), model_namespace)
        self.planar_scan = model_namespace["planar_scan"]
        cached_scan_features = _rebind(model.scan_features, model_namespace)
        # Retain other model names without mutating the original model module.
        cached_model = types.SimpleNamespace(**dict(vars(model), scan_features=cached_scan_features))
        preprocessing_namespace = dict(vars(preprocessing), model=cached_model)
        self.make_scan = _rebind(preprocessing.make_scan, preprocessing_namespace)
        self.reference_make_scan = preprocessing.make_scan
