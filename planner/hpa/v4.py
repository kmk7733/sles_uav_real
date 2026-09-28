"""Pinned ROGX V4 inference on an already synchronized planar observation.

This module consumes a normalized (2, 512) scan and an encoded, *unnormalized*
eight-element PX4 state. It does not acquire depth, synchronize messages, choose
goals, transform planning frames, or produce aircraft commands. All ten output
accelerations use the fixed body FLU frame at the observation anchor.

CUDA is required in normal operation. CPU is an explicit offline test option;
there is no device or checkpoint fallback. The unchanged deployment model is
loaded from verified bytes without importing training or simulator modules.
"""
import builtins
import hashlib
import io
import json
from pathlib import Path
import time
import types

import numpy as np

from .scan_cache import CachedScanProjection


CANDIDATE = "v4_lr1e3_wd1_b64_es10_s1_e40"
MANIFEST_SHA256 = "a91d533dd5ba9e84c5545fa91a545c5a4a4e3fd95cdea518bfefaab60487f6b1"
CHECKPOINT_SHA256 = "d05e6dc4b0cb410eea1cf2f274946481244ad1fa6c23cc77856d1201c9d2edc2"
BEST_EPOCH = 35
CHUNK = 10
NODE_DT = 0.1
MINIMUM_VALID_BEAM_FRACTION = 0.05
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "deploy" / "rogx_hpa_v4"


class V4ContractError(ValueError):
    """Bundle or observation differs from the approved V4 contract."""


def _load_private_modules(directory, payloads):
    """Resolve bundle-local imports without touching process import caches."""
    modules = {}
    original_import = builtins.__import__

    def bundle_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return original_import(name, globals, locals, fromlist, level)

    for name in ("hpa_model", "real_geometry", "real_preprocess"):
        module = types.ModuleType("_planner_hpa_pinned_v4_" + name)
        module.__file__ = str(directory / (name + ".py"))
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=bundle_import)
        modules[name] = module
        exec(compile(payloads[name + ".py"], module.__file__, "exec"), module.__dict__)
    # Functions consume the verified configuration snapshot, not a later disk
    # edit. The packaged make_scan/encode_px4_state/action_chunk code is intact.
    configuration = json.loads(payloads["preprocessing.json"])
    modules["real_preprocess"].config = lambda: configuration
    return modules


def _verified_payloads(bundle_dir):
    """Read once, verify every payload, then use those same bytes for loading."""
    directory = Path(bundle_dir).expanduser().resolve(strict=True)
    manifest_path = directory / "SHA256.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise V4ContractError("V4 manifest must be a regular local file")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise V4ContractError("manifest differs from the pinned V4 bundle")
    manifest = json.loads(manifest_bytes)
    payloads = {}
    for name, expected in manifest.items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise V4ContractError("V4 member must be a regular local file: " + name)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise V4ContractError("V4 checksum mismatch: " + name)
        payloads[name] = payload
    info = json.loads(payloads["model_info.json"])
    if (info["candidate"] != CANDIDATE or info["best_epoch"] != BEST_EPOCH
            or info["source_checkpoint_sha256"] != CHECKPOINT_SHA256
            or manifest["best.pt"] != CHECKPOINT_SHA256):
        raise V4ContractError("unexpected V4 candidate, checkpoint, or best epoch")
    return directory, manifest, info, payloads


def validate_scan_state(scan, state):
    """Validate and copy caller data; state normalization belongs to the net.

    State order is body vx/vy, body angular-z, body goal direction x/y, planar
    goal distance, sin(raw PX4 yaw), cos(raw PX4 yaw). Its yaw may differ from
    the planning-frame anchor yaw by an explicitly established frame alignment.
    Synchronization, acquisition timestamps and reset epochs belong to the
    caller's frozen observation, never to unstamped mutable mapper attributes.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        scan = np.array(scan, dtype=np.float32, copy=True)
        state = np.array(state, dtype=np.float32, copy=True)
    if scan.shape != (2, 512) or state.shape != (8,):
        raise V4ContractError("expected scan (2,512) and state (8,)")
    if not np.isfinite(scan).all() or not np.isfinite(state).all():
        raise V4ContractError("nonfinite V4 observation")
    if not ((scan >= 0) & (scan <= 1)).all():
        raise V4ContractError("scan range and validity channels must be in [0,1]")
    if not np.isin(scan[1], [0, 1]).all():
        raise V4ContractError("scan validity channel must be binary")
    if float(scan[1].mean()) < MINIMUM_VALID_BEAM_FRACTION:
        raise V4ContractError("too few valid V4 depth bearings")
    if state[5] < 0:
        raise V4ContractError("planar goal distance must be nonnegative")
    if not np.isclose(np.hypot(state[6], state[7]), 1.0, rtol=0, atol=1e-5):
        raise V4ContractError("state yaw must be an unnormalized sin/cos pair")
    expected_direction_norm = min(float(state[5]) / 1e-6, 1.0)
    if not np.isclose(np.hypot(state[3], state[4]), expected_direction_norm,
                      rtol=0, atol=1e-5):
        raise V4ContractError("state goal direction and distance violate V4 encoding")
    return scan, state


class PinnedV4Runtime:
    """The approved V4 checkpoint behind a small, array-only inference API.

    ``infer(scan, state)`` returns physical (10,3) ``[ax, ay, yaw_acceleration]``
    in the fixed anchor body FLU frame. ``last_inference_ms`` includes validation,
    host/device transfers and physical-unit scaling, with CUDA synchronization.
    It excludes acquisition, scan projection, and ROS transport. The producer
    rotates every XY node by the same explicit planning-frame anchor yaw.
    """
    chunk = CHUNK
    dt = NODE_DT
    node_dt = NODE_DT
    candidate = CANDIDATE
    checkpoint_sha256 = CHECKPOINT_SHA256

    def __init__(self, bundle_dir=None, device="cuda", allow_cpu_for_tests=False):
        if device not in ("cuda", "cpu"):
            raise V4ContractError("device must be cuda or explicit offline-test cpu")
        if device == "cpu" and allow_cpu_for_tests is not True:
            raise V4ContractError("CPU requires allow_cpu_for_tests=True; no fallback is permitted")
        directory, manifest, info, payloads = _verified_payloads(
            DEFAULT_BUNDLE if bundle_dir is None else bundle_dir)
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("V4 requires working CUDA; CPU fallback is disabled")
        self._torch = torch
        self.device = device
        self.bundle_dir = directory
        self.manifest = dict(manifest)
        self.model_info = dict(info)
        self.last_inference_ms = None
        # Do not put generic bundle modules on sys.path or in sys.modules:
        # an earlier checkpoint loader could otherwise poison the import cache.
        modules = _load_private_modules(directory, payloads)
        module = modules["hpa_model"]
        self._model = module
        self._preprocess = modules["real_preprocess"]
        self._scan_projection = CachedScanProjection(
            module, self._preprocess, payloads["hpa_model.py"])
        self._net, checkpoint = module.load_policy(io.BytesIO(payloads["best.pt"]))
        self._checkpoint = checkpoint
        expected_spec = {"module": "hpa.model", "cls": "DepthPolicy", "state_dim": 8,
                         "action_dim": 3, "chunk": CHUNK, "in_ch": 2,
                         "scan": True, "state_hidden": 0}
        if checkpoint["model"] != expected_spec or checkpoint["train"]["epoch"] != BEST_EPOCH:
            raise V4ContractError("loaded model differs from the pinned V4 architecture")
        if checkpoint["data_meta"]["node_dt"] != NODE_DT:
            raise V4ContractError("V4 output interval must be 0.1 seconds")
        self._action_mean = np.asarray(checkpoint["norm"]["action_mean"], dtype=np.float64)
        self._action_std = np.asarray(checkpoint["norm"]["action_std"], dtype=np.float64)
        if (self._action_mean.shape != (3,) or self._action_std.shape != (3,)
                or not np.isfinite(self._action_mean).all()
                or not np.isfinite(self._action_std).all()
                or not (self._action_std > 0).all()):
            raise V4ContractError("invalid V4 physical-output normalization")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._net.to(device).eval()

    @property
    def metadata(self):
        return {"candidate": CANDIDATE, "best_epoch": BEST_EPOCH,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "manifest_sha256": MANIFEST_SHA256, "device": self.device,
                "chunk": CHUNK, "node_dt_s": NODE_DT,
                "scan_projection": "approved arithmetic with cached float64 pixel rays",
                "pixel_ray_cache": self._scan_projection.rays.metadata,
                "action_frame": "fixed body FLU at the observation anchor"}

    def make_scan(self, depth_m, roll_pitch, K):
        """Packaged projection using live K and the fixed training mount.

        Returns the original float32 scan. A caller matching float16 training
        storage must explicitly round-trip float16 then float32 after this
        projection, and record that choice. Only K/resolution-dependent pixel
        rays are cached; all per-frame arithmetic is unchanged. This method
        performs no ROS I/O.
        """
        return self._scan_projection.make_scan(depth_m, roll_pitch, K)

    def make_scan_reference(self, depth_m, roll_pitch, K):
        """Unmodified bundle projection for paired parity/latency measurements."""
        return self._scan_projection.reference_make_scan(depth_m, roll_pitch, K)

    def encode_px4_state(self, position_enu, quaternion_xyzw, velocity_enu,
                         angular_body, goal_enu):
        """Packaged state encoding; state normalization remains inside the net.

        ``angular_body`` must come from PX4 odometry's body angular velocity,
        not from velocity_local.angular or an Euler yaw derivative. The caller
        establishes acquisition-time alignment before passing these arrays.
        """
        return self._preprocess.encode_px4_state(
            position_enu, quaternion_xyzw, velocity_enu, angular_body, goal_enu)

    def infer(self, scan, state):
        self.last_inference_ms = None
        if self.device == "cuda":
            self._torch.cuda.synchronize(self.device)
        begin = time.perf_counter()
        scan, state = validate_scan_state(scan, state)
        output = self._preprocess.action_chunk(
            self._net, self._checkpoint, scan, state, self.device)
        if output.shape != (CHUNK, 3) or not np.isfinite(output).all():
            raise V4ContractError("invalid V4 physical action chunk")
        if self.device == "cuda":
            self._torch.cuda.synchronize(self.device)
        self.last_inference_ms = 1000.0 * (time.perf_counter() - begin)
        output.setflags(write=False)
        return output
