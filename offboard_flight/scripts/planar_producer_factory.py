"""Construct the shared planar producers without ROS or simulator imports.

``build_producer(mode, config, occupancy, goal, action_provider=...)`` returns a
ProducerAssembly. Its ``producer.plan`` is the existing core interface. The
factory does not publish commands, read ROS parameters, load a checkpoint, or
choose a coordinate transform. HPA modes require a provider returning
``planner.hpa.BodyActionChunk`` in the planning frame at the observation anchor.

All HAA limits, sampling and cost settings are explicit. No hardware profile is
selected automatically. See ``planar_producer_config.example.json`` for an
inactive example. Optional DeSimplex settings default to the unchanged core
defaults, not an experiment profile. ``effective_config`` includes those values.

An epoch/new episode requires a NEW assembly. DeSimplex has no reset method;
calling HPA.reset does not clear the simulator's committed chunk. A caller must
also retire its sensor snapshots and references at that boundary. Changing a
goal within an episode is still the core's ordinary plan(..., goal=...) call;
this factory does not invent a goal-triggered reset or a wall-clock commit rule.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from planner.dynamics import PlanarDynamics, PlanarLimits
from planner.haa.capped import CappedDynamics
from planner.haa.cost import FrontierMPPI
from planner.mppi import PlanarCostWeights
from planner.trajectory_safety import TrajectorySafetyValidator
from planner.desimplex import DeSimplexSwitcher
from planner.supervisor import (
    N_R_DEFAULT, N_M_DEFAULT, TRANSITION_STEPS_DEFAULT,
    RECOVERY_MARGIN_DEFAULT, WARM_HAA_DEFAULT, N_LOOK_DEFAULT,
    HANDOVER_DECEL_DEFAULT, SEED_NOMINAL_DEFAULT,
)


LIMIT_FIELDS = ("v_max", "a_max", "omega_max", "alpha_max", "tilt_max", "j_max")
WEIGHT_FIELDS = ("w_goal", "w_term_pos", "w_term_vel", "w_obs", "d_influence",
                 "R_dnu", "w_yaw", "yaw_mode")
HAA_FIELDS = ("limits", "horizon", "num_samples", "sigma", "temperature",
              "seed", "goal_tol", "cap_velocity", "use_geodesic",
              "w_frontier", "c_occupied", "c_unknown", "weights")
DS_DEFAULTS = {
    "n_r": N_R_DEFAULT, "n_m": N_M_DEFAULT, "probe_seed": 0,
    "transition_steps": TRANSITION_STEPS_DEFAULT,
    "recovery_margin": RECOVERY_MARGIN_DEFAULT, "warm_haa": WARM_HAA_DEFAULT,
    "n_look": N_LOOK_DEFAULT, "handover_decel": HANDOVER_DECEL_DEFAULT,
    "seed_nominal": SEED_NOMINAL_DEFAULT,
}


def _fields(value, required, optional=(), name="config"):
    if not isinstance(value, dict):
        raise ValueError("%s must be a dictionary" % name)
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        raise ValueError("%s: missing=%s unknown=%s" %
                         (name, sorted(missing), sorted(unknown)))


def _number(value, name, minimum=0.0, strictly_positive=False):
    if isinstance(value, (bool, str)):
        raise ValueError("%s must be a finite number" % name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % name)
    if not np.isfinite(result) or result < minimum or (strictly_positive and result == 0):
        raise ValueError("invalid %s: %r" % (name, value))
    return result


def _integer(value, name, minimum=0, maximum=None):
    result = _number(value, name, minimum=minimum)
    if result != int(result) or (maximum is not None and result > maximum):
        raise ValueError("invalid integer %s: %r" % (name, value))
    return int(result)


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError("%s must be a JSON boolean" % name)
    return value


def _limits(value, name):
    _fields(value, LIMIT_FIELDS, name=name)
    result = {key: _number(value[key], name + "." + key, strictly_positive=True)
              for key in LIMIT_FIELDS}
    if result["tilt_max"] >= np.pi / 2:
        raise ValueError("%s.tilt_max is radians and must be below pi/2" % name)
    return result


def _vector(value, name, size=3):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("%s must contain %d finite nonnegative values" % (name, size))
    return result.tolist()


def _normalize(mode, config):
    if mode not in ("haa", "hpa", "desimplex"):
        raise ValueError("mode must be haa, hpa or desimplex")
    _fields(config, ("dt",), ("label", "haa", "hpa", "safety", "haa_true_limits",
                              "desimplex"))
    cfg = copy.deepcopy(config)
    cfg["dt"] = _number(cfg["dt"], "dt", strictly_positive=True)
    if "label" in cfg and not isinstance(cfg["label"], str):
        raise ValueError("label must be a string")
    required = {"haa": ("haa", "safety"), "hpa": ("hpa",),
                "desimplex": ("haa", "hpa", "safety", "haa_true_limits")}[mode]
    if any(key not in cfg for key in required):
        raise ValueError("%s requires explicit sections %s" % (mode, required))
    if "haa" in cfg:
        h = cfg["haa"]
        _fields(h, HAA_FIELDS, ("cost_normalise", "beta_min", "max_sample_sweeps"), "haa")
        h["limits"] = _limits(h["limits"], "haa.limits")
        for key in ("horizon", "num_samples"):
            h[key] = _integer(h[key], "haa." + key, minimum=1)
        h["seed"] = _integer(h["seed"], "haa.seed", maximum=2**32 - 1)
        for key in ("temperature", "goal_tol"):
            h[key] = _number(h[key], "haa." + key, strictly_positive=True)
        for key in ("cap_velocity", "use_geodesic"):
            h[key] = _boolean(h[key], "haa." + key)
        h["sigma"] = _vector(h["sigma"], "haa.sigma")
        for key in ("w_frontier", "c_occupied", "c_unknown"):
            h[key] = _number(h[key], "haa." + key, minimum=-float("inf"))
        w = h["weights"]
        _fields(w, WEIGHT_FIELDS, name="haa.weights")
        for key in ("w_goal", "w_term_pos", "w_term_vel", "w_obs", "w_yaw"):
            w[key] = _number(w[key], "haa.weights." + key)
        if w["d_influence"] is not None:
            w["d_influence"] = _number(w["d_influence"], "haa.weights.d_influence", strictly_positive=True)
        w["R_dnu"] = _vector(w["R_dnu"], "haa.weights.R_dnu")
        if w["yaw_mode"] not in ("velocity", "goal", "goal_in_view", "hold"):
            raise ValueError("unknown haa.weights.yaw_mode")
        # These are PlanarMPPI's unchanged constructor defaults. Logging them
        # makes behavior visible even though the existing ROS node omits them.
        h["cost_normalise"] = _boolean(h.get("cost_normalise", True), "haa.cost_normalise")
        h["beta_min"] = _number(h.get("beta_min", 1.0 / 64.0), "haa.beta_min", strictly_positive=True)
        if h["beta_min"] > 1:
            raise ValueError("haa.beta_min must not exceed 1")
        h["max_sample_sweeps"] = _integer(h.get("max_sample_sweeps", 24), "haa.max_sample_sweeps", minimum=1)
    if "safety" in cfg:
        s = cfg["safety"]
        _fields(s, ("r_safe", "sweep_step"), name="safety")
        s["r_safe"] = _number(s["r_safe"], "safety.r_safe")
        s["sweep_step"] = _number(s["sweep_step"], "safety.sweep_step", strictly_positive=True)
    if "hpa" in cfg:
        h = cfg["hpa"]
        _fields(h, ("limits", "goal_tol"), ("commit",), "hpa")
        h["limits"] = _limits(h["limits"], "hpa.limits")
        h["goal_tol"] = _number(h["goal_tol"], "hpa.goal_tol", strictly_positive=True)
        h["commit"] = _integer(h.get("commit", 1), "hpa.commit", minimum=1)
    if "haa_true_limits" in cfg:
        cfg["haa_true_limits"] = _limits(cfg["haa_true_limits"], "haa_true_limits")
    if mode in ("hpa", "desimplex") and cfg["dt"] != 0.1:
        raise ValueError("the fixed V4 action chunk requires dt=0.1")
    if mode == "desimplex":
        if cfg["hpa"]["goal_tol"] != cfg["haa"]["goal_tol"]:
            raise ValueError("HPA and HAA must have the same goal tolerance")
        for key in LIMIT_FIELDS:
            if cfg["haa"]["limits"][key] > cfg["haa_true_limits"][key]:
                raise ValueError("tightened HAA %s exceeds its true envelope" % key)
    if "desimplex" in cfg or mode == "desimplex":
        options = cfg.get("desimplex", {})
        _fields(options, (), DS_DEFAULTS, "desimplex")
        d = dict(DS_DEFAULTS)
        d.update(options)
        for key in ("n_r", "n_m", "transition_steps", "probe_seed"):
            d[key] = _integer(d[key], "desimplex." + key,
                              maximum=(2**32 - 1 if key == "probe_seed" else None))
        d["n_look"] = _integer(d["n_look"], "desimplex.n_look", minimum=1)
        d["recovery_margin"] = _number(d["recovery_margin"], "desimplex.recovery_margin")
        for key in ("warm_haa", "handover_decel", "seed_nominal"):
            d[key] = _boolean(d[key], "desimplex." + key)
        cfg["desimplex"] = d
    # Enforce serializable provenance rather than accidentally logging objects.
    json.dumps(cfg, allow_nan=False)
    return cfg


def _haa(config, dt, validator):
    args = dict(config)
    limits = PlanarLimits(**args.pop("limits"))
    dyn_cls = CappedDynamics if args.pop("cap_velocity") else PlanarDynamics
    weights = PlanarCostWeights(**args.pop("weights"))
    return FrontierMPPI(dyn_cls(limits, dt=dt), validator, weights=weights, **args)


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ("planner/supervisor.py", "planner/transition.py", "planner/dynamics.py",
             "planner/mppi.py", "planner/safety.py", "planner/grid.py", "planner/types.py",
             "planner/haa/capped.py", "planner/haa/cost.py", "planner/haa/geodesic.py",
             "planner/hpa/producer.py", "planner/hpa/commit.py", "planner/hpa/reference.py",
             "planner/hpa/__init__.py", "planner/hpa/v4.py",
             "planner/desimplex.py", "planner/trajectory_safety.py",
             "offboard_flight/scripts/planar_producer_factory.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


class ProducerAssembly:
    """Explicit handles and construction provenance; contains no ROS lifecycle."""

    def __init__(self, mode, producer, config, validator=None, haa=None,
                 probe=None, hpa=None, action_provider=None, validity_gate=None):
        self.mode, self.producer = mode, producer
        self.validator, self.haa, self.probe, self.hpa = validator, haa, probe, hpa
        self._config = copy.deepcopy(config)
        active = ["haa", "safety"] if mode == "haa" else ["hpa"]
        if mode == "desimplex":
            active = ["haa", "safety", "hpa", "haa_true_limits", "desimplex"]
        self._description = {
            "mode": mode, "producer_class": type(producer).__name__,
            "producer_role": ("DeSimplexSwitcher" if mode == "desimplex"
                              else type(producer).__name__),
            "validator_role": ("TrajectorySafetyValidator"
                               if validator is not None else None),
            "config": copy.deepcopy(config), "active_sections": active,
            "source_sha256": _source_hashes(),
            "control_output": "none; caller receives MPPIResult only",
            "time_contract": "one plan call per algorithm tick; no wall-clock substitution",
        }
        if haa is not None:
            self._description["haa_effective"] = {
                "dynamics": type(haa.dyn).__name__,
                "a_max_eff": haa.dyn.lim.a_max_eff,
                "d_influence": haa.d_influence,
            }
        if hpa is not None:
            self._description["hpa_contract"] = {
                "chunk": 10, "node_dt_s": 0.1,
                "action": ["ax", "ay", "yaw_acceleration"],
                "frame": "fixed body FLU at explicit planning-frame anchor yaw",
                "provider": type(action_provider).__module__ + "." + type(action_provider).__qualname__,
                "model_binding": "external provider; factory does not load or verify weights",
                "validity_gate_supplied": validity_gate is not None,
            }
            metadata = getattr(action_provider, "metadata", None)
            if isinstance(metadata, dict):
                json.dumps(metadata, allow_nan=False)
                self._description["provider_reported_metadata"] = copy.deepcopy(metadata)

    @property
    def effective_config(self):
        return copy.deepcopy(self._description)

    def describe(self):
        """JSON suitable for a construction log; no ROS publisher is created."""
        return json.dumps(self._description, sort_keys=True, allow_nan=False)

    def set_occupancy(self, occupancy):
        """Set one snapshot on the already shared validator, without a solve."""
        if self.validator is not None:
            self.validator.occ = occupancy


def build_producer(mode, config, occupancy=None, goal=None,
                   action_provider=None, validity_gate=None):
    """Build haa/hpa/desimplex from explicit settings.

    HPA-only requires just dt and hpa settings. HAA/DeSimplex require an explicit
    occupancy object (FreeSpace is allowed when deliberately supplied). The
    action provider owns model binding and observation acquisition. Unit tests
    can inject a labelled synthetic provider; that is never marked as V4
    validation. The production provider must use the pinned V4 runtime.

    This returns the ORIGINAL DeSimplexSupervisor when selected, including its
    whole .plan path and bridge monitoring. No switching formula is copied here.
    """
    cfg = _normalize(mode, config)
    validator = haa = probe = hpa = None
    if mode in ("haa", "desimplex"):
        if occupancy is None:
            raise ValueError("%s requires an explicit occupancy snapshot" % mode)
        validator = TrajectorySafetyValidator(occupancy, **cfg["safety"])
        haa = _haa(cfg["haa"], cfg["dt"], validator)
    if mode in ("hpa", "desimplex"):
        if not callable(action_provider):
            raise ValueError("HPA requires an explicit action_provider")
        if validity_gate is not None and not callable(validity_gate):
            raise ValueError("validity_gate must be callable")
        # Lazy: existing HAA-only users need neither HPA nor torch imports.
        from planner.hpa import HPAProducer
        h = cfg["hpa"]
        hpa = HPAProducer(action_provider, PlanarLimits(**h["limits"]),
                          dt=cfg["dt"], chunk=10, commit=h["commit"],
                          goal_tol=h["goal_tol"], validity_gate=validity_gate)
    if mode == "desimplex":
        target = np.asarray(goal, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("DeSimplex requires a finite goal (2,) in the planning frame")
        # Each constructor starts an independent RNG and nominal/reference
        # cache. Settings and initial seed are identical; validator is shared.
        probe = _haa(cfg["haa"], cfg["dt"], validator)
        dyn_full = PlanarDynamics(PlanarLimits(**cfg["hpa"]["limits"]), dt=cfg["dt"])
        producer = DeSimplexSwitcher(
            haa, probe, dyn_full, validator, target, hpa=hpa,
            lim_x=PlanarLimits(**cfg["haa_true_limits"]), **cfg["desimplex"])
    else:
        producer = haa if mode == "haa" else hpa
    return ProducerAssembly(mode, producer, cfg, validator=validator, haa=haa,
                            probe=probe, hpa=hpa, action_provider=action_provider,
                            validity_gate=validity_gate)
