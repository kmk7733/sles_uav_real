#!/usr/bin/env python3
"""DeSimplex ROS node: V4 HPA and HAA under the ORIGINAL DeSimplexSupervisor.

Inputs, output, trajectory lifetime, goal arrival and the controller contract
are exactly hpa_planner_node's (depth + PX4 local ENU -> existing Controller
PositionTarget; CollisionStopGuard in front of the mission). The one change is
the producer: each synchronized depth frame is ONE supervisor tick, i.e. one
full ``DeSimplexSupervisor.plan()`` call (HPA proposal, S_HAA/R_Nr/M_Nm checks,
recovery, hand-back and bridge), as in the simulator where the learned HPA is
re-asked every 10 Hz tick (hpa.commit=1). The call count, not a ROS timer,
advances the commit/bridge counters.

HAA needs the existing /grid_map of the current alignment epoch, so the
supervisor plans in that world frame: the PX4 state at the depth anchor goes
local->world, the chosen reference goes world->local with the same epoch
alignment. The V4 network itself still sees only PX4 local ENU inputs and the
goal in local ENU. Vicon is never a planner input.

``fault=True`` best-effort braking is logged as such; it is NOT a verified
safe recovery.
"""
import json
import math
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hpa_planner_node import HPAPlannerNode, main as hpa_main, parser as hpa_parser, validate_args as hpa_validate
from shadow_core import Rejected

# planar_planner_node map/footprint contract (unknown unsafe, occupancy threshold 50,
# footprint cleared by r_safe + 0.05 before each solve), with the unknown boundary
# inflated by r_quad 0.31 (docs/THREE_ARM_TEST30B_RESULTS_20260928.md 4.2.3).
OCC_THRESH = 50
UNKNOWN_UNSAFE = True
UNKNOWN_INFLATE = 0.31
FOOTPRINT_EXTRA = 0.05


def parser():
    result = hpa_parser()
    result.description = __doc__
    result.add_argument("--grid-topic", default="/grid_map")
    result.add_argument("--haa-backend", choices=("numpy", "cuda"), default="numpy",
                        help="MPPI sample-batch evaluation for the HAA and its probe; cuda is "
                             "planner/haa/cuda_batch.py (same double arithmetic, verified identical)")
    result.add_argument("--snapshot-dir", type=Path,
                        help="save every distinct occupancy grid used (npz) for the simulator comparison")
    result.add_argument("--supervisor-worker", action="store_true",
                        help="run the supervisor (grid parsing, footprint clearing, plan, snapshots) in a "
                             "separate process (desimplex_core.py) so it does not share the GIL with the "
                             "ROS callbacks and the 50 Hz output; same computation")
    result.add_argument("--parallel-probe", action="store_true",
                        help="with --supervisor-worker: solve the n_look end-point recovery probe in a helper "
                             "process while the one-step probe runs (desimplex_core.ParallelLookProbe); same answers")
    result.add_argument("--probe-cache", action="store_true",
                        help="with --supervisor-worker: answer a repeated S_HAA probe of the same state within one "
                             "tick from the first solve (desimplex_core.TickProbeCache); same answers")
    return result


def validate_args(args):
    hpa_validate(args)
    if args.goal_current_position:
        raise ValueError("DeSimplex needs an explicit --goal-local (HAA plans to it)")
    for flag in ("parallel_probe", "probe_cache"):
        if getattr(args, flag, False) and not getattr(args, "supervisor_worker", False):
            raise ValueError("--%s needs --supervisor-worker" % flag.replace("_", "-"))
    if not args.alignment_topic:
        raise ValueError("DeSimplex needs the shared alignment: HAA plans on the world-frame grid")
    if args.output_topic == "/hpa_shadow/position_target":
        args.output_topic = "/hpa_shadow/desimplex/position_target"
    if args.arrived_topic == "/hpa_shadow/goal_arrive_tf":
        args.arrived_topic = "/hpa_shadow/desimplex/goal_arrive_tf"


def _array(value):
    return None if value is None else np.asarray(value, dtype=float).tolist()


class DeSimplexPlannerNode(HPAPlannerNode):
    worker = None            # desimplex_core.SupervisorWorker with --supervisor-worker
    worker_grid_seq = None   # grid seq the worker holds

    def __init__(self, args, rospy, bundle, manifest, model_info, log, controller):
        self.grid = None
        self.grid_seq = 0
        self.goal_world = None
        self.ticks = 0
        self.planning_chunk = None
        self.snapshot_dir = args.snapshot_dir
        super().__init__(args, rospy, bundle, manifest, model_info, log, controller)
        if getattr(args, "supervisor_worker", False):
            from desimplex_core import SupervisorWorker
            self.worker = SupervisorWorker(self.producer_config, getattr(args, "haa_backend", "numpy"),
                                           self.snapshot_dir, parallel_probe=getattr(args, "parallel_probe", False),
                                           probe_cache=getattr(args, "probe_cache", False))
            self.emit("desimplex_worker", pid=self.worker.pid, haa_backend=getattr(args, "haa_backend", "numpy"),
                      parallel_probe=getattr(args, "parallel_probe", False),
                      probe_cache=getattr(args, "probe_cache", False), warmup_ms=self.worker.warmup_ms)

    # --------------------------------------------------------- construction

    def build_assembly(self, build_producer):
        cfg = self.producer_config
        for key in ("haa", "hpa", "safety", "haa_true_limits"):
            if key not in cfg:
                raise Rejected("DeSimplex producer config lacks section " + key)
        self._build_producer = build_producer
        self.r_safe = float(cfg["safety"]["r_safe"])
        return None      # built on the first grid of the bound alignment epoch

    def hpa_producer(self):
        return None if self.assembly is None else self.assembly.hpa

    def ticks_until_refresh(self):
        """As HPAPlannerNode, counting the bridge ticks that consume the committed plan first."""
        if self.worker is not None:      # computed by the worker after its last tick
            return self.worker.ticks_until_refresh
        from planner.hpa.commit import ticks_until_fresh
        if self.assembly is None:
            return 1
        bridge = getattr(self.assembly.producer, "_bridge", None)
        return ticks_until_fresh(self.commit, getattr(self.assembly.hpa, "_commit_state", None),
                                 bridge_ticks=0 if bridge is None else int(bridge[1]) + 1)

    def tick_needs_observation(self):
        if self.worker is not None:      # computed by the worker after its last tick
            return self.worker.next_needs_observation
        # During a bridge the supervisor does not call the HPA at all
        # (planner/supervisor.py _decide), so no policy observation is due.
        if self.assembly is not None and getattr(self.assembly.producer, "_bridge", None) is not None:
            return False
        return super().tick_needs_observation()

    def producer_goal_tol(self):
        # The factory requires hpa.goal_tol == haa.goal_tol for DeSimplex.
        return float(self.producer_config["hpa"]["goal_tol"])

    def extra_subscribers(self, rospy):
        from nav_msgs.msg import OccupancyGrid
        return [rospy.Subscriber(self.args.grid_topic, OccupancyGrid, self.grid_callback,
                                 queue_size=1, buff_size=2 ** 22)]

    # ------------------------------------------------------------------ map

    def grid_callback(self, msg):
        """planar_planner_node._grid_cb: current-epoch world grid only."""
        from planner.grid import PlanarOccupancy
        with self.lock:
            now = self.rospy.Time.now().to_sec()
            if not self.alignment.is_ready(now=now, max_age=self.args.alignment_max_age):
                return
            epoch, valid_from = self.alignment.epoch, self.alignment.valid_from
            if (msg.header.frame_id != self.alignment.world_frame or valid_from is None or
                    msg.header.stamp.to_sec() < valid_from or
                    abs(msg.info.map_load_time.to_sec() - valid_from) > 1e-6):
                self.counts["grid_rejected_epoch_or_frame"] += 1
                return
        if self.worker is not None:
            # The worker parses it (desimplex_core.SupervisorCore._parse_grid).
            data = np.asarray(msg.data, dtype=np.int8)
            if data.size != msg.info.width * msg.info.height:
                self.counts["grid_parse_failed"] += 1
                self.reject("grid parse failed: %d cells for %dx%d" % (data.size, msg.info.width, msg.info.height))
                return
            with self.lock:
                if epoch == self.alignment.epoch and self.alignment.ready:
                    self.grid_seq += 1
                    self.grid = dict(epoch=epoch, occ=None, stamp=msg.header.stamp.to_sec(), seq=self.grid_seq,
                                     msg=dict(frame_id=msg.header.frame_id, width=msg.info.width,
                                              height=msg.info.height, resolution=msg.info.resolution,
                                              origin=[msg.info.origin.position.x, msg.info.origin.position.y],
                                              data=data))
                    self.counts["grids"] += 1
            return
        try:
            occ = PlanarOccupancy.from_occupancy_grid_msg(
                msg, occ_thresh=OCC_THRESH, unknown_unsafe=UNKNOWN_UNSAFE,
                unknown_inflate=UNKNOWN_INFLATE, r_safe=self.r_safe)
        except Exception as error:      # parse errors never reach the planner
            self.counts["grid_parse_failed"] += 1
            self.reject("grid parse failed: %s" % error)
            return
        with self.lock:
            if epoch == self.alignment.epoch and self.alignment.ready:
                self.grid_seq += 1
                self.grid = dict(epoch=epoch, occ=occ, stamp=msg.header.stamp.to_sec(), seq=self.grid_seq,
                                 msg=dict(frame_id=msg.header.frame_id, width=msg.info.width, height=msg.info.height,
                                          resolution=msg.info.resolution,
                                          origin=[msg.info.origin.position.x, msg.info.origin.position.y],
                                          data=np.asarray(msg.data, dtype=np.int8)))
                self.counts["grids"] += 1

    def emit(self, event, **fields):
        super().emit(event, **fields)
        # A computed tick whose reference is not flown (late/stale at completion,
        # expired at acceptance) advanced the supervisor all the same. The
        # simulator flies every non-FAILED tick, so count and log each one.
        if event == "prediction" and isinstance(fields.get("planner"), dict):
            flown = self.lifecycle.last_anchor == fields["anchor_stamp"]
            if not flown:
                self.counts["desimplex_tick_not_flown"] += 1
                super().emit("desimplex_tick_not_flown", tick=fields["planner"]["tick"],
                             anchor_stamp=fields["anchor_stamp"],
                             discard_reason=fields.get("discard_reason") or self.lifecycle.reason)

    def _save_grid(self, grid):
        if self.snapshot_dir is None:
            return None
        path = self.snapshot_dir / ("grid_%06d.npz" % grid["seq"])
        if not path.exists():
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            m = grid["msg"]
            np.savez_compressed(path, frame_id=m.get("frame_id", ""), data=m["data"], width=m["width"], height=m["height"],
                                resolution=m["resolution"], origin=np.asarray(m["origin"]),
                                stamp=grid["stamp"], epoch=grid["epoch"])
        return path.name

    # ---------------------------------------------------------------- tick

    def _plan_tick(self, xi, a_prev, stamp, epoch):
        from planner.types import MPPIResult, PlanarReferenceSequence
        from planner.hpa import BodyActionChunk
        with self.lock:
            now = self.rospy.Time.now().to_sec()
            if not self.alignment.is_ready(now=now, max_age=self.args.alignment_max_age):
                raise Rejected("waiting for shared alignment")
            if self.alignment.valid_from is None or stamp < self.alignment.valid_from:
                raise Rejected("observation predates alignment epoch")
            grid = self.grid
            if grid is None or grid["epoch"] != self.alignment.epoch:
                raise Rejected("waiting for current-epoch grid")
            align = self.alignment
            yaw_offset = float(align.yaw)
            translation = np.asarray(align.translation, dtype=float)
            if self.goal_world is None:
                self.goal_world = align.local_to_world(np.asarray(self.goal[:3], dtype=float))[:2]
                self.emit("desimplex_goal_bound", goal_local_enu=self.goal.tolist(),
                          goal_world=self.goal_world.tolist(), alignment_epoch=align.epoch)
        rot = np.array([[math.cos(yaw_offset), -math.sin(yaw_offset)],
                        [math.sin(yaw_offset), math.cos(yaw_offset)]])
        xi = np.asarray(xi, dtype=float)
        p_w = rot.dot(xi[:2]) + translation[:2]
        v_w = rot.dot(xi[2:4])
        xi_w = np.array([p_w[0], p_w[1], v_w[0], v_w[1],
                         math.atan2(math.sin(xi[4] + yaw_offset), math.cos(xi[4] + yaw_offset)), xi[5]])
        a_prev_w = rot.dot(np.asarray(a_prev, dtype=float))
        # The chunk's anchor_yaw is the PLANNING-frame yaw its body actions are
        # rotated by (planner.hpa.reference.world_actions): world here.
        chunk = self.current_chunk
        # Body actions are rotated by the yaw the policy OBSERVED (the chunk is
        # computed ahead of the tick that starts it), expressed in the world frame.
        chunk_yaw_w = math.atan2(math.sin(chunk.anchor_yaw + yaw_offset), math.cos(chunk.anchor_yaw + yaw_offset))
        if self.worker is not None:
            return self._worker_tick(xi_w, a_prev_w, stamp, grid, chunk, rot, yaw_offset, translation, chunk_yaw_w)
        self.planning_chunk = BodyActionChunk(chunk.actions, chunk_yaw_w)
        occ = grid["occ"]
        occ.clear_disc(p_w[0], p_w[1], self.r_safe + FOOTPRINT_EXTRA)
        if self.assembly is None:
            self.assembly = self._build_producer(
                "desimplex", self.producer_config, occupancy=occ, goal=self.goal_world,
                action_provider=lambda state, goal: self.planning_chunk)
            if getattr(self.args, "haa_backend", "numpy") == "cuda":
                from planner.haa.cuda_batch import enable_cuda
                backend = enable_cuda(self.assembly.haa, self.assembly.probe)
                self.emit("haa_backend", backend="cuda", library=backend.path, shared_map_cache=True)
            self.emit("desimplex_constructed", producer=self.assembly.effective_config,
                      grid_seq=grid["seq"], goal_world=self.goal_world.tolist())
        supervisor = self.assembly.producer
        supervisor.set_occupancy(occ)
        self.ticks += 1
        t_sup = time.perf_counter()
        try:
            result = supervisor.plan(xi_w, goal=self.goal_world, a_prev=a_prev_w)
        except Exception as error:
            # The supervisor's counters/bridge may be half-updated: never continue.
            with self.lock:
                self.emit("desimplex_tick", anchor_stamp=stamp, tick=self.ticks, grid_seq=grid["seq"],
                          grid_file=self._save_grid(grid), xi_world=xi_w.tolist(),
                          goal_world=self.goal_world.tolist(), a_prev_world=a_prev_w.tolist(),
                          chunk_actions=_array(chunk.actions), chunk_anchor_yaw_world=chunk_yaw_w,
                          exception=repr(error))
                self.invalidate_locked("DeSimplex supervisor exception: %r" % (error,))
            raise RuntimeError("DeSimplex supervisor exception") from error
        supervisor_ms = 1000 * (time.perf_counter() - t_sup)
        d = supervisor.last_decision
        t_save = time.perf_counter()
        grid_file = self._save_grid(grid)
        save_ms = 1000 * (time.perf_counter() - t_save)
        hst = getattr(self.assembly.hpa, "_commit_state", None)
        log = dict(tick=self.ticks, grid_seq=grid["seq"], hpa_commit_index=None if hst is None else int(hst["i"]), grid_file=grid_file,
                   supervisor_ms=supervisor_ms, grid_save_ms=save_ms,
                   xi_world=xi_w.tolist(), goal_world=self.goal_world.tolist(), a_prev_world=a_prev_w.tolist(),
                   alignment=dict(epoch=grid["epoch"], yaw=yaw_offset, translation=translation.tolist()),
                   chunk_actions=_array(getattr(chunk, "actions", None)),
                   chunk_anchor_yaw_local=chunk.anchor_yaw, chunk_anchor_yaw_world=chunk_yaw_w,
                   status=str(result.status), result_reason=result.reason,
                   mode=getattr(d, "mode", None), source=getattr(d, "source", None),
                   decision_reason=getattr(d, "reason", None), fault=bool(getattr(d, "fault", False)),
                   switched=bool(getattr(d, "switched", False)),
                   bridge_active=getattr(supervisor, "_bridge", None) is not None,
                   U_world=_array(result.U), X_world=_array(result.X),
                   fault_note="fault=True is best-effort braking, not a verified safe recovery")
        ref = result.reference
        brief = dict(tick=self.ticks, source=log["source"], mode=log["mode"], fault=log["fault"],
                     bridge_active=log["bridge_active"], status=log["status"])
        if ref is None:
            with self.lock:     # every tick is logged, FAILED ones included
                self.emit("desimplex_tick", anchor_stamp=stamp, **log)
            return result, brief
        log["reference_world"] = dict(p=_array(ref.p), v=_array(ref.v), a=_array(ref.a),
                                      psi=_array(ref.psi), psi_dot=_array(ref.psi_dot), dt=float(ref.dt))
        local = _to_local(ref.p, ref.v, ref.a, ref.psi, ref.psi_dot, ref.dt, rot, yaw_offset, translation)
        with self.lock:
            self.emit("desimplex_tick", anchor_stamp=stamp, **log)
        return MPPIResult(result.status, local, result.U, result.X, result.cost, result.n_valid,
                          result.n_samples, result.beta, result.reason), brief

    def _worker_tick(self, xi_w, a_prev_w, stamp, grid, chunk, rot, yaw_offset, translation, chunk_yaw_w):
        """_plan_tick's supervisor part in the worker: same request, same log record."""
        from planner.types import MPPIResult
        m = grid["msg"]
        req = dict(grid_seq=grid["seq"], xi_world=xi_w, goal_world=self.goal_world, a_prev_world=a_prev_w,
                   chunk_actions=np.asarray(chunk.actions, dtype=float), chunk_anchor_yaw_world=chunk_yaw_w,
                   grid=None if self.worker_grid_seq == grid["seq"] else
                   dict(m, seq=grid["seq"], epoch=grid["epoch"], stamp=grid["stamp"]))
        self.ticks += 1
        t_call = time.perf_counter()
        try:
            r = self.worker.tick(req)
        except Exception as error:
            with self.lock:
                self.emit("desimplex_tick", anchor_stamp=stamp, tick=self.ticks, grid_seq=grid["seq"],
                          grid_file=None, xi_world=xi_w.tolist(),
                          goal_world=self.goal_world.tolist(), a_prev_world=a_prev_w.tolist(),
                          chunk_actions=_array(chunk.actions), chunk_anchor_yaw_world=chunk_yaw_w,
                          exception=repr(error))
                self.invalidate_locked("DeSimplex supervisor exception: %r" % (error,))
            raise RuntimeError("DeSimplex supervisor exception") from error
        worker_ms = 1000 * (time.perf_counter() - t_call)
        self.worker_grid_seq = grid["seq"]
        if r["tick"] != self.ticks:
            with self.lock:
                self.invalidate_locked("DeSimplex worker tick %s != node tick %s" % (r["tick"], self.ticks))
            raise RuntimeError("DeSimplex worker tick mismatch")
        if r["backend"] is not None:
            self.emit("haa_backend", **r["backend"])
        if r["constructed"] is not None:
            self.emit("desimplex_constructed", producer=r["constructed"], grid_seq=grid["seq"],
                      goal_world=self.goal_world.tolist())
        log = dict(tick=self.ticks, grid_seq=grid["seq"], hpa_commit_index=r["hpa_commit_index"],
                   grid_file=r["grid_file"], supervisor_ms=r["supervisor_ms"], grid_save_ms=r["grid_save_ms"],
                   grid_save_deferred_ms=r.get("grid_save_deferred_ms"),
                   worker_roundtrip_ms=worker_ms, look_probe=r["look_probe"], probe_cache=r["probe_cache"],
                   xi_world=xi_w.tolist(), goal_world=self.goal_world.tolist(), a_prev_world=a_prev_w.tolist(),
                   alignment=dict(epoch=grid["epoch"], yaw=yaw_offset, translation=translation.tolist()),
                   chunk_actions=_array(chunk.actions),
                   chunk_anchor_yaw_local=chunk.anchor_yaw, chunk_anchor_yaw_world=chunk_yaw_w,
                   status=r["status"], result_reason=r["result_reason"],
                   mode=r["mode"], source=r["source"], decision_reason=r["decision_reason"], fault=r["fault"],
                   switched=r["switched"], bridge_active=r["bridge_active"],
                   U_world=_array(r["U"]), X_world=_array(r["X"]),
                   fault_note="fault=True is best-effort braking, not a verified safe recovery")
        brief = dict(tick=self.ticks, source=log["source"], mode=log["mode"], fault=log["fault"],
                     bridge_active=log["bridge_active"], status=log["status"])
        ref = r["reference"]
        local = None
        if ref is not None:
            log["reference_world"] = dict((k, _array(v)) if k != "dt" else (k, v) for k, v in ref.items())
            local = _to_local(ref["p"], ref["v"], ref["a"], ref["psi"], ref["psi_dot"], ref["dt"],
                              rot, yaw_offset, translation)
        with self.lock:     # every tick is logged, FAILED ones included
            self.emit("desimplex_tick", anchor_stamp=stamp, **log)
        return MPPIResult(r["status"], local, r["U"], r["X"], r["cost"], r["n_valid"], r["n_samples"],
                          r["beta"], r["result_reason"]), brief


def _to_local(p, v, a, psi, psi_dot, dt, rot, yaw_offset, translation):
    """World-frame reference -> PX4 local ENU with the tick's epoch alignment."""
    from planner.types import PlanarReferenceSequence
    inv = rot.T
    psi = np.asarray(psi)
    return PlanarReferenceSequence(
        p=(np.asarray(p) - translation[:2]).dot(inv.T), v=np.asarray(v).dot(inv.T), a=np.asarray(a).dot(inv.T),
        psi=np.arctan2(np.sin(psi - yaw_offset), np.cos(psi - yaw_offset)),
        psi_dot=np.asarray(psi_dot), dt=dt)


def main(argv=None):
    return hpa_main(argv, node_class=DeSimplexPlannerNode, node_name="desimplex_planner_node",
                    make_parser=parser, check_args=validate_args)


if __name__ == "__main__":
    sys.exit(main())
