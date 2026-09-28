"""One DeSimplex supervisor tick, without ROS: in the node's process or a worker.

`SupervisorCore.tick(request) -> reply` owns everything the ORIGINAL
DeSimplexSupervisor needs and nothing the node needs: the factory assembly
(built on the first tick), the current-epoch occupancy grid (parsed here,
footprint cleared on the same object every tick as planar_planner_node does),
the optional CUDA MPPI backend and grid snapshots. The request carries the
world-frame state, goal, a_prev, the V4 chunk with its world anchor yaw, and a
new grid only when its sequence number changed.

`SupervisorWorker` runs the same core in a separate process so the supervisor's NumPy work does not share a GIL with the
node's ROS callbacks and 50 Hz output. Requests and replies are plain dicts of
NumPy arrays; the computation is identical in both modes.
"""
from multiprocessing.connection import Connection
import os
import socket
import subprocess
import sys
from pathlib import Path
import time
import traceback
from types import SimpleNamespace as NS

import numpy as np

# planar_planner_node map/footprint contract.
OCC_THRESH = 50
UNKNOWN_UNSAFE = True
UNKNOWN_INFLATE = 0.0
FOOTPRINT_EXTRA = 0.05


def _list(value):
    return None if value is None else np.asarray(value, dtype=float)


def _parse_grid(g, r_safe):
    from planner.grid import PlanarOccupancy
    msg = NS(header=NS(frame_id=g["frame_id"]),
             info=NS(width=int(g["width"]), height=int(g["height"]), resolution=float(g["resolution"]),
                     origin=NS(position=NS(x=float(g["origin"][0]), y=float(g["origin"][1])))),
             data=np.asarray(g["data"], dtype=np.int8))
    return PlanarOccupancy.from_occupancy_grid_msg(msg, occ_thresh=OCC_THRESH, unknown_unsafe=UNKNOWN_UNSAFE,
                                                   unknown_inflate=UNKNOWN_INFLATE, r_safe=r_safe)


def _warm_imports(haa_backend):
    # Import now (cv2/scipy: ~0.8 s on Xavier), not inside the first 10 Hz tick.
    import planar_producer_factory  # noqa: F401
    import planner.grid  # noqa: F401
    import planner.hpa  # noqa: F401
    if haa_backend == "cuda":
        from planner.haa.cuda_batch import enable_cuda
        enable_cuda()     # library load + CUDA context now, not in the first tick


def _look_end(sup, hpa_ref):
    """The (state, a_prev) DeSimplexSupervisor._committed_ok probes, or None."""
    from planner.types import NXI, S_POS, S_VEL
    if hpa_ref is None or not hpa_ref.n_nodes or sup.n_look <= 1:
        return None
    i = min(sup.n_look, hpa_ref.n_nodes) - 1
    xi_end = np.empty(NXI, dtype=np.float64)
    xi_end[S_POS] = hpa_ref.p[i]
    xi_end[S_VEL] = hpa_ref.v[i]
    xi_end[4] = float(hpa_ref.psi[i])
    xi_end[5] = float(hpa_ref.psi_dot[i])
    return xi_end, np.array(hpa_ref.a[i], dtype=np.float64)


class ParallelLookProbe(object):
    """Solve the n_look end-point recovery in a helper process, concurrently.

    Per HPA-mode tick DeSimplexSupervisor.step makes two recovery() calls: the
    one-step R_Nr test on f(x, u_hpa) and, only if that passes, _committed_ok's
    test on the end of the committed window. Both are pure functions of (state,
    a_prev, goal, grid, r_safe): in_s_haa resets the probe's nominal AND its RNG
    before every solve and the map cache compares contents. So the second one
    is started when step() begins (hpa_ref is known then) on an identical
    supervisor that holds an identical grid -- same parse, same cumulative
    footprint clearing -- and recovery() hands back its result when called
    with exactly those arguments, adding the probe counters it would have
    added. If the serial code would not have asked (the one-step test failed),
    the answer is discarded, exactly as the question was never asked.
    """

    def __init__(self, producer_config, haa_backend):
        self.channel = _Channel("--probe-fd", dict(producer_config=producer_config, haa_backend=haa_backend))
        self.grid_seq = None      # grid the helper holds
        self.clears = []          # footprint clears on the current grid not yet sent
        self.new_grid = None
        self.job = None           # (xi, a_prev) in flight
        self.outstanding = False
        self.used = self.discarded = 0

    def on_grid(self, grid_meta):
        self.new_grid, self.clears = grid_meta, []

    def on_clear(self, x, y):
        self.clears.append((float(x), float(y)))

    def install(self, sup):
        step, recovery = sup.step, sup.recovery

        from planner.supervisor import MODE_HPA

        def parallel_step(xi, u_hpa, a_prev=None, hpa_ref=None):
            # Only a tick that starts in HPA mode is likely to ask; a hand-back
            # tick falls through to the serial recovery(), which is the same answer.
            job = _look_end(sup, hpa_ref) if sup.mode == MODE_HPA else None
            if job is not None:
                self._start(job, sup.goal)
            return step(xi, u_hpa, a_prev=a_prev, hpa_ref=hpa_ref)

        def parallel_recovery(xi, a_prev=None):
            job = self.job
            if (job is not None and a_prev is not None and np.array_equal(np.asarray(xi, dtype=np.float64), job[0])
                    and np.array_equal(np.asarray(a_prev, dtype=np.float64), job[1])):
                r = self._finish()
                sup.n_probe += r["n_probe"]
                sup.probe_s += r["probe_s"]
                sup.n_decel += r["n_decel"]
                self.used += 1
                return r["result"]
            return recovery(xi, a_prev=a_prev)

        sup.step, sup.recovery = parallel_step, parallel_recovery

    def _start(self, job, goal):
        if self.outstanding:        # the previous answer was never asked for
            self._finish()
            self.discarded += 1
        req = dict(xi=job[0], a_prev=job[1], goal=np.asarray(goal, dtype=np.float64), clears=self.clears,
                   grid=self.new_grid)
        self.channel.conn.send(req)
        if self.new_grid is not None:
            self.grid_seq = self.new_grid["seq"]
        self.new_grid, self.clears = None, []
        self.job, self.outstanding = job, True

    def _finish(self):
        msg = self.channel.recv(5.0, "answer the look-ahead probe")
        self.job, self.outstanding = None, False
        return msg["reply"]

    def close(self):
        self.channel.close()


class TickProbeCache(object):
    """Answer a repeated S_HAA membership question within ONE supervisor tick from the first answer.

    A hand-back tick whose bridge cannot be built asks in_s_haa(x, a_prev) for
    the SAME state three times: in_m, again in_m after the hand-back is
    withdrawn, and recovery()'s k = 0 probe (recovery_margin 0). in_s_haa is a
    pure function of (state, a_prev, goal, grid, validator radius): it resets
    the probe's nominal and RNG before solving, and nothing outside it reads
    the probe's state. So the repeat returns a copy of the first answer and
    adds the probe counters the solve would have added. The cache is emptied
    at the start of every plan() call; the grid does not change inside one.
    """

    def __init__(self):
        self.memo = {}
        self.hits = self.solves = 0

    def install(self, sup):
        import copy
        plan, in_s_haa = sup.plan, sup.in_s_haa

        def cached_plan(*a, **k):
            self.memo = {}
            return plan(*a, **k)

        def cached_in_s_haa(xi, a_prev=None):
            xi = np.asarray(xi, dtype=np.float64).reshape(-1)
            key = (xi.tobytes(), None if a_prev is None else np.asarray(a_prev, dtype=np.float64).tobytes(),
                   float(sup.validator.r_safe), np.asarray(sup.goal, dtype=np.float64).tobytes(),
                   id(sup.probe.validator.occ))
            hit = self.memo.get(key)
            if hit is not None:
                ok, res, solved, dt = hit
                if solved:
                    sup.n_probe += 1
                    sup.probe_s += dt
                self.hits += 1
                return ok, copy.deepcopy(res)
            n0, s0 = sup.n_probe, sup.probe_s
            ok, res = in_s_haa(xi, a_prev=a_prev)
            solved = sup.n_probe != n0
            self.solves += solved
            self.memo[key] = (ok, copy.deepcopy(res), solved, sup.probe_s - s0)
            return ok, res

        sup.plan, sup.in_s_haa = cached_plan, cached_in_s_haa


class ProbeHelper(object):
    """Helper-process side of ParallelLookProbe: an identical supervisor, recovery() only."""

    def __init__(self, producer_config, haa_backend="numpy"):
        self.config = producer_config
        self.r_safe = float(producer_config["safety"]["r_safe"])
        self.haa_backend = haa_backend
        self.occ = None
        self.assembly = None
        _warm_imports(haa_backend)

    def tick(self, req):
        from planar_producer_factory import build_producer
        if req["grid"] is not None:
            self.occ = _parse_grid(req["grid"], self.r_safe)
        for x, y in req["clears"]:
            self.occ.clear_disc(x, y, self.r_safe + FOOTPRINT_EXTRA)
        if self.assembly is None:
            self.assembly = build_producer("desimplex", self.config, occupancy=self.occ, goal=req["goal"],
                                           action_provider=lambda state, goal: None)
            if self.haa_backend == "cuda":
                from planner.haa.cuda_batch import enable_cuda
                enable_cuda(self.assembly.haa, self.assembly.probe)
        sup = self.assembly.producer
        sup.set_occupancy(self.occ)
        sup.goal = np.asarray(req["goal"], dtype=np.float64).reshape(-1)[:2]
        n0, s0, d0 = sup.n_probe, sup.probe_s, sup.n_decel
        result = sup.recovery(req["xi"], a_prev=req["a_prev"])
        return dict(result=result, n_probe=sup.n_probe - n0, probe_s=sup.probe_s - s0, n_decel=sup.n_decel - d0)


class SupervisorCore(object):
    def __init__(self, producer_config, haa_backend="numpy", snapshot_dir=None, parallel_probe=False,
                 probe_cache=False):
        self.config = producer_config
        self.r_safe = float(producer_config["safety"]["r_safe"])
        self.haa_backend = haa_backend
        self.snapshot_dir = None if snapshot_dir is None else Path(snapshot_dir)
        self.assembly = None
        self.grid = None           # dict(seq, epoch, occ, meta)
        self.holder = {"chunk": None}
        self.ticks = 0
        _warm_imports(haa_backend)
        self.look = ParallelLookProbe(producer_config, haa_backend) if parallel_probe else None
        self.cache = TickProbeCache() if probe_cache else None

    def _save(self, g):
        if self.snapshot_dir is None:
            return None, 0.0
        t0 = time.perf_counter()
        path = self.snapshot_dir / ("grid_%06d.npz" % g["seq"])
        if not path.exists():
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frame_id=g["frame_id"], data=np.asarray(g["data"], dtype=np.int8),
                                width=g["width"], height=g["height"], resolution=g["resolution"],
                                origin=np.asarray(g["origin"]), stamp=g["stamp"], epoch=g["epoch"])
        return path.name, 1000 * (time.perf_counter() - t0)

    def needs_observation(self):
        """True when the next tick's supervisor call will ask the HPA for a new chunk."""
        if self.assembly is None:
            return True
        if getattr(self.assembly.producer, "_bridge", None) is not None:
            return False
        st = getattr(self.assembly.hpa, "_commit_state", None)
        commit = int(self.config["hpa"].get("commit", 1))
        return st is None or st["res"] is None or st["i"] >= commit

    def tick(self, req):
        from planar_producer_factory import build_producer
        from planner.hpa import BodyActionChunk
        reply = dict(constructed=None, backend=None)
        g = req.get("grid")
        if g is not None and (self.grid is None or g["seq"] != self.grid["seq"]):
            self.grid = dict(seq=g["seq"], epoch=g["epoch"], occ=_parse_grid(g, self.r_safe), meta=g)
            if self.look is not None:
                self.look.on_grid(g)
        if self.grid is None or self.grid["seq"] != req["grid_seq"]:
            raise RuntimeError("request references grid %s the core does not hold" % req["grid_seq"])
        grid_file, save_ms = self._save(self.grid["meta"])
        xi_w = np.asarray(req["xi_world"], dtype=float)
        goal_w = np.asarray(req["goal_world"], dtype=float)
        occ = self.grid["occ"]
        occ.clear_disc(xi_w[0], xi_w[1], self.r_safe + FOOTPRINT_EXTRA)
        if self.look is not None:
            self.look.on_clear(xi_w[0], xi_w[1])
        if req.get("chunk_actions") is not None:
            self.holder["chunk"] = BodyActionChunk(np.asarray(req["chunk_actions"], dtype=float),
                                                   float(req["chunk_anchor_yaw_world"]))
        if self.assembly is None:
            self.assembly = build_producer("desimplex", self.config, occupancy=occ, goal=goal_w,
                                           action_provider=lambda state, goal: self.holder["chunk"])
            if self.haa_backend == "cuda":
                from planner.haa.cuda_batch import enable_cuda
                backend = enable_cuda(self.assembly.haa, self.assembly.probe)
                reply["backend"] = dict(backend="cuda", library=backend.path, shared_map_cache=True)
            reply["constructed"] = self.assembly.effective_config
            if self.look is not None:
                self.look.install(self.assembly.producer)
            if self.cache is not None:
                self.cache.install(self.assembly.producer)
        supervisor = self.assembly.producer
        supervisor.set_occupancy(occ)
        self.ticks += 1
        t0 = time.perf_counter()
        result = supervisor.plan(xi_w, goal=goal_w, a_prev=np.asarray(req["a_prev_world"], dtype=float))
        supervisor_ms = 1000 * (time.perf_counter() - t0)
        d = supervisor.last_decision
        hst = getattr(self.assembly.hpa, "_commit_state", None)
        ref = result.reference
        reply.update(
            tick=self.ticks, grid_file=grid_file, grid_save_ms=save_ms, supervisor_ms=supervisor_ms,
            status=str(result.status), result_reason=result.reason, cost=float(result.cost),
            n_valid=int(result.n_valid), n_samples=int(result.n_samples), beta=float(result.beta),
            U=_list(result.U), X=_list(result.X),
            mode=getattr(d, "mode", None), source=getattr(d, "source", None),
            decision_reason=getattr(d, "reason", None), fault=bool(getattr(d, "fault", False)),
            switched=bool(getattr(d, "switched", False)),
            bridge_active=getattr(supervisor, "_bridge", None) is not None,
            hpa_commit_index=None if hst is None else int(hst["i"]),
            reference=None if ref is None else dict(p=np.asarray(ref.p), v=np.asarray(ref.v), a=np.asarray(ref.a),
                                                     psi=np.asarray(ref.psi), psi_dot=np.asarray(ref.psi_dot),
                                                     dt=float(ref.dt)),
            next_needs_observation=self.needs_observation(),
            look_probe=None if self.look is None else dict(used=self.look.used, discarded=self.look.discarded),
            probe_cache=None if self.cache is None else dict(hits=self.cache.hits, solves=self.cache.solves))
        return reply


def _serve(conn, make):
    """Worker/helper process body: one object, one request at a time."""
    init = conn.recv()
    try:
        obj = make(**init)
        conn.send(dict(ok=True, ready=True, pid=os.getpid()))
    except Exception as error:
        conn.send(dict(ok=False, error=repr(error), trace=traceback.format_exc()))
        return
    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        if req is None:
            return
        try:
            conn.send(dict(ok=True, reply=obj.tick(req)))
        except Exception as error:  # the supervisor may be half-updated: the node stops
            conn.send(dict(ok=False, error=repr(error), trace=traceback.format_exc()))


class _Channel(object):
    """A child Python process running this file, pickled dicts over a socketpair.

    A plain subprocess (not multiprocessing spawn, which would re-import the
    node's __main__ and with it torch). The child inherits this process's
    sys.path, so it imports the same planner.
    """

    def __init__(self, role_flag, init, start_timeout=60.0):
        parent, child = socket.socketpair()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), role_flag, str(child.fileno())],
                                     pass_fds=(child.fileno(),), env=env)
        child.close()
        self.conn = Connection(parent.detach())
        self.conn.send(init)
        self.pid = self.recv(start_timeout, "start")["pid"]

    def recv(self, timeout, what):
        if not self.conn.poll(timeout):
            raise RuntimeError("DeSimplex child did not %s within %.1f s" % (what, timeout))
        try:
            msg = self.conn.recv()
        except EOFError:
            raise RuntimeError("DeSimplex child exited (code %s)" % self.proc.poll())
        if not msg["ok"]:
            raise RuntimeError("DeSimplex child: %s\n%s" % (msg["error"], msg.get("trace", "")))
        return msg

    def close(self):
        try:
            self.conn.send(None)
        except (OSError, EOFError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.conn.close()


class SupervisorWorker(object):
    """The same SupervisorCore in a separate Python process."""

    def __init__(self, producer_config, haa_backend="numpy", snapshot_dir=None, parallel_probe=False,
                 probe_cache=False, reply_timeout=5.0):
        self.channel = _Channel("--fd", dict(producer_config=producer_config, haa_backend=haa_backend,
                                             snapshot_dir=None if snapshot_dir is None else str(snapshot_dir),
                                             parallel_probe=parallel_probe, probe_cache=probe_cache))
        self.pid = self.channel.pid
        self.reply_timeout = reply_timeout
        self.next_needs_observation = True

    def tick(self, req):
        self.channel.conn.send(req)
        reply = self.channel.recv(self.reply_timeout, "reply")["reply"]
        self.next_needs_observation = reply["next_needs_observation"]
        return reply

    def close(self):
        self.channel.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", type=int)
    ap.add_argument("--probe-fd", type=int)
    a = ap.parse_args()
    if a.probe_fd is not None:
        _serve(Connection(a.probe_fd), ProbeHelper)
    else:
        _serve(Connection(a.fd), SupervisorCore)
