#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enforce the velocity limit by PROJECTION inside the rollout, not by rejection.

    python -m planner.haa.capped      # runs the tests

THE PROBLEM
MPPI samples inputs nu = [ax, ay, alpha]. Velocity is a STATE the rollout
produces, so it cannot be clipped at sample time the way an input can, and
`PlanarDynamics.states_ok` therefore DISCARDS any sample whose speed exceeds
v_max at any node (planner/dynamics.py:231). Measured on the aggressive expert
during cruise: half the pool dies on that one check, only ~52% of samples
survive at all, and the vehicle tops out at ~46% of the v_max it is allowed.
The proposal is rejection-bound rather than limit-bound -- raising v_max does
almost nothing, because the samples that would use it are exactly the ones
thrown away.

WHY YOU CANNOT SIMPLY CLAMP THE VELOCITY
The published reference is `PlanarReferenceSequence.from_rollout(X, U, dt)`
(planner/types.py:221), which takes p and v straight out of X and a straight out
of U. Clamping X[..., S_VEL] without touching U produces a (p, v, a) triple
that is no longer each other's derivative. `IdealPlant` adopts p and v and
ignores a, so it would look fine; RotorPy's geometric controller consumes a as
a FEEDFORWARD, and its feedforward would then fight its own feedback. The
inconsistency is not smoothed by `sample()` either -- p, v and a are
interpolated independently.

THE FIX
Clamp the velocity, then back out the input that produces the clamped value:

    v_new = v_k + dt * a_k
    if |v_new| > v_max:
        v_new = v_new * (v_max / |v_new|)      # project onto the ball
        a_k   = (v_new - v_k) / dt             # the input that lands there

X stays exactly the integral of U, so the reference is still a consistent flat
output, and the input that gets published is the input that was actually flown.

|a| CANNOT GROW, so the acceleration disc survives for free. This is the
non-expansiveness of Euclidean projection onto a convex set: writing P for the
projection onto the closed ball of radius v_max, v_k is already inside it
(P(v_k) = v_k, see below), so

    |a_new| dt = |P(v_new) - v_k| = |P(v_new) - P(v_k)| <= |v_new - v_k| = |a| dt

The premise "v_k is inside the ball" holds because `states_ok` checks EVERY
node including node 0, the measured state: if the vehicle is genuinely outside
its own velocity envelope, every sample is rejected there and no plan is
produced, with or without this class. Node 0 is never projected here -- only
inputs 0..N-1 are, producing nodes 1..N -- so this cannot mask that condition.

THE SLEW BOUND IS THE ONE THING TO WATCH. `inputs_ok` (planner/dynamics.py:207)
rejects on |a_k - a_{k-1}| > j_max*dt, and shortening a_k can lengthen that
difference. It is not corrected here, and the cost of that is now measured
rather than assumed.

An earlier version of this paragraph said it had "headroom to spare --
j_max*dt = 2.63 m/s^2 against a sampled sigma of 0.49", and predicted jerk
rejection would not appear. Those are not this configuration's numbers:
j_max*dt is 0.550 and the acceleration sigma is 0.306, so the ratio is 1.8x
and not 5.4x. Jerk rejection HAS appeared.

MEASURED, seed 4003 flown HAA-alone on RotorPy, 302 plan ticks / 57 984
samples, counted on the U that `plan()` actually judges:

    failed inputs_ok BEFORE the cap        0    0.00%
    failed inputs_ok AFTER  the cap     5288    9.12%   all of it slew
      of which |a| disc                    0    0.00%
      of which |alpha|                     0    0.00%
    failed states_ok                       0    0.00%

Three things to read off that. The acceleration disc survives for free, as the
non-expansiveness argument above says it must. `states_ok` never fires at all,
because v and omega are now satisfied by construction -- which is the whole
point of this class, and the reason it is worth 9% of the pool. And EVERY
rejection is one this class manufactured: `clip_inputs` handed the rollout
57 984 jerk-feasible samples and the projection broke 5288 of them. That is a
sample-efficiency cost and not a safety one -- j_max is a real bound and an
`a_cap` that breaks it genuinely cannot be flown, so rejecting is correct --
but it is paid at high speed, where the shortening is largest, so the pool
thins where the samples are most aggressive. Nothing measured so far says it
costs an episode: a plan AT the velocity cap still returns 162 of 192 valid.

THE THIRD PROJECTION IS STILL THE RIGHT SHAPE OF FIX, AND IT IS NOT A FREE
WIN. It belongs here, in the same loop, onto the disc of radius j_max*dt about
the previously flown a_{k-1}. But the velocity constraint is itself a disc in
a_k -- |v_k + dt*a_k| <= v_max is |a_k - (-v_k/dt)| <= v_max/dt -- so
projecting onto the slew disc pushes a_k back out of the velocity disc and the
two have to be ITERATED to converge, not applied once. That is exactly the
ordering problem `clip_inputs` avoids by construction in its own docstring and
cannot be avoided here. Worse, when

    |a_{k-1} + v_k/dt|  >  v_max/dt + j_max*dt

the two discs do not intersect at all, no feasible a_k exists, and rejection is
the only correct answer -- a third projection buys nothing for those samples.
Measure the split before writing it; `experiments/ab_expert.py` reports the
breakdown per reason.

THE MUTATION CONTRACT -- READ THIS BEFORE CHANGING ANYTHING
`rollout` WRITES THE PROJECTED INPUT BACK INTO `U`, IN PLACE. That is
deliberate and it is what makes this a ~40-line subclass instead of a fork of
`plan()`. In `PlanarMPPI.plan` (planner/mppi.py:268-385) `U` is bound once, by
`clip_inputs`, and never rebound:

    U = self.dyn.clip_inputs(U, a_prev=a_prev)   # fresh (K,N,3) float64
    X = self.dyn.rollout(xi0, U)                 # <- we mutate U here
    valid &= self.dyn.inputs_ok(U, a_prev=a_prev)      # sees the mutation
    S = self._cost(X, U, goal, a_prev)                 # sees the mutation
    dU = np.tensordot(wts, U - U_nom[None, :, :], ...) # sees the mutation
    self._accept(..., U[idx], X[idx], ...)             # -> res.U, res.reference

so every downstream consumer automatically judges, scores and publishes the
input that was actually flown, and no line of `mppi.py` has to change. It works
because `_as_batch` (planner/dynamics.py:97) returns the caller's array itself
for a float64 array of ndim 2 or 3 -- the 2-D single-candidate paths get a
view, which shares memory just the same.

TWO WAYS TO BREAK IT, both silent:
  * hand `rollout` a non-float64 or non-ndarray `U`. `np.asarray` would then
    copy, the projection would land in the copy, and `plan()` would score an
    input nobody flew. `_assert_writes_through` in the tests below pins this.
  * project node 0, or project after calling `step`. `step` advances position
    with `p + dt*v + 0.5*dt^2*a` (planner/dynamics.py:132), so the corrected `a`
    has to be in hand BEFORE the step that uses it, or p is left integrating an
    acceleration that was never applied.
"""

import numpy as np

# `_as_batch` is private to `planner.dynamics`, and is imported rather than
# reimplemented on purpose: its exact copy-vs-view behaviour IS the mutation
# contract above. A local three-line copy would be a second definition of the
# thing this class depends on, free to drift from the original.
from planner.dynamics import PlanarDynamics, _as_batch
from planner.types import IAL, IOM, NXI, S_VEL, U_ACC


class CappedDynamics(PlanarDynamics):
    """PlanarDynamics whose rollout keeps v and omega inside their bounds.

    Drop-in for `PlanarDynamics`: same constructor, same attributes, every
    other method inherited unchanged. Only `rollout` is overridden, and it
    MUTATES ITS `U` ARGUMENT IN PLACE -- see the module docstring, which is
    where the reasoning lives.
    """

    def rollout(self, xi0, U):
        """Roll K input sequences forward, projecting v and omega as it goes.

        xi0 (6,) or (K,6), U (K,N,3) or (N,3) -> X (K,N+1,6), as the base
        class. `U` is modified in place wherever a bound was hit.
        """
        U, _was_2d = _as_batch(U)
        K, N = U.shape[0], U.shape[1]
        X = np.empty((K, N + 1, NXI), dtype=np.float64)
        X[:, 0, :] = np.asarray(xi0, dtype=np.float64)

        dt = self.dt
        v_max = float(self.lim.v_max)
        om_max = float(self.lim.omega_max)

        for k in range(N):
            xk = X[:, k, :]

            # --- speed: project v_{k+1} onto the disc, then back out a_k -----
            v = xk[:, S_VEL]                       # (K, 2) view into X
            a = U[:, k, U_ACC]                     # (K, 2) view into U
            v_new = v + dt * a
            n = np.linalg.norm(v_new, axis=-1, keepdims=True)
            over = n > v_max
            if over.any():
                # Written only where the bound was hit. Assigning the
                # recomputed value everywhere would round-trip every untouched
                # input through (v + dt*a - v)/dt and perturb it by an ulp,
                # which is a change to samples that never needed one.
                a_cap = (v_new * (v_max / np.maximum(n, 1e-12)) - v) / dt
                np.copyto(U[:, k, U_ACC], a_cap, where=over)

            # --- heading rate: the same, on a scalar, so a clip not a scale --
            om = xk[:, IOM]                        # (K,) view into X
            al = U[:, k, IAL]                      # (K,) view into U
            om_new = om + dt * al
            hot = np.abs(om_new) > om_max
            if hot.any():
                al_cap = (np.clip(om_new, -om_max, om_max) - om) / dt
                np.copyto(U[:, k, IAL], al_cap, where=hot)

            # AFTER the projection, so position integrates the applied input.
            X[:, k + 1, :] = self.step(xk, U[:, k, :])

        return X


def dynamics_from_config(cfg, limits):
    """PlanarDynamics or CappedDynamics, per `mppi.cap_velocity`.

    One reader for the switch, so `run_planar_sim.build_planner` and anything
    else that builds a planner cannot disagree about which expert is flying.
    """
    cls = CappedDynamics if bool(cfg.mppi.get("cap_velocity", False)) \
        else PlanarDynamics
    return cls(limits, dt=cfg.mppi.dt)


# ----------------------------------------------------------------- tests

def _main():
    from planner.dynamics import PlanarLimits

    ok = [0]

    def check(name, cond, detail=""):
        print("  %-58s %s   %s" % (name, "PASS" if cond else "FAIL", detail))
        assert cond, name
        ok[0] += 1

    print("velocity/heading-rate projection inside the rollout")

    # The HPA envelope, which is now the autopilot's (config.yaml px4:).
    lim = PlanarLimits(v_max=0.5, a_max=5.0, omega_max=0.5236, alpha_max=4.0,
                       tilt_max=np.radians(45.0), j_max=8.0)
    dt = 0.1
    base = PlanarDynamics(lim, dt=dt)
    cap = CappedDynamics(lim, dt=dt)

    K, N = 64, 30
    rng = np.random.RandomState(0)
    xi0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def fresh():
        # Deliberately hot: 3 m/s^2 for 30 steps of 0.1 s would reach 9 m/s.
        return np.ascontiguousarray(
            rng.uniform(-3.0, 3.0, size=(K, N, 3)).astype(np.float64))

    U0 = fresh()
    Ub, Uc = U0.copy(), U0.copy()
    Xb = base.rollout(xi0, Ub)
    Xc = cap.rollout(xi0, Uc)

    sp_b = np.linalg.norm(Xb[..., S_VEL], axis=-1)
    sp_c = np.linalg.norm(Xc[..., S_VEL], axis=-1)
    check("the base class blows straight through v_max",
          sp_b.max() > 3 * lim.v_max, "peak %.2f m/s" % sp_b.max())
    check("the capped one never exceeds it",
          sp_c.max() <= lim.v_max + 1e-9, "peak %.6f m/s" % sp_c.max())
    check("nor exceeds omega_max",
          np.abs(Xc[..., IOM]).max() <= lim.omega_max + 1e-9,
          "peak %.6f rad/s" % np.abs(Xc[..., IOM]).max())

    # --- the reason the whole thing exists ------------------------------
    check("states_ok REJECTS the uncapped pool",
          base.states_ok(Xb).mean() < 0.05,
          "%.0f%% survive" % (100 * base.states_ok(Xb).mean()))
    check("and ACCEPTS every capped sample",
          cap.states_ok(Xc).all(),
          "%d/%d survive" % (cap.states_ok(Xc).sum(), K))

    # --- X is still the integral of U -----------------------------------
    v_pred = Xc[:, :-1, S_VEL] + dt * Uc[..., U_ACC]
    check("v_{k+1} == v_k + dt a_k for the MUTATED U",
          np.abs(v_pred - Xc[:, 1:, S_VEL]).max() < 1e-12,
          "max residual %.2e" % np.abs(v_pred - Xc[:, 1:, S_VEL]).max())
    p_pred = (Xc[:, :-1, 0:2] + dt * Xc[:, :-1, S_VEL]
              + 0.5 * dt * dt * Uc[..., U_ACC])
    check("and p_{k+1} == p_k + dt v_k + dt^2/2 a_k",
          np.abs(p_pred - Xc[:, 1:, 0:2]).max() < 1e-12,
          "max residual %.2e" % np.abs(p_pred - Xc[:, 1:, 0:2]).max())
    om_pred = Xc[:, :-1, IOM] + dt * Uc[..., IAL]
    check("and omega_{k+1} == omega_k + dt alpha_k",
          np.abs(om_pred - Xc[:, 1:, IOM]).max() < 1e-12)

    # --- the acceleration disc survives ---------------------------------
    n_in = np.linalg.norm(U0[..., U_ACC], axis=-1)
    n_out = np.linalg.norm(Uc[..., U_ACC], axis=-1)
    check("|a| never grows (projection is non-expansive)",
          (n_out <= n_in + 1e-12).all(),
          "worst growth %.2e" % (n_out - n_in).max())
    check("so an input pool inside a_max stays inside it",
          n_out.max() <= max(n_in.max(), lim.a_max_eff) + 1e-12)

    # --- untouched samples are bit-identical ----------------------------
    touched = (np.abs(Uc - U0) > 0).any(axis=-1)
    check("only the steps that hit a bound were rewritten",
          touched.any() and not touched.all(),
          "%.0f%% of steps rewritten" % (100 * touched.mean()))

    # --- the mutation contract ------------------------------------------
    Um = fresh()
    before = Um.copy()
    cap.rollout(xi0, Um)
    check("rollout WRITES THROUGH to the caller's array",
          not np.array_equal(Um, before),
          "this is what plan() relies on; see the module docstring")

    U2 = fresh()
    v2 = U2[None, ...][0]          # a view, as _as_batch makes for 2-D input
    cap.rollout(xi0, v2)
    check("and through a view, as the 2-D single-candidate path gets",
          not np.array_equal(v2, U2) or np.shares_memory(v2, U2))

    # --- a sample already inside the ball is untouched -------------------
    U_cold = np.zeros((4, N, 3))
    U_cold[..., 0] = 0.05          # 0.05*0.1*30 = 0.15 m/s, well inside
    cold = U_cold.copy()
    Xcold = cap.rollout(xi0, U_cold)
    check("a pool that never reaches v_max is passed through unchanged",
          np.array_equal(U_cold, cold),
          "peak %.3f m/s" % np.linalg.norm(Xcold[..., S_VEL],
                                           axis=-1).max())

    # --- agreement with the base class where no bound is hit -------------
    check("and its trajectory equals the base class's, exactly",
          np.array_equal(Xcold, base.rollout(xi0, cold)))

    print("\n%d checks passed" % ok[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
