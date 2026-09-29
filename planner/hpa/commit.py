"""The call-count commitment from hpa.rollout._commit_plan (same fix, same semantics).

commit=c: a fresh answer, then cached indices 1..c-1, then a fresh answer on
the next call -- a period of c calls. With commit = chunk length (10) every
action of the chunk is used exactly once and the (c+1)-th call starts the next
chunk at its node 0. (Until 2026-09-28 the condition was `i >= commit`: a
period of c+1 calls whose last call repeated node 9 through `min(i, 9)`.)

The counter is the index of the committed plan that is due NOW. A supervisor
bridge does not call the producer but still consumes one tick of the committed
plan per tick (DeSimplexSupervisor._advance_bridge advances `_commit_state`),
so after a bridge the plan resumes where the bridge landed on it. Neither
wall-clock scheduling nor sensor freshness belongs in this wrapper.
"""
import numpy as np

from planner.types import MPPIResult


def commit_plan(base_plan, commit):
    """Wrap an unbound producer method; retain existing simulator semantics."""
    if isinstance(commit, bool) or int(commit) != commit or int(commit) < 1:
        raise ValueError("commit must be a positive integer")
    commit = int(commit)

    def plan(self, state, goal=None, a_prev=None, **kw):
        st = getattr(self, '_commit_state', None)
        if st is None:
            st = self._commit_state = {'i': commit, 'res': None}
        if st['res'] is None or st['i'] + 1 >= commit:
            res = base_plan(self, state, goal=goal, a_prev=a_prev, **kw)
            st['res'], st['i'] = res, 0
            return res
        st['i'] += 1
        base = st['res']
        if base.reference is None:
            st['i'] = commit
            return base
        U = np.asarray(base.U)
        X = np.asarray(base.X)
        j = min(st['i'], U.shape[0] - 1)
        Ui = np.concatenate([U[j:], np.zeros((j, U.shape[1]))], axis=0)
        Xi = np.concatenate([X[j:], np.repeat(X[-1:], j, axis=0)], axis=0)
        return MPPIResult(base.status, base.reference.shifted(j), Ui, Xi,
                          base.cost, base.n_valid, base.n_samples, base.beta,
                          'learned HPA (committed %d/%d)' % (st['i'], commit))
    return plan


def ticks_until_fresh(commit, commit_state, bridge_ticks=0):
    """Ticks from now until the tick whose producer call starts a new chunk (1 = the next tick).

    Mirrors `commit_plan` above: a fresh answer when there is no cached result
    or when i + 1 >= commit. `bridge_ticks` supervisor bridge ticks come first;
    each consumes one tick of the committed plan without calling the producer.
    Used on the vehicle to call the policy early enough that the chunk is ready
    when that tick comes; it changes nothing about which tick is fresh.
    """
    if commit_state is None or commit_state.get("res") is None:
        return 1 + int(bridge_ticks)
    i, t = int(commit_state["i"]), 0
    for _ in range(int(bridge_ticks)):
        t += 1
        i += 1
    while True:
        t += 1
        if i + 1 >= commit:
            return t
        i += 1
