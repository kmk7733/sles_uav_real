"""The exact call-count commitment from hpa.rollout._commit_plan.

Preserved boundary: commit=10 emits a fresh answer, cached indices 1..10
(index 10 repeats node 9), then refreshes on the next call. During a supervisor
bridge the producer is not called, so this counter does not advance. Neither
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
        if st['res'] is None or st['i'] >= commit:
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
