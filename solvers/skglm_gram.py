from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.penalties import L1
    from skglm.solvers.gram_cd import gram_cd_solver
    from skglm.utils import compiled_clone


class Solver(BaseSolver):
    name = "skglm-gram"

    parameters = {
        'use_acc': [True, False],
        'greedy_cd': [True, False]
    }

    def __init__(self, use_acc, greedy_cd):
        self.use_acc = use_acc
        self.greedy_cd = greedy_cd

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_samples = self.X.shape[0]
        self.l1_penalty = compiled_clone(L1(lmbd / n_samples))

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        self.coef = gram_cd_solver(self.X, self.y, self.l1_penalty,
                                   tol=1e-12, verbose=0, max_iter=n_iter,
                                   use_acc=self.use_acc,
                                   greedy_cd=self.greedy_cd)[0]

    def get_result(self):
        return self.coef
