import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.gpu.solvers import JaxSolver


class Solver(BaseSolver):
    name = 'Jax'
    stopping_strategy = 'iteration'

    parameters = {
        "use_auto_diff": [True, False]
    }

    def __init__(self, use_auto_diff):
        self.use_auto_diff = use_auto_diff

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.solver = JaxSolver(use_auto_diff=self.use_auto_diff)

        # cache jxa grad autodiff compilation
        self.run(2)

    def run(self, n_iter):
        self.solver.max_iter = n_iter
        self.coef = self.solver.solve(self.X, self.y, self.lmbd)

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 50

    def get_result(self):
        return self.coef
