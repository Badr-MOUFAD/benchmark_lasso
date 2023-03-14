import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from skglm.gpu.solvers import NumbaSolver
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'Numba'
    stopping_strategy = 'iteration'

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.solver = NumbaSolver()

        # cache numba compilation
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
