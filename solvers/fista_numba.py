import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from skglm.gpu.solvers import NumbaSolver
    from skglm.gpu.solvers.numba_solver import QuadraticNumba, L1Numba

    from numba.core.errors import NumbaPerformanceWarning


class Solver(BaseSolver):
    name = 'Numba'
    stopping_strategy = 'iteration'

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
        self.solver = NumbaSolver()
        self.datafit = QuadraticNumba()
        self.penalty = L1Numba(lmbd / len(y))

        # cache numba compilation
        self.run(5)

    def run(self, n_iter):
        self.solver.max_iter = n_iter
        self.coef = self.solver.solve(self.X, self.y, self.datafit, self.penalty)

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 50

    def get_result(self):
        return self.coef
