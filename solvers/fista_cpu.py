from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from skglm.gpu.solvers import CPUSolver
    from skglm.gpu.solvers.base import BaseL1, BaseQuadratic


class Solver(BaseSolver):
    name = 'CPU'
    stopping_strategy = 'iteration'

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.solver = CPUSolver()
        self.datafit = BaseQuadratic()
        self.penalty = BaseL1(lmbd / len(y))

        # cache potential compilation
        self.run(2)

    def run(self, n_iter):
        self.solver.max_iter = n_iter
        self.coef = self.solver.solve(self.X, self.y, self.datafit, self.penalty)

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 50

    def get_result(self):
        return self.coef
