from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from skglm.gpu.solvers.pytorch_solver import (PytorchSolver, L1Pytorch,
                                                  QuadraticPytorch)


class Solver(BaseSolver):
    name = 'FyTorch'
    stopping_strategy = 'iteration'

    parameters = {
        "use_auto_diff": [True, False]
    }

    def __init__(self, use_auto_diff):
        self.use_auto_diff = use_auto_diff

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.solver = PytorchSolver(use_auto_diff=self.use_auto_diff)
        self.datafit = QuadraticPytorch()
        self.penalty = L1Pytorch(lmbd / len(y))

        # cache potential compilation
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
