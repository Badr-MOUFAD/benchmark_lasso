from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.skglm_jax.datafits import QuadraticJax
    from skglm.skglm_jax.penalties import L1Jax
    from skglm.skglm_jax.fista import Fista


class Solver(BaseSolver):
    name = "fista-jax"
    stopping_strategy = "iteration"

    parameters = {
        "use_auto_diff": [True, False]
    }

    def __init__(self, use_auto_diff):
        self.use_auto_diff = use_auto_diff

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_samples = self.X.shape[0]
        self.datafit = QuadraticJax()
        self.penalty = L1Jax(self.lmbd / n_samples)
        self.solver = Fista(max_iter=1, use_auto_diff=self.use_auto_diff)

        # Cache Numba compilation
        self.run(2)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
        else:
            self.solver.max_iter = n_iter
            coef = self.solver.solve(
                self.X, self.y, self.datafit, self.penalty)

            if self.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_]

            self.coef = coef

    @staticmethod
    def get_next(stop_val):
        return stop_val + 30

    def get_result(self):
        return self.coef
