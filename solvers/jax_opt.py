from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import numpy as np
    import jax.numpy as jnp

    from jaxopt import prox
    from jaxopt import objective
    from jaxopt import BlockCoordinateDescent
    from jaxopt import AndersonWrapper

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


class Solver(BaseSolver):
    name = "jax-opt-cd-aa"
    stopping_strategy = "iteration"

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.data = (X, y)
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

        self.n_samples, self.n_features = X.shape

        self.w_init = jnp.zeros(X.shape[1])
        datafit = objective.least_squares

        self.solver = AndersonWrapper(
            solver=BlockCoordinateDescent(
                datafit,
                block_prox=prox.prox_lasso,
                maxiter=1,
                tol=1e-9,
                jit=True
            ),
            ridge=1e-4,
            history_size=5
        )

        # cache jit compilation
        self.run(2)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.n_features + self.fit_intercept])
        else:
            self.solver.maxiter = n_iter
            coef, _ = self.solver.run(
                self.w_init,
                hyperparams_prox=self.lmbd / self.n_samples,
                data=self.data
            )

            if self.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_]

            self.coef = np.asarray(coef)

    @staticmethod
    def get_next(stop_val):
        return stop_val + 100

    def get_result(self):
        return self.coef
