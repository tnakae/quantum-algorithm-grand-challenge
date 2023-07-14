import numpy as np

from sbovqaopt.typing import FloatArray
from sbovqaopt.optimizer_iteration import SBOOptimizerIteration


CENTER: FloatArray = np.asarray([
    0.5,
    1.0,
    -0.02,
    0.3,
    0.5,
    1.0,
    -0.02,
    0.3,
], dtype=np.float_)
RNS = np.random.RandomState(seed=123)


def test_func(params: FloatArray) -> float:
    res: float = np.mean((params - CENTER) ** 2) + RNS.standard_normal() * 0.01
    return res


def test_optimizer_iteration() -> None:
    test_optimizer_iteration = SBOOptimizerIteration()
    patch_size = 0.1
    maxiter = 64
    epsilon_i = 0.1
    x0: FloatArray = np.zeros_like(CENTER)

    for niter in np.arange(maxiter):
        optimize_bounds_size = (
            patch_size * (1.0 - epsilon_i) * (1.0 - niter / maxiter)
        )
        res = test_optimizer_iteration.minimize_kde(test_func, x0,
                                                    patch_size, optimize_bounds_size,
                                                    16)
        x0 = res.x
        print(x0)
        print(res.fun)


if __name__ == "__main__":
    test_optimizer_iteration()
