'''
Defines the Optimizer class.
'''
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
from quri_parts.algo.optimizer import (
    Optimizer,
    OptimizerState,
    OptimizerStatus,
)

from .optimizer_iteration import SBOOptimizerIteration

POINT = npt.NDArray[np.float_]


class SBOOptimizerState(OptimizerState):
    step: int = 0
    local_minima_found: POINT = np.asarray([])

    def __init__(self, x0: POINT):
        super().__init__(x0)


class SBOOptimizer(Optimizer):
    """
    Implements surrogate-based optimization using a Gaussian kernel.

    Parameters (notation from paper)

    maxiter: number of optimization iterations (M)
    patch_size: length of sampling hypercube sides (ℓ)
    npoints_per_patch: sample points per iteration (𝜏)
    epsilon_i: initial fraction of patch to exclude for optimization
               region (ε_i)
    epsilon_int: fraction of patch to exclude for edge effects on each
                 iteration (ε_int)
    epsilon_f: final fraction of patch to include when performing final
               averaging (ε_f)
    nfev_final_avg: number of function evaluations to perform to calculate
                    final function value (if nfev_final_avg == 0, then
                    no final function value will be calculated)
    """
    def __init__(
        self,
        maxiter: int = 100,
        patch_size: float = 0.1,
        npoints_per_patch: int = 20,
        epsilon_i: float = 0.0,
        epsilon_int: float = 0.05,
        epsilon_f: float = 0.5,
        nfev_final_avg: int = 0,
    ) -> None:
        super().__init__()

        # general optimizer arguments
        self.maxiter = maxiter
        self.patch_size = patch_size
        self.npoints_per_patch = npoints_per_patch
        self.epsilon_i = epsilon_i
        self.epsilon_int = epsilon_int
        self.epsilon_f = epsilon_f
        self.nfev_final_avg = nfev_final_avg

    def get_init_state(self, x0: POINT) -> OptimizerState:
        return SBOOptimizerState(x0)

    def step(
        self,
        state: OptimizerState,
        cost_function: Callable[[POINT], float],
        grad_functino: Optional[Callable[[POINT], POINT]] = None,
    ) -> OptimizerState:
        """Minimize the scalar function"""
        assert isinstance(state, SBOOptimizerState)
        optimizer_iteration = SBOOptimizerIteration()

        current_x = state.params
        local_minima_found = state.local_minima_found
        optimize_bounds_size = (
            self.patch_size
            * (1.0 - self.epsilon_i)
            * (1.0 - state.step / self.maxiter)
        )
        res = optimizer_iteration.minimize_kde(
            cost_function,
            current_x,
            self.patch_size,
            optimize_bounds_size,
            self.npoints_per_patch,
        )
        new_x = res.x
        distance = np.linalg.norm(new_x - current_x, ord=np.inf)
        current_x = new_x
        if distance < (self.patch_size / 2) * (1 - self.epsilon_int):
            # local minimum found within this patch area
            if local_minima_found.shape[0] == 0:
                local_minima_found = np.asarray([current_x])
            else:
                local_minima_found = np.vstack([local_minima_found, current_x])

        # use all nearby local minima to calculate the optimal x
        local_minima_near_current_x = [
            local_minimum
            for local_minimum in local_minima_found
            if (
                np.linalg.norm(local_minimum - current_x, ord=np.inf)
                < (self.patch_size / 2) * self.epsilon_f
            )
        ]
        optimal_x = (
            np.mean(local_minima_near_current_x, axis=0)
            if local_minima_near_current_x
            else current_x
        )

        result = qiskitopt.OptimizerResult()
        result.nfev = (
            (self.maxiter * self.npoints_per_patch)
            + self.nfev_final_avg
        )
        result.nit = self.maxiter

        result.x = optimal_x
        if self.nfev_final_avg > 0:
            result.fun = np.mean(
                [fun(optimal_x) for _ in range(self.nfev_final_avg)]
            )
        else:
            result.fun = (
                'final function value not evaluated '
                + 'because nfev_final_avg == 0'
            )

        return result
