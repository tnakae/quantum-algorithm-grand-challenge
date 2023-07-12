"""
Defines the SBO Optimizer class.
"""
from typing import Any, Callable, Optional

import numpy as np
from quri_parts.algo.optimizer import (Optimizer, OptimizerState,
                                       OptimizerStatus)

from .optimizer_iteration import SBOOptimizerIteration
from .typing import FloatArray


class SBOOptimizerState(OptimizerState):
    local_minima_found: FloatArray

    def __init__(
        self,
        local_minima_found: FloatArray = np.asarray([], dtype=np.float_),
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.local_minima_found = local_minima_found

    def get_current_x(self) -> FloatArray:
        return self.params


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
    ) -> None:
        super().__init__()

        # general optimizer arguments
        self.maxiter = maxiter
        self.patch_size = patch_size
        self.npoints_per_patch = npoints_per_patch
        self.epsilon_i = epsilon_i
        self.epsilon_int = epsilon_int
        self.epsilon_f = epsilon_f

    def get_init_state(self, x0: FloatArray) -> OptimizerState:
        return SBOOptimizerState(x0)

    def step(
        self,
        state: OptimizerState,
        cost_function: Callable[[FloatArray], float],
        grad_function: Optional[Callable[[FloatArray], FloatArray]] = None,
    ) -> OptimizerState:
        """Minimize the scalar function"""
        assert isinstance(state, SBOOptimizerState)
        optimizer_iteration = SBOOptimizerIteration()

        current_x = state.get_current_x()
        niter = state.niter
        local_minima_found = state.local_minima_found
        optimize_bounds_size = (
            self.patch_size * (1.0 - self.epsilon_i) * (1.0 - niter / self.maxiter)
        )
        res = optimizer_iteration.minimize_kde(
            cost_function,
            current_x,
            self.patch_size,
            optimize_bounds_size,
            self.npoints_per_patch,
        )
        new_x: FloatArray = res.x
        cost: float = res.fun
        distance = np.linalg.norm(new_x - current_x, ord=np.inf)
        current_x = new_x
        if distance < (self.patch_size / 2) * (1 - self.epsilon_int):
            # local minimum found within this patch area
            if local_minima_found.shape[0] == 0:
                local_minima_found = np.asarray([current_x])
            else:
                local_minima_found = np.vstack([local_minima_found, current_x])

        niter += 1

        result_state: SBOOptimizerState

        if niter < self.maxiter:
            result_state = SBOOptimizerState(
                params=current_x,
                niter=niter,
                local_minima_found=local_minima_found,
                cost=cost,
                status=OptimizerStatus.SUCCESS,
            )
        else:
            # use all nearby local minima to calculate the optimal x
            local_minima_near_current_x = [
                local_minimum
                for local_minimum in local_minima_found
                if (
                    np.linalg.norm(local_minimum - current_x, ord=np.inf)
                    < (self.patch_size / 2) * self.epsilon_f
                )
            ]

            optimal_x: FloatArray = (
                np.mean(local_minima_near_current_x, axis=0)
                if local_minima_near_current_x
                else current_x
            )
            optimal_cost = cost_function(optimal_x)

            result_state = SBOOptimizerState(
                params=optimal_x,
                niter=niter,
                local_minima_found=local_minima_found,
                cost=optimal_cost,
                status=OptimizerStatus.CONVERGED,
            )

        return result_state
