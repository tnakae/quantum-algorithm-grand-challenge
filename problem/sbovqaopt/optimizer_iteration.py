"""
Defines the OptimizerIteration class.
"""
from typing import Callable, Tuple

import numpy as np
import scipy.optimize

from .typing import FloatArray


def scott_bandwidth(n: int, d: int) -> float:
    """
    Scott's Rule per D.W. Scott,
    "Multivariate Density Estimation: Theory, Practice, and Visualization",
    John Wiley & Sons, New York, Chicester, 1992
    """
    res: float = n ** (-1.0 / (d + 4))
    return res


class SBOOptimizerIteration:
    """
    Implements a single iteration of surrogate-based optimization using
    a Gaussian kernel.
    """

    def __init__(self) -> None:
        pass

    def get_conditional_expectation_with_gradient(
        self,
        training_data: FloatArray,
        x: FloatArray,
        bandwidth_function: Callable[..., float] = scott_bandwidth,
    ) -> Tuple[float, FloatArray]:
        # normalize training data coordinates
        training_x = training_data[:, :-1]
        training_x_mean: FloatArray = np.mean(training_x, axis=0)
        training_x_std: FloatArray = np.std(training_x, axis=0)
        training_x_std[training_x_std == 0.0] = 1.0
        training_x_normalized: FloatArray = (
            training_x - training_x_mean
        ) / training_x_std

        # normalize input coordinates
        x_normalized: FloatArray = (x - training_x_mean) / training_x_std

        # normalize training data z-values
        training_z: FloatArray = training_data[:, -1]
        training_z_mean = float(np.mean(training_z))
        training_z_std = float(np.std(training_z) or 1.0)
        training_z_normalized: FloatArray = (
            training_z - training_z_mean
        ) / training_z_std

        # get the normalized conditional expectation in z
        bandwidth = bandwidth_function(*training_x.shape)
        gaussians: FloatArray = np.exp(
            -1
            / (2 * bandwidth**2)
            * np.linalg.norm((training_x_normalized - x_normalized), axis=1) ** 2
        )
        exp_z_normalized: float = np.sum(training_z_normalized * gaussians) / np.sum(
            gaussians
        )

        # calculate the gradients along each x coordinate
        grad_gaussians: FloatArray = np.array(
            [
                (1 / (bandwidth**2))
                * (training_x_normalized[:, i] - x_normalized[i])
                * gaussians
                for i in range(len(x_normalized))
            ]
        )
        grad_exp_z_normalized: FloatArray = np.array(
            [
                (
                    np.sum(gaussians)
                    * np.sum(training_z_normalized * grad_gaussians[i])
                    - np.sum(training_z_normalized * gaussians)
                    * np.sum(grad_gaussians[i])
                )
                / (np.sum(gaussians) ** 2)
                for i in range(len(grad_gaussians))
            ]
        )

        # undo the normalization and return the expectation value and gradients
        exp_z = training_z_mean + training_z_std * exp_z_normalized
        grad_exp_z = training_z_std * grad_exp_z_normalized

        return exp_z, grad_exp_z

    def minimize_kde(
        self,
        f: Callable[[FloatArray], float],
        patch_center_x: FloatArray,
        patch_size: float,
        optimize_bounds_size: float,
        npoints_per_patch: int,
    ) -> scipy.optimize.OptimizeResult:
        training_point_angles = self._generate_x_coords(
            patch_center_x, patch_size, npoints_per_patch
        )
        measured_values: FloatArray = np.atleast_2d(
            [f(x) for x in training_point_angles]
        ).T

        return self._minimize_kde(
            training_point_angles,
            measured_values,
            patch_center_x,
            optimize_bounds_size,
        )

    def _minimize_kde(
        self,
        angles: FloatArray,
        values: FloatArray,
        patch_center_x: FloatArray,
        optimize_bounds_size: float,
    ) -> scipy.optimize.OptimizeResult:
        training: FloatArray = np.concatenate((angles, values), axis=1)
        num_angles = len(patch_center_x)
        bounds_limits = np.array(
            [
                [
                    patch_center_x[angle] - optimize_bounds_size / 2,
                    patch_center_x[angle] + optimize_bounds_size / 2,
                ]
                for angle in range(num_angles)
            ]
        )
        bounds = scipy.optimize.Bounds(
            lb=bounds_limits[:, 0], ub=bounds_limits[:, 1], keep_feasible=True
        )
        return scipy.optimize.minimize(
            fun=lambda x: self.get_conditional_expectation_with_gradient(training, x),
            jac=True,
            x0=patch_center_x,
            bounds=bounds,
            method="L-BFGS-B",
        )

    def _generate_x_coords(
        self,
        center: FloatArray,
        patch_size: float,
        num_points: int = 40,
    ) -> FloatArray:
        """
        Generate num_points sample coordinates using Latin hypercube sampling.
        _lhsclassic copied from:
        https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py#L123-L141
        """

        def _lhsclassic(n: int, samples: int) -> FloatArray:
            # Generate the intervals
            cut = np.linspace(0, 1, samples + 1)

            # Fill points uniformly in each interval
            u = np.random.rand(samples, n)
            a = cut[:samples]
            b = cut[1:samples + 1]
            rdpoints = np.zeros_like(u)
            for j in range(n):
                rdpoints[:, j] = u[:, j] * (b - a) + a

            # Make the random pairings
            H = np.zeros_like(rdpoints)
            for j in range(n):
                order = np.random.permutation(range(samples))
                H[:, j] = rdpoints[order, j]

            return H

        n_dim = len(center)
        lhs_points = _lhsclassic(n_dim, num_points)
        return np.array(
            [((point - 0.5) * 2 * patch_size) + center for point in lhs_points]
        )
