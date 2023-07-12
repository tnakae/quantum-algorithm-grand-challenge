from typing import Any

import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.algo.optimizer import (Optimizer, OptimizerState,
                                       OptimizerStatus)
from quri_parts.core.estimator import ConcurrentParametricQuantumEstimator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.operator import Operator
from quri_parts.core.sampling.shots_allocator import \
    create_equipartition_shots_allocator
from quri_parts.core.state import (ComputationalBasisState,
                                   ParametricCircuitQuantumState)
from quri_parts.openfermion.operator import operator_from_openfermion_op

from sbovqaopt import FloatArray, SBOOptimizer
from utils.challenge_2023 import (ChallengeSampling,
                                  QuantumCircuitTimeExceededError)

challenge_sampling = ChallengeSampling(noise=True)


def cost_fn(
    hamiltonian: Operator,
    parametric_state: ParametricCircuitQuantumState,
    param_values: FloatArray,
    estimator: ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState],
) -> float:
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return list(estimate)[0].value.real


def vqe(
    hamiltonian: Operator,
    parametric_state: ParametricCircuitQuantumState,
    estimator: ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState],
    init_params: FloatArray,
    optimizer: Optimizer,
) -> OptimizerState:
    opt_state: OptimizerState = optimizer.get_init_state(init_params)

    def c_fn(param_values: FloatArray) -> float:
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values: FloatArray) -> FloatArray:
        return np.zeros_like(param_values)

    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            print(f"iteration {opt_state.niter}")
            print(opt_state.cost)
        except QuantumCircuitTimeExceededError:
            print("Reached the limit of shots")
            return opt_state

        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        # make hf + HEreal ansatz
        hf_gates = ComputationalBasisState(n_qubits, bits=0b00001111).circuit.gates
        hw_ansatz = HardwareEfficientReal(qubit_count=n_qubits, reps=1)
        hw_hf = hw_ansatz.combine(hf_gates)

        parametric_state = ParametricCircuitQuantumState(n_qubits, hw_hf)

        hardware_type = "sc"
        shots_allocator = create_equipartition_shots_allocator()
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = 10**4

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, measurement_factory, shots_allocator, hardware_type
            )
        )

        optimizer = SBOOptimizer()
        init_param: FloatArray = (
            np.random.rand(hw_ansatz.parameter_count) * 2 * np.pi * 0.001
        )

        result = vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            optimizer,
        )
        print(f"iteration used: {result.niter}")
        return result.cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
