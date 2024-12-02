"""Benchmarking module for ZNE and unoptimized circuits."""

from typing import Callable
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from mitiq import zne

from unopt.noise import depolarizing_noise_model
from unopt.recipe import elementary_recipe
from unopt.qem import execute_no_shot_noise, execute


@dataclass
class BenchResults:
    avg_ideal_value: float
    avg_unmit_value: float
    avg_unmit_error: float
    avg_zne_fold_value: float
    avg_zne_fold_error: float
    avg_zne_unopt_value: float
    avg_zne_unopt_error: float
    percent_improvement_unmit: float
    percent_improvement_zne_fold: float
    original_circuit_depth: int
    avg_zne_fold_circuit_depths: list[int]
    avg_zne_unopt_circuit_depths: list[int]


def bench(
    qc: QuantumCircuit,
    backend: Backend = AerSimulator(),
    noise_model: NoiseModel = depolarizing_noise_model(error=0.01),
    shots: int = 10_000,
    scale_factors_zne: list[float] = [1, 3, 5],
    iterations_unopt: list[int] = [1, 2, 3],
    fold_method: Callable = zne.scaling.fold_global,
    extrapolation_method: Callable = zne.RichardsonFactory,
    trials: int = 1,
    verbose: bool = False,
) -> BenchResults:
    """Calculate ideal, unmitigated, ZNE-fold, and ZNE-unoppt values/data."""
    # Initialize accumulators for results across trials
    ideal_values = []
    unmit_values = []
    zne_fold_values = []
    zne_unopt_values = []
    folded_depths_list = []
    unopt_depths_list = []

    original_depth = qc.depth()

    for trial in range(trials):
        if verbose:
            print(f"Trial {trial + 1}/{trials}")

        # Ideal (noiseless) expectation value:
        ideal_value = np.around(execute_no_shot_noise(qc), decimals=1)
        ideal_values.append(ideal_value)

        # Unmitigated expectation value:
        unmit_value = execute(circuit=qc, backend=backend, shots=shots, noise_model=noise_model)
        unmit_values.append(unmit_value)

        # ZNE + Fold:
        folded_circuits = [fold_method(qc, s) for s in scale_factors_zne]
        folded_values = [
            execute(circuit=circ, backend=backend, shots=shots, noise_model=noise_model) for circ in folded_circuits
        ]
        folded_depths = [circ.depth() for circ in folded_circuits]
        folded_depths_list.append(folded_depths)

        factory = extrapolation_method(scale_factors_zne)
        [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_zne, folded_values)]
        zne_fold_values.append(factory.reduce())

        # ZNE + Unopt:
        unoptimized_circuits = [elementary_recipe(qc, iterations=i) for i in iterations_unopt]
        unoptimized_values = [
            execute(circuit=c, backend=backend, shots=shots, noise_model=noise_model) for c in unoptimized_circuits
        ]
        unoptimized_depths = [circ.depth() for circ in unoptimized_circuits]
        unopt_depths_list.append(unoptimized_depths)

        scale_factors_unopt = [depth / original_depth for depth in unoptimized_depths]
        factory = extrapolation_method(scale_factors_unopt)
        [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_unopt, unoptimized_values)]
        zne_unopt_values.append(factory.reduce())

    # Average results across trials
    avg_ideal_value = np.mean(ideal_values)
    avg_unmit_value = np.mean(unmit_values)
    avg_zne_fold_value = np.mean(zne_fold_values)
    avg_zne_unopt_value = np.mean(zne_unopt_values)

    # Calculate errors and improvements
    avg_unmit_error = abs(avg_ideal_value - avg_unmit_value)
    avg_zne_fold_error = abs(avg_ideal_value - avg_zne_fold_value)
    avg_zne_unopt_error = abs(avg_ideal_value - avg_zne_unopt_value)

    percent_improvement_unmit = ((avg_unmit_error - avg_zne_unopt_error) / avg_zne_unopt_error) * 100
    percent_improvement_zne_fold = ((avg_zne_fold_error - avg_zne_unopt_error) / avg_zne_unopt_error) * 100

    if verbose:
        print(f"Average ideal value: {avg_ideal_value}")
        print(f"Average unmitigated expectation value: {avg_unmit_value}")
        print(f"Average ZNE + fold value: {avg_zne_fold_value}")
        print(f"Average ZNE + unopt value: {avg_zne_unopt_value}")
        print(f"ZNE/unopt improvement over unmitigated: {percent_improvement_unmit:.2f}%")
        print(f"ZNE/unopt improvement over ZNE/fold: {percent_improvement_zne_fold:.2f}%")

    return BenchResults(
        avg_ideal_value=avg_ideal_value,
        avg_unmit_value=avg_unmit_value,
        avg_unmit_error=avg_unmit_error,
        avg_zne_fold_value=avg_zne_fold_value,
        avg_zne_fold_error=avg_zne_fold_error,
        avg_zne_unopt_value=avg_zne_unopt_value,
        avg_zne_unopt_error=avg_zne_unopt_error,
        percent_improvement_unmit=percent_improvement_unmit,
        percent_improvement_zne_fold=percent_improvement_zne_fold,
        original_circuit_depth=original_depth,
        avg_zne_fold_circuit_depths=np.mean(folded_depths_list, axis=0).tolist(),
        avg_zne_unopt_circuit_depths=np.mean(unopt_depths_list, axis=0).tolist(),
    )
