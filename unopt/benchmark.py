"""Benchmarking module for ZNE and unoptimized circuits."""

from typing import Callable
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from mitiq import zne

from unopt.noise_models import depolarizing_noise_model
from unopt.recipe import elementary_recipe
from unopt.qem import execute_no_shot_noise, execute


@dataclass
class BenchResults:
    ideal_value: float
    unmit_value: float
    unmit_error: float
    zne_fold_value: float
    zne_fold_error: float
    zne_unopt_value: float
    zne_unopt_error: float
    percent_improvement_unmit: float
    percent_improvement_zne_fold: float
    zne_fold_circuit_depths: list[int]
    zne_unopt_circuit_depths: list[int]


def bench(
    qc: QuantumCircuit,
    backend: Backend = AerSimulator(),
    noise_model: NoiseModel = depolarizing_noise_model(error=0.01),
    shots: int = 10_000,
    scale_factors_zne: list[float] = [1, 3, 5],
    scale_factors_unopt: list[int] = [1, 3, 5],
    fold_method: Callable = zne.scaling.fold_global,
    extrapolation_method: Callable = zne.RichardsonFactory,
    verbose: bool = False,
) -> BenchResults:
    """Calculate ideal, unmitigated, ZNE-fold, and ZNE-unoppt values/data."""

    # Ideal (noiseless) expectation value:
    ideal_value = np.around(execute_no_shot_noise(qc), decimals=1)

    # Unmitigated expectation value:
    unmit_value = execute(circuit=qc, backend=backend, shots=shots, noise_model=noise_model)
    unmit_error = abs(ideal_value - unmit_value)

    # ZNE + Fold:
    folded_circuits = [fold_method(qc, s) for s in scale_factors_zne]
    folded_values = [
        execute(circuit=circ, backend=backend, shots=shots, noise_model=noise_model) for circ in folded_circuits
    ]
    folded_depths = [circ.depth() for circ in folded_circuits]

    factory = extrapolation_method(scale_factors_zne)
    [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_zne, folded_values)]

    zne_fold_value = factory.reduce()

    zne_fold_error = abs(ideal_value - zne_fold_value)

    # ZNE + Unopt:
    unoptimized_circuits = [elementary_recipe(qc, iterations=i) for i in scale_factors_unopt]
    unoptimized_values = [
        execute(circuit=c, backend=backend, shots=shots, noise_model=noise_model) for c in unoptimized_circuits
    ]
    unoptimized_depths = [c.depth() for c in unoptimized_circuits]

    factory = extrapolation_method(scale_factors_unopt)
    [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_unopt, unoptimized_values)]

    zne_unopt_value = factory.reduce()
    zne_unopt_error = abs(ideal_value - zne_unopt_value)

    # Diagnostic information:
    unmit_error = abs(ideal_value - unmit_value)
    zne_fold_error = abs(ideal_value - zne_fold_value)
    zne_unopt_error = abs(ideal_value - zne_unopt_value)

    percent_improvement_unmit = ((unmit_error - zne_unopt_error) / zne_unopt_error) * 100
    percent_improvement_zne_fold = ((zne_fold_error - zne_unopt_error) / zne_unopt_error) * 100

    if verbose:
        print(f"Initial circuit:\n {qc}")
        print(f"Ideal value: {ideal_value}\n")

        print(f"Unmitigated expectation value: {unmit_value}")
        print(f"Unmitigated estimation error: {unmit_error}\n")

        print(f"Noise-scaled expectation values from {fold_method.__name__}: \n {folded_values}")
        print(f"Folded circuit depths: {folded_depths}")

        print(f"The {extrapolation_method.__name__} zero-noise extrapolation is: {zne_fold_value}")

        print(f"ZNE expectation value: {zne_fold_value}")
        print(f"ZNE estimation error: {zne_fold_error}\n")

        print(f"Noise-scaled expectation values from circuit unoptimization: \n {unoptimized_values}")
        print(f"Unoptimized circuit depths: {unoptimized_depths}")

        print(f"The {extrapolation_method.__name__} zero-noise extrapolation is {zne_unopt_value}")

        print(f"ZNE/unopt improvement over unmitigated: {percent_improvement_unmit:.2f}%")
        print(f"ZNE/unopt improvement over ZNE/fold: {percent_improvement_zne_fold:.2f}%")

    return BenchResults(
        ideal_value=ideal_value,
        unmit_value=unmit_value,
        unmit_error=unmit_error,
        zne_fold_value=zne_fold_value,
        zne_fold_error=zne_fold_error,
        zne_unopt_value=zne_unopt_value,
        zne_unopt_error=zne_unopt_error,
        percent_improvement_unmit=percent_improvement_unmit,
        percent_improvement_zne_fold=percent_improvement_zne_fold,
        zne_fold_circuit_depths=folded_depths,
        zne_unopt_circuit_depths=unoptimized_depths,
    )
