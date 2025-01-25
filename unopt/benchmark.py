"""Benchmarking module for ZNE and unoptimized circuits."""

from typing import Any, Callable
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from mitiq import zne

from unopt.noise import depolarizing_noise_model
from unopt.recipe import unoptimize_circuit
from unopt.qem import execute_no_shot_noise, execute


@dataclass
class BenchTrialResults:
    trial_number: int
    ideal_value: float
    unmit_value: float
    zne_fold_value: float
    zne_fold_depths: list[int]
    zne_unopt_value: float
    zne_unopt_depths: list[int]
    density_matrix: np.ndarray

    def __str__(self) -> str:
        return (
            f"Trial {self.trial_number}:\n"
            f"  Ideal Value: {self.ideal_value}\n"
            f"  Unmitigated Value: {self.unmit_value}\n"
            f"  ZNE + Fold Value: {self.zne_fold_value}\n"
            f"  ZNE + Fold Depths: {self.zne_fold_depths}\n"
            f"  ZNE + Unopt Value: {self.zne_unopt_value}\n"
            f"  ZNE + Unopt Depths: {self.zne_unopt_depths}\n"
        )


@dataclass
class BenchAverageResults:
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

    def __str__(self) -> str:
        return (
            f"Averages Across All Trials:\n"
            f"  Ideal Value: {self.avg_ideal_value}\n"
            f"  Unmitigated Value: {self.avg_unmit_value}\n"
            f"  ZNE + Fold Value: {self.avg_zne_fold_value} (Error: {self.avg_zne_fold_error})\n"
            f"  ZNE + Unopt Value: {self.avg_zne_unopt_value} (Error: {self.avg_zne_unopt_error})\n"
            f"  Percent Improvement (Unmit): {self.percent_improvement_unmit:.2f}%\n"
            f"  Percent Improvement (ZNE + Fold): {self.percent_improvement_zne_fold:.2f}%\n"
            f"  Original Circuit Depth: {self.original_circuit_depth}\n"
            f"  Avg Folded Depths: {self.avg_zne_fold_circuit_depths}\n"
            f"  Avg Unoptimized Depths: {self.avg_zne_unopt_circuit_depths}\n"
        )


@dataclass
class BenchResults:
    average_results: BenchAverageResults
    trial_results: list[BenchTrialResults]

    def __str__(self) -> str:
        trials_summary = "\n".join(str(trial) for trial in self.trial_results)
        return f"{self.average_results}\n\nTrial Details:\n{trials_summary}"


def bench(
    qc: QuantumCircuit,
    backend: Any = AerSimulator(),
    noise_model: NoiseModel = depolarizing_noise_model(error=0.01),
    shots: int = 10_000,
    scale_factors_zne: list[float] = [1, 3, 5],
    iterations_unopt: list[int] = [1, 2, 3],
    fold_method: Callable = zne.scaling.fold_global,
    extrapolation_method: Callable = zne.RichardsonFactory,
    trials: int = 1,
    verbose: bool = False,
) -> BenchResults:
    """Calculate ideal, unmitigated, ZNE-fold, and ZNE-unopt values/data."""
    trial_results = []
    ideal_values = []
    unmit_values = []
    zne_fold_values = []
    zne_unopt_values = []
    folded_depths_list = []
    unopt_depths_list = []

    original_depth = qc.depth()

    for trial in range(trials):
        if verbose:
            print(f"Running Trial {trial + 1}/{trials}...")

        # Ideal (noiseless) expectation value:
        ideal_value, density_matrix = execute_no_shot_noise(qc, return_density_matrix=True)
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
        zne_fold_value = factory.reduce()
        zne_fold_values.append(zne_fold_value)

        # ZNE + Unopt:
        unoptimized_circuits = [unoptimize_circuit(qc, iterations=i) for i in iterations_unopt]
        unoptimized_values = [
            execute(circuit=c, backend=backend, shots=shots, noise_model=noise_model) for c in unoptimized_circuits
        ]
        unoptimized_depths = [circ.depth() for circ in unoptimized_circuits]
        unopt_depths_list.append(unoptimized_depths)

        scale_factors_unopt = [depth / original_depth for depth in unoptimized_depths]
        factory = extrapolation_method(scale_factors_unopt)
        [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_unopt, unoptimized_values)]
        zne_unopt_value = factory.reduce()
        zne_unopt_values.append(zne_unopt_value)

        # Store trial-specific results
        trial_results.append(
            BenchTrialResults(
                trial_number=trial + 1,
                ideal_value=ideal_value,
                unmit_value=unmit_value,
                zne_fold_value=zne_fold_value,
                zne_fold_depths=folded_depths,
                zne_unopt_value=zne_unopt_value,
                zne_unopt_depths=unoptimized_depths,
                density_matrix=density_matrix,
            )
        )

    # Compute averages
    avg_ideal_value = np.mean(ideal_values)
    avg_unmit_value = np.mean(unmit_values)
    avg_zne_fold_value = np.mean(zne_fold_values)
    avg_zne_unopt_value = np.mean(zne_unopt_values)

    avg_unmit_error = abs(avg_ideal_value - avg_unmit_value)
    avg_zne_fold_error = abs(avg_ideal_value - avg_zne_fold_value)
    avg_zne_unopt_error = abs(avg_ideal_value - avg_zne_unopt_value)

    percent_improvement_unmit = ((avg_unmit_error - avg_zne_unopt_error) / avg_zne_unopt_error) * 100
    percent_improvement_zne_fold = ((avg_zne_fold_error - avg_zne_unopt_error) / avg_zne_unopt_error) * 100

    average_results = BenchAverageResults(
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

    return BenchResults(average_results=average_results, trial_results=trial_results)
