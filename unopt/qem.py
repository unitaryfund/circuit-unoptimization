"""Quantum error mitigation with unoptimized circuits."""

from typing import Callable
from collections import defaultdict
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel

from mitiq import zne

from unopt.noise_models import depolarizing_noise_model
from unopt.recipe import elementary_recipe


def bench(
    qc: QuantumCircuit,
    backend: Backend = AerSimulator(),
    noise_model: NoiseModel = depolarizing_noise_model(error=0.01),
    shots: int = 10_000,
    scale_factors_zne: list[float] = [1, 3, 5],
    scale_factors_unopt: list[int] = [1, 3, 5],
    fold_method: Callable = zne.scaling.fold_global,
    extrapolation_method: Callable = zne.RichardsonFactory,
    num_trials: int = 1,
) -> defaultdict[int, dict[str, float]]:
    """Calculate ideal, unmitigated, ZNE-fold, and ZNE-unoppt values/data."""
    print(f"Initial circuit:\n {qc}")

    results: defaultdict[int, dict[str, float]] = defaultdict(dict)

    for trial in range(1, num_trials + 1):
        print("********************************************")
        print(f"Trial number {trial}")
        print("********************************************")

        # Ideal (noiseless) expectation value:
        ideal_value = np.around(execute_no_shot_noise(qc), decimals=1)
        print(f"Ideal value: {ideal_value}\n")

        # Unmitigated expectation value:
        unmit_value = execute(circuit=qc, backend=backend, shots=shots, noise_model=noise_model)
        unmit_error = abs(ideal_value - unmit_value)
        print(f"Unmitigated expectation value: {unmit_value}")
        print(f"Unmitigated estimation error: {unmit_error}\n")

        # ZNE + Fold:
        folded_circuits = [fold_method(qc, s) for s in scale_factors_zne]
        folded_values = [
            execute(circuit=circ, backend=backend, shots=shots, noise_model=noise_model) for circ in folded_circuits
        ]
        print(f"Noise-scaled expectation values from {fold_method.__name__}: \n {folded_values}")

        factory = extrapolation_method(scale_factors_zne)
        [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_zne, folded_values)]

        zne_fold_value = factory.reduce()
        print(f"The {extrapolation_method.__name__} zero-noise extrapolation is: {zne_fold_value}")

        zne_fold_error = abs(ideal_value - zne_fold_value)
        print(f"ZNE expectation value: {zne_fold_value}")
        print(f"ZNE estimation error: {zne_fold_error}\n")

        # ZNE + Unopt:
        unoptimized_circuits = [elementary_recipe(qc, iterations=i) for i in scale_factors_unopt]
        unoptimized_values = [
            execute(circuit=c, backend=backend, shots=shots, noise_model=noise_model) for c in unoptimized_circuits
        ]
        print(f"Noise-scaled expectation values from circuit unoptimization: \n {unoptimized_values}")

        factory = extrapolation_method(scale_factors_unopt)
        [factory.push({"scale_factor": s}, val) for s, val in zip(scale_factors_unopt, unoptimized_values)]

        zne_unopt_value = factory.reduce()
        zne_unopt_error = abs(ideal_value - zne_unopt_value)
        print(f"The {extrapolation_method.__name__} zero-noise extrapolation is {zne_unopt_value}")

        # Diagnostic information:
        unmit_error = abs(ideal_value - unmit_value)
        zne_fold_error = abs(ideal_value - zne_fold_value)
        zne_unopt_error = abs(ideal_value - zne_unopt_value)

        percent_improvement_unmit = ((unmit_error - zne_unopt_error) / zne_unopt_error) * 100
        percent_improvement_zne_fold = ((zne_fold_error - zne_unopt_error) / zne_unopt_error) * 100

        print(f"ZNE/unopt improvement over unmitigated: {percent_improvement_unmit:.2f}%")
        print(f"ZNE/unopt improvement over ZNE/fold: {percent_improvement_zne_fold:.2f}%")

        results[trial] = {
            "ideal_value": ideal_value,
            "unmit_value": unmit_value,
            "unmit_error": unmit_error,
            "zne_fold_value": zne_fold_value,
            "zne_fold_error": zne_fold_error,
            "zne_unopt_value": zne_unopt_value,
            "zne_unopt_error": zne_unopt_error,
            "percent_improvement_unmit": percent_improvement_unmit,
            "percent_improvement_zne_fold": percent_improvement_zne_fold,
        }

    return results


def execute(
    circuit: QuantumCircuit,
    backend: Backend,
    shots: float,
    noise_model: NoiseModel | None = None,
) -> float:
    """Executor for a Qiskit circuit."""
    circuit_with_measurement = circuit.copy()
    circuit_with_measurement.measure_all()

    # Transpile the circuit:
    compiled_circuit = transpile(
        circuit_with_measurement,
        backend,
        basis_gates=noise_model.basis_gates if noise_model is not None else None,
        optimization_level=0,
    )

    # Execute the circuit:
    result = backend.run(compiled_circuit, noise_model=noise_model, shots=shots).result()
    counts = result.get_counts()

    # Calculate expectation value of Z on qubit 0.
    total_counts = sum(counts.values())
    expectation = 0.0
    for outcome, count in counts.items():
        # Reverse the outcome string due to Qiskit's little-endian ordering.
        bitstring = outcome[::-1]
        if bitstring[0] == "0":
            expectation += count / total_counts
        else:
            expectation -= count / total_counts
    return expectation


def execute_no_shot_noise(qc: QuantumCircuit, noise_model: NoiseModel | None = None) -> float:
    """Executor that uses density matrix simulator to reduce all shot noise.

    Adapted from the `execute_with_noise` function from mitiq.
    https://github.com/unitaryfund/mitiq/blob/ee85edf48557c85c4f6d0b1e1d74d74fc882c9d4/mitiq/interface/mitiq_qiskit/qiskit_utils.py#L85
    """
    qc = qc.copy()
    qc.save_density_matrix()

    basis_gates = None if noise_model is None else noise_model.basis_gates + ["save_density_matrix"]

    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    job = backend.run(qc, optimization_level=0, noise_model=noise_model, shots=1, basis_gates=basis_gates)

    rho = np.asarray(job.result().data()["density_matrix"])
    return float(rho[0, 0].real)
