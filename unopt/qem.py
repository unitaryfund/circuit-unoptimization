"""Quantum error mitigation with unoptimized circuits."""

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel


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
