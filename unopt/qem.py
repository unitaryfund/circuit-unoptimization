"""Quantum error mitigation with unoptimized circuits."""

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_aer.noise import NoiseModel


def execute(
    circuit: QuantumCircuit,
    backend: Backend,
    shots: int,
    noise_model: NoiseModel | None = None,
) -> float:
    """
    Execute a Qiskit quantum circuit and calculate the expectation value of the Z operator
    on the 0th qubit.

    This function measures all qubits in the circuit, simulates the circuit on the specified
    backend, and calculates the expectation value of the Z operator on the first qubit
    based on the measurement outcomes.

    Args:
        circuit (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The Qiskit backend to run the circuit on.
        shots (int): The number of measurement shots to use.
        noise_model (NoiseModel | None, optional): An optional noise model to simulate.
            If provided, the circuit will be simulated with the specified noise model.

    Returns:
        float: The expectation value of the Z operator on the 0th qubit.

    Notes:
        - Qiskit follows little-endian ordering for qubits. The outcome bitstrings
          are reversed during processing to ensure correct calculations.
        - The function assumes that the input circuit does not already contain measurement
          operations, as it adds measurement gates to all qubits in the circuit.
    """
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


def execute_no_shot_noise(
    qc: QuantumCircuit, noise_model: NoiseModel | None = None, return_density_matrix: bool = False
) -> tuple[float, np.ndarray | None]:
    """Executor that uses density matrix simulator to reduce all shot noise.

    Adapted from the `execute_with_noise` function from mitiq.
    https://github.com/unitaryfund/mitiq/blob/ee85edf48557c85c4f6d0b1e1d74d74fc882c9d4/mitiq/interface/mitiq_qiskit/qiskit_utils.py#L85

    Args:
        qc: The quantum circuit to execute.
        noise_model: The noise model to apply, if any.
        return_density_matrix: Whether to include the density matrix in the result.

    Returns:
        A tuple containing:
        - The expectation value of the Z operator on qubit 0.
        - The density matrix, or None if `return_density_matrix` is False.
    """
    qc = qc.copy()
    qc.save_density_matrix()

    basis_gates = None if noise_model is None else noise_model.basis_gates + ["save_density_matrix"]

    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    job = backend.run(qc, optimization_level=0, noise_model=noise_model, shots=1, basis_gates=basis_gates)

    rho = np.asarray(job.result().data()["density_matrix"])
    expectation_value = float(rho[0, 0].real)

    if return_density_matrix:
        return expectation_value, rho
    return expectation_value, None
