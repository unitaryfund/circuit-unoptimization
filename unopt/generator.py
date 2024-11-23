"""Generate quantum circuits for testing purposes."""

import random

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary


def generate_random_two_qubit_gate_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """Generate a random quantum circuit consisting of two-qubit gates.

    Args:
        num_qubits: The number of qubits in the circuit.
        depth: The number of layers (depth) of two-qubit gates.

    Returns:
        The generated random circuit.
    """
    circuit = QuantumCircuit(num_qubits)
    qubit_index = list(range(num_qubits))

    for _ in range(depth):
        # Shuffle qubits for randomness.
        random.shuffle(qubit_index)

        for k in range(num_qubits // 2):
            # Select pairs of qubits.
            targets = [qubit_index[2 * k], qubit_index[2 * k + 1]]

            # Generate a random 2-qubit unitary.
            random_unitary_gate = random_unitary(4).to_instruction()

            # Add the random unitary to the circuit.
            circuit.append(random_unitary_gate, targets)

    return circuit
