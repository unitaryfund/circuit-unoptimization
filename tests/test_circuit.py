"""Tests for circuit generation utility functions."""

import pytest

from unopt.circuit import generate_random_two_qubit_gate_circuit


@pytest.mark.parametrize(
    "num_qubits,depth",
    [
        (6, 5),
        (4, 3),
        (8, 10),
    ],
)
def test_generate_random_two_qubit_gate_circuit(num_qubits: int, depth: int) -> None:
    circuit = generate_random_two_qubit_gate_circuit(num_qubits, depth)
    assert circuit.num_qubits == num_qubits, f"Expected {num_qubits} qubits, got {circuit.num_qubits}"
    assert circuit.depth() >= depth, f"Expected depth >= {depth}, got {circuit.depth()}"
