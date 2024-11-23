"""Tests for circuit utility functions."""

from unopt.generator import generate_random_two_qubit_gate_circuit

import pytest


@pytest.mark.parametrize(
    "num_qubits,depth",
    [
        (6, 5),
        (4, 3),
        (8, 10),
    ],
)
def test_generate_random_two_qubit_gate_circuit(num_qubits, depth):
    circuit = generate_random_two_qubit_gate_circuit(num_qubits, depth)
    assert (
        circuit.num_qubits == num_qubits
    ), f"Expected {num_qubits} qubits, got {circuit.num_qubits}"
    assert circuit.depth() >= depth, f"Expected depth >= {depth}, got {circuit.depth()}"
