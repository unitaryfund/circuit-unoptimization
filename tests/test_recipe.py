"""Tests for the components of the elementary recipe (ER)."""

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from unopt.circuit import generate_random_two_qubit_gate_circuit
from unopt.recipe import decompose, unoptimize_circuit


@pytest.mark.parametrize(
    "strategy,iterations,circuit_generator,decomposition_method",
    [
        ("P_c", 1, lambda: generate_random_two_qubit_gate_circuit(4, 5), "default"),
        ("P_c", 2, lambda: generate_random_two_qubit_gate_circuit(6, 10), "default"),
        ("P_r", 1, lambda: generate_random_two_qubit_gate_circuit(4, 5), "kak"),
        ("P_r", 2, lambda: generate_random_two_qubit_gate_circuit(6, 10), "kak"),
        ("P_c", 1, lambda: generate_random_two_qubit_gate_circuit(4, 5), "basis"),
        ("P_c", 2, lambda: generate_random_two_qubit_gate_circuit(6, 10), "basis"),
    ],
)
def test_unoptimize_circuit_unitary_equivalence(
    strategy: str, iterations: int, circuit_generator: QuantumCircuit, decomposition_method: str
) -> None:
    """Test that the full elementary recipe maintains unitary equivalence."""
    sample_circuit = circuit_generator()
    original_unitary = Operator(sample_circuit)

    # Apply the decomposition method explicitly during the recipe.
    sample_circuit = decompose(sample_circuit, method=decomposition_method)

    processed_qc = unoptimize_circuit(sample_circuit, iterations=iterations, strategy=strategy)
    processed_unitary = Operator(processed_qc)

    assert original_unitary.equiv(processed_unitary), (
        f"Unitary equivalence not maintained for strategy={strategy}, "
        f"iterations={iterations}, decomposition_method={decomposition_method}, circuit={sample_circuit}"
    )
