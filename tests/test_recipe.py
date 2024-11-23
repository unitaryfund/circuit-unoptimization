"""Tests for the components of the elementary recipe (ER)."""

import pytest
from qiskit.quantum_info import Operator
from unopt.recipe import elementary_recipe
from unopt.generator import generate_random_two_qubit_gate_circuit


@pytest.mark.parametrize(
    "strategy,iterations,circuit_generator",
    [
        ("P_c", 1, lambda: generate_random_two_qubit_gate_circuit(4, 5)),
        ("P_c", 2, lambda: generate_random_two_qubit_gate_circuit(6, 10)),
        ("P_r", 1, lambda: generate_random_two_qubit_gate_circuit(4, 5)),
        ("P_r", 2, lambda: generate_random_two_qubit_gate_circuit(6, 10)),
    ],
)
def test_elementary_recipe_maintains_unitary_equivalence(
    strategy, iterations, circuit_generator
):
    """Test that the full elementary recipe maintains unitary equivalence."""
    sample_circuit = circuit_generator()
    original_unitary = Operator(sample_circuit)
    processed_qc = elementary_recipe(
        sample_circuit, iterations=iterations, strategy=strategy
    )
    processed_unitary = Operator(processed_qc)
    assert original_unitary.equiv(processed_unitary), (
        f"Unitary equivalence not maintained for strategy={strategy}, "
        f"iterations={iterations}, circuit={sample_circuit}"
    )
