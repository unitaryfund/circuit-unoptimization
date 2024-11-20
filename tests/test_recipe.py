from qiskit.quantum_info import Operator
from unopt.recipe import (
    gate_insert,
    gate_swap,
    decompose_circuit,
    synthesize_circuit,
    elementary_recipe,
)


def test_gate_insert_maintains_unitary_equivalence(sample_circuit):
    """Test that unitary equivalence is maintained after gate insertion."""
    original_unitary = Operator(sample_circuit)
    inserted_qc, B1_info = gate_insert(sample_circuit)
    assert B1_info is not None, "B1_info should not be None if gates were inserted."
    inserted_unitary = Operator(inserted_qc)
    assert original_unitary.equiv(inserted_unitary), "Unitary equivalence not maintained after gate insertion."


def test_gate_swap_maintains_unitary_equivalence(sample_circuit):
    """Test that unitary equivalence is maintained after gate swapping."""
    inserted_qc, B1_info = gate_insert(sample_circuit)
    swapped_qc = gate_swap(inserted_qc, B1_info)
    original_unitary = Operator(sample_circuit)
    swapped_unitary = Operator(swapped_qc)
    assert original_unitary.equiv(swapped_unitary), "Unitary equivalence not maintained after gate swapping."


def test_decompose_maintains_unitary_equivalence(sample_circuit):
    """Test that unitary equivalence is maintained after decomposition."""
    inserted_qc, B1_info = gate_insert(sample_circuit)
    swapped_qc = gate_swap(inserted_qc, B1_info)
    decomposed_qc = decompose_circuit(swapped_qc)
    original_unitary = Operator(sample_circuit)
    decomposed_unitary = Operator(decomposed_qc)
    assert original_unitary.equiv(decomposed_unitary), "Unitary equivalence not maintained after decomposition."


def test_synthesize_maintains_unitary_equivalence(sample_circuit):
    """Test that unitary equivalence is maintained after synthesis."""
    inserted_qc, B1_info = gate_insert(sample_circuit)
    swapped_qc = gate_swap(inserted_qc, B1_info)
    decomposed_qc = decompose_circuit(swapped_qc)
    synthesized_qc = synthesize_circuit(decomposed_qc)
    original_unitary = Operator(sample_circuit)
    synthesized_unitary = Operator(synthesized_qc)
    assert original_unitary.equiv(synthesized_unitary), "Unitary equivalence not maintained after synthesis."


def test_elementary_recipe_maintains_unitary_equivalence(sample_circuit):
    """Test that the full elementary recipe maintains unitary equivalence."""
    original_unitary = Operator(sample_circuit)
    processed_qc = elementary_recipe(sample_circuit, iterations=1, strategy="P_c")
    processed_unitary = Operator(processed_qc)
    assert original_unitary.equiv(processed_unitary), "Unitary equivalence not maintained after applying the full recipe."