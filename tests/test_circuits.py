"""Tests for circuit utility functions."""
from qiskit import QuantumCircuit
from unopt.circuits import random_two_qubit_circuit


def test_random_circuit_properties():
    num_qubits = 5
    depth = 10

    # Generate the random circuit
    circuit = random_two_qubit_circuit(num_qubits, depth)

    # Assert that the circuit is an instance of QuantumCircuit
    assert isinstance(circuit, QuantumCircuit), "Generated object is not a QuantumCircuit"

    # Assert the circuit has the correct number of qubits
    assert circuit.num_qubits == num_qubits, f"Expected {num_qubits} qubits, but got {circuit.num_qubits}"

    # Assert the circuit has at least `depth` two-qubit gates
    two_qubit_gates = [
        instruction.operation
        for instruction in circuit.data
        if len(instruction.qubits) == 2
    ]
    assert len(two_qubit_gates) >= depth, f"Expected at least {depth} two-qubit gates, but got {len(two_qubit_gates)}"

    # Assert the gates are among the allowed set
    allowed_gates = {"cx", "cz", "swap", "iswap"}
    gate_names = {instruction.operation.name for instruction in circuit.data}
    assert gate_names.issubset(allowed_gates), f"Unexpected gate types found: {gate_names - allowed_gates}"
