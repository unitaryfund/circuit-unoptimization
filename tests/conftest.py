import pytest
from qiskit import QuantumCircuit


@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Creates a simple quantum circuit with a single qubit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return qc
