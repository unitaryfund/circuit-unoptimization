import pytest
from qiskit import QuantumCircuit


@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """Creates a simple quantum circuit with a single qubit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return qc


@pytest.fixture
def two_qubit_circuit() -> QuantumCircuit:
    """Create a two-qubit entangled circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def counts() -> dict[str, int]:
    """Provide example counts from circuit execution."""
    return {"00": 500, "01": 300, "10": 100, "11": 100}
