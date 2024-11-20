import pytest
from unopt.circuits import random_two_qubit_circuit


@pytest.fixture
def sample_circuit():
    """Fixture to create a random circuit for testing."""
    return random_two_qubit_circuit(4, 5)
