import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from unopt.qem import execute, execute_no_shot_noise
from unopt.noise import depolarizing_noise_model


def test_execute_no_noise(simple_circuit: QuantumCircuit) -> None:
    """Test the `execute` function without a noise model."""
    simulator = AerSimulator()
    result = execute(simple_circuit, backend=simulator, shots=1000)
    # For a Hadamard gate, the expectation value of Z is 0
    assert np.isclose(result, 0.0, atol=0.1), f"Expected 0, but got {result}"


def test_execute_with_noise(simple_circuit: QuantumCircuit) -> None:
    """Test the `execute` function with a depolarizing noise model."""
    simulator = AerSimulator()
    noise_model = depolarizing_noise_model(error=0.1)
    result = execute(simple_circuit, backend=simulator, shots=1000, noise_model=noise_model)
    # Expect a deviation from 0 due to noise
    assert -1.0 < result < 1.0, "Result should be bounded between -1 and 1"


def test_empty_circuit() -> None:
    """Test `execute` and `execute_no_shot_noise` on an empty circuit."""
    qc = QuantumCircuit(1)
    simulator = AerSimulator()
    result = execute(qc, backend=simulator, shots=1000)
    noiseless_result = execute_no_shot_noise(qc)
    # Expect default value (all 0 state, Z expectation = 1)
    assert np.isclose(result, 1.0), f"Unexpected result for empty circuit: {result}"
    assert np.isclose(noiseless_result, 1.0), f"Unexpected result for empty circuit: {noiseless_result}"
