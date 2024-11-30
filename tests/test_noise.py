"""Tests for noise model utility functions."""

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from unopt.noise import amplitude_damping_noise_model, depolarizing_noise_model


@pytest.mark.parametrize(
    "noise_model_func,params,expected_gates",
    [
        (amplitude_damping_noise_model, {"prob_1": 0.05, "prob_2": 0.1}, {"u1", "u2", "u3", "cx"}),
        (depolarizing_noise_model, {"error": 0.02}, {"u1", "u2", "u3", "cx"}),
    ],
)
def test_noise_model_creation_and_simulation(
    noise_model_func: NoiseModel, params: dict[str, float], expected_gates: set[str]
) -> None:
    """Test noise model creation and its usability in a simulation."""
    # Create the noise model.
    noise_model = noise_model_func(**params)

    # Check the type of the noise model.
    assert isinstance(noise_model, NoiseModel), "The noise model should be an instance of NoiseModel"

    # Check that all expected gates are in the noise model.
    registered_gates = set(noise_model.noise_instructions)
    assert registered_gates == expected_gates, f"Unexpected gates in noise model: {registered_gates}"

    # Test the noise model in a simulation.
    simulator = AerSimulator()

    # Define a simple test circuit.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Execute the circuit with the noise model.
    result = simulator.run(qc, noise_model=noise_model, shots=100).result()
    assert result.success, f"Simulation with {noise_model_func.__name__} failed"
