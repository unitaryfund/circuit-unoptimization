"""Qiskit noise models for simulating hardware backend for error mitigation."""

from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
)


def amplitude_damping_noise_model(prob_1: float = 0.04, prob_2: float = 0.08) -> NoiseModel:
    """Defines an amplitude damping noise model with one-qubit and two-qubit errors.

    Args:
        prob_1: One-qubit gate error rate (default 4%).
        prob_2: Two-qubit gate error rate (default 8%).

    Returns:
        Amplitude damping noise model.
    """
    # One and two-qubit error rates.
    error_1 = amplitude_damping_error(param_amp=prob_1, excited_state_population=1)
    error_2 = amplitude_damping_error(param_amp=prob_2, excited_state_population=1)
    error_2 = error_2.tensor(error_2)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    return noise_model


def depolarizing_noise_model(error: float = 0.01) -> NoiseModel:
    """Defines an depolarizing noise model with one-qubit.

    Args:
        error: One-qubit gate error rate (default 1%).

    Returns:
        Depolarizing noise model.
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 1), ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 2), "cx")

    return noise_model
