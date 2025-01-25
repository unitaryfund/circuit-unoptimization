"""Heavy output utilities for the Quantum Volume benchmarking suite."""

import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


def get_ideal_probabilities(model_circuit: QuantumCircuit) -> dict[str, float]:
    """Calculate the ideal probabilities for a quantum circuit.

    Args:
        model_circuit: The quantum circuit to simulate.

    Returns:
        A dictionary mapping bitstrings to their respective probabilities.
    """
    zero = Statevector.from_label("0" * model_circuit.num_qubits)
    sv = zero.evolve(model_circuit)
    return sv.probabilities_dict()


def get_heavy_strings(ideal_probs: dict[str, float]) -> tuple[float, list[str]]:
    """Determine the heavy output strings and the median probability.

    Args:
        ideal_probs: A dictionary of ideal probabilities.

    Returns:
        A tuple containing:
            - The median probability.
            - A list of heavy strings (bitstrings with probabilities above the median).
    """
    prob_median = float(np.real(np.median(list(ideal_probs.values()))))
    heavy_strings = list(
        filter(
            lambda x: ideal_probs[x] > prob_median,
            list(ideal_probs.keys()),
        )
    )
    return prob_median, heavy_strings


def hop(counts: dict[str, int], ideal_probs: dict[str, float]) -> float:
    """Calculate the heavy output probability from counts and ideal probabilities.

    Args:
        counts: A dictionary of bitstring counts from quantum circuit execution.
        ideal_probs: A dictionary of ideal probabilities.

    Returns:
        The heavy output probability.
    """
    _, heavy_strings = get_heavy_strings(ideal_probs)
    shots = sum(counts.values())
    return sum([counts.get(value, 0) for value in heavy_strings]) / shots


def calc_z_value(mean: float, sigma: float) -> float:
    """Calculate the z-value based on the mean and standard deviation.

    Args:
        mean: The mean value.
        sigma: The standard deviation.

    Returns:
        The z-value.
    """
    if sigma == 0:
        sigma = 1e-10
        print("Standard deviation sigma should not be zero.")
    return (mean - 2 / 3.0) / sigma


def calc_confidence_level(z_value: float) -> float:
    """Calculate the confidence level based on the z-value.

    Args:
        z_value: The z-value.

    Returns:
        The confidence level.
    """
    return 0.5 * (1 + math.erf(z_value / 2**0.5))


def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Evaluate a quadratic function of the form ax^2 + bx + c.

    Args:
        x: The input value.
        a: The coefficient of the quadratic term (x^2).
        b: The coefficient of the linear term (x).
        c: The constant term.

    Returns:
        The result of the quadratic equation for the given x, a, b, and c.
    """
    return a * x**2 + b * x + c


def get_exact_hop(qc: QuantumCircuit) -> tuple[float, dict[str, float]]:
    """Compute the exact heavy output probability (HOP) for a quantum circuit.

    Args:
        qc: The quantum circuit for which the exact HOP is calculated.

    Returns:
        A tuple containing:
            - The exact heavy output probability (HOP).
            - A dictionary of ideal probabilities for all bitstrings.
    """
    qc.remove_final_measurements()
    ideal_probs = get_ideal_probabilities(qc)
    _, heavy_strings = get_heavy_strings(ideal_probs)
    theoretical_HOP = sum(ideal_probs[s] for s in heavy_strings)
    return theoretical_HOP, ideal_probs
