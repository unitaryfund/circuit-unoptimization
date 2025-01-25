"""Tests for quantum volume (QV) functions."""

import pytest
import math
from qiskit import QuantumCircuit
from unopt.qv import (
    get_ideal_probabilities,
    get_heavy_strings,
    hop,
    calc_z_value,
    calc_confidence_level,
    quadratic,
    get_exact_hop,
)


@pytest.mark.parametrize(
    "x,a,b,c,expected",
    [
        (0, 1, 2, 3, 3),
        (1, 1, 2, 3, 6),
        (2, 1, 2, 3, 11),
        (3, 2, -1, 4, 19),
    ],
)
def test_quadratic(x: float, a: float, b: float, c: float, expected: float) -> None:
    """Test the quadratic function."""
    result = quadratic(x, a, b, c)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_get_ideal_probabilities(simple_circuit: QuantumCircuit) -> None:
    """Test get_ideal_probabilities with a simple circuit."""
    qc_no_measure = simple_circuit.remove_final_measurements(inplace=False)
    probs = get_ideal_probabilities(qc_no_measure)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert set(probs.keys()) == {"0", "1"}  # Expect '0' and '1' for single qubit


def test_get_heavy_strings() -> None:
    """Test get_heavy_strings with an example probability dictionary."""
    ideal_probs = {"00": 0.4, "01": 0.1, "10": 0.2, "11": 0.3}
    median, heavy_strings = get_heavy_strings(ideal_probs)
    assert math.isclose(median, 0.25, rel_tol=1e-9)
    assert set(heavy_strings) == {"00", "11"}


def test_hop() -> None:
    """Test heavy output probability (HOP) calculation."""
    counts = {"00": 500, "01": 300, "10": 100, "11": 100}
    ideal_probs = {"00": 0.6, "01": 0.1, "10": 0.1, "11": 0.2}
    hop_value = hop(counts, ideal_probs)
    assert math.isclose(hop_value, 0.6, rel_tol=1e-9)


@pytest.mark.parametrize(
    "mean,sigma,expected_z",
    [
        (0.7, 0.1, 0.3333),
        (2 / 3, 0.1, 0.0),
        (0.6, 0.2, -0.3333),
    ],
)
def test_calc_z_value(mean: float, sigma: float, expected_z: float) -> None:
    z_value = calc_z_value(mean, sigma)
    assert math.isclose(z_value, expected_z, rel_tol=1e-4)


@pytest.mark.parametrize(
    "z_value,expected_confidence",
    [
        (0.0, 0.5),
        (1.0, 0.8413447461),
        (-1.0, 0.1586552539),
    ],
)
def test_calc_confidence_level(z_value: float, expected_confidence: float) -> None:
    """Test confidence level calculation."""
    confidence = calc_confidence_level(z_value)
    assert math.isclose(confidence, expected_confidence, rel_tol=1e-9)


def test_get_exact_hop(two_qubit_circuit: QuantumCircuit) -> None:
    """Test get_exact_hop for a two-qubit circuit."""
    qc_no_measure = two_qubit_circuit.remove_final_measurements(inplace=False)
    theoretical_hop, ideal_probs = get_exact_hop(qc_no_measure)

    # Check that the sum of probabilities equals 1.0
    assert math.isclose(sum(ideal_probs.values()), 1.0, rel_tol=1e-9)

    # Check that the theoretical HOP matches the expected value (in this case, 0.0, as no heavy strings exist)
    assert math.isclose(theoretical_hop, 0.0, rel_tol=1e-9)
