"""Plots for diagonostics of circuit unoptimization."""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from unopt.recipe import elementary_recipe


def plot_circuit_growth(qc: QuantumCircuit, iterations: int = 5) -> tuple[list[int], list[float]]:
    """Plot the growth of circuit depth across iterations of the recipe.

    Args:
        qc: The original quantum circuit.
        iterations: Number of recipe iterations to apply.
    """
    depths = [qc.depth()]  # Original depth
    for i in range(1, iterations + 1):
        qc_unoptimized = elementary_recipe(qc, iterations=i)
        depths.append(qc_unoptimized.depth())

    # Plot depth growth
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations + 1), depths, marker="o", label="Circuit Depth")
    plt.title("Circuit Growth Across Recipe Iterations")
    plt.xlabel("Recipe Iterations")
    plt.ylabel("Circuit Depth")
    plt.grid()
    plt.legend()
    plt.show()

    # Print depth ratios
    ratios = [depths[0] / d for d in depths[1:]]
    print(f"Depth Ratios (Original Depth / Depth After Iteration): {ratios}")
    return depths, ratios


def compare_scale_factors(qc: QuantumCircuit, iterations: int = 5) -> None:
    """Compare scale factors from folding and unoptimization.

    Args:
        qc: The original quantum circuit.
        iterations: Number of iterations to compare.
    """
    # Folding scale factors
    folded_scale_factors = [1 + 2 * i for i in range(iterations + 1)]  # Typical for folding

    # Unoptimization scale factors
    _, unoptimization_ratios = plot_circuit_growth(qc, iterations=iterations)

    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations + 1), folded_scale_factors, label="Folding Scale Factors", marker="o")
    plt.plot(range(1, iterations + 1), unoptimization_ratios, label="Unoptimization Scale Factors", marker="s")
    plt.title("Comparison of Scale Factors")
    plt.xlabel("Recipe Iterations")
    plt.ylabel("Scale Factor")
    plt.grid()
    plt.legend()
    plt.show()


def extrapolation_fit(scale_factors: list[float], noisy_values: list[float]) -> float:
    """Fit extrapolation models to scale factors and noisy expectation values.

    Args:
        scale_factors: Scale factors corresponding to noisy circuits.
        noisy_values: Noisy expectation values at each scale factor.

    Returns:
        Extrapolated value at zero noise.
    """
    # Perform linear fitting for extrapolation
    coeffs = np.polyfit(scale_factors, noisy_values, 1)  # Linear fit
    extrapolated_value = np.polyval(coeffs, 0)  # Extrapolate to zero noise

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(scale_factors, noisy_values, label="Noisy Values", color="red")
    plt.plot(scale_factors, np.polyval(coeffs, scale_factors), label="Linear Fit", color="blue")
    plt.axvline(x=0, color="green", linestyle="--", label="Zero Noise")
    plt.title("Extrapolation to Zero Noise")
    plt.xlabel("Scale Factor")
    plt.ylabel("Expectation Value")
    plt.grid()
    plt.legend()
    plt.show()

    print(f"Extrapolated Zero-Noise Value: {extrapolated_value}")
    return extrapolated_value
