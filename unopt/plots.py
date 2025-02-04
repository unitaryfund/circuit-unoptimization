"""Plots for diagonostics of circuit unoptimization."""

import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

from qiskit_aer import AerSimulator
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile
from scipy.stats import linregress
from scipy.optimize import curve_fit

from unopt.benchmark import BenchResults
from unopt.recipe import unoptimize_circuit
from unopt.qv import get_exact_hop, hop
from unopt.qaoa import create_qaoa_circuit, measure_sample_cuts, calculate_max_cut_cost
from unopt.utils import quadratic
from unopt.noise import depolarizing_noise_model


def plot_circuit_depths_from_results(results: BenchResults) -> None:
    """Plot circuit depths for ZNE+Fold vs ZNE+Unopt from `BenchResults`.

    Args:
        results: Instance of `BenchResults` returned by `bench`.
    """
    avg_results = results.average_results  # Access the BenchAverageResults object

    plt.figure(figsize=(10, 6))

    folded_depths = avg_results.avg_zne_fold_circuit_depths
    unopt_depths = avg_results.avg_zne_unopt_circuit_depths
    scale_factors_zne = range(1, len(folded_depths) + 1)
    scale_factors_unopt = range(1, len(unopt_depths) + 1)

    # Plot the depths
    plt.plot(
        scale_factors_zne,
        folded_depths,
        label="ZNE+Fold",
        marker="o",
        linestyle="--",
    )
    plt.plot(
        scale_factors_unopt,
        unopt_depths,
        label="ZNE+Unopt",
        marker="s",
        linestyle="-",
    )

    plt.title("Circuit Depth Growth Across Scale Factors")
    plt.xlabel("Scale Factor")
    plt.ylabel("Circuit Depth")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_avg_circuit_depths(results: BenchResults) -> None:
    """Plot average circuit depths for ZNE+Fold vs ZNE+Unopt from `BenchResults`.

    Args:
        results: Instance of `BenchResults` returned by `bench`.
    """
    avg_results = results.average_results  # Access the BenchAverageResults object

    plt.figure(figsize=(10, 6))

    folded_depths = avg_results.avg_zne_fold_circuit_depths
    unopt_depths = avg_results.avg_zne_unopt_circuit_depths
    scale_factors_zne = range(1, len(folded_depths) + 1)
    scale_factors_unopt = range(1, len(unopt_depths) + 1)

    # Plot the averaged depths
    plt.plot(
        scale_factors_zne,
        folded_depths,
        label="ZNE+Fold (Avg Depths)",
        marker="o",
        linestyle="--",
    )
    plt.plot(
        scale_factors_unopt,
        unopt_depths,
        label="ZNE+Unopt (Avg Depths)",
        marker="s",
        linestyle="-",
    )

    plt.title("Average Circuit Depth Growth Across Scale Factors")
    plt.xlabel("Scale Factor")
    plt.ylabel("Circuit Depth (Averaged Over Trials)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_quantum_volume(
    num_qubits: int,
    unoptimization_strategy: str = "P_c",
    unoptimization_rounds: int = 35,
    seed: int = 10,
    shots: int = 1_000_000,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # (Square) Quantum Volume circuits (equal depth and width).
    qc = QuantumVolume(num_qubits=num_qubits, depth=num_qubits, seed=seed)
    qc.measure_all()
    qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3)
    print(f"{qc.count_ops()=}")

    theoretical_HOP, ideal_probs = get_exact_hop(copy.deepcopy(qc))
    print(f"Ideal HOP: {theoretical_HOP}")

    init = dict(qc.count_ops())["u3"] + dict(qc.count_ops())["cx"]

    x, y = [], []
    for idx in range(unoptimization_rounds):
        print(f"Running iteration {idx} out of {unoptimization_rounds}")

        scaled = unoptimize_circuit(copy.deepcopy(qc), iterations=idx, strategy=unoptimization_strategy)
        scaled = transpile(scaled, basis_gates=["u3", "cx"], optimization_level=3)
        scaled_count = dict(scaled.count_ops())["u3"] + dict(scaled.count_ops())["cx"]
        scale_factor = scaled_count / float(init)

        backend = AerSimulator()
        noise_model = depolarizing_noise_model(error=0.001)
        # noise_model = amplitude_damping_noise_model()

        result = backend.run(scaled, noise_model=noise_model, shots=shots).result()
        counts = result.get_counts()

        experimental_prob = hop(counts, ideal_probs)
        print(f"{experimental_prob=}")
        y.append(experimental_prob)
        x.append(scale_factor)
    print(f"{x=}, {y=}")

    res = linregress(x, y)

    print(f"Zero-noise limit: {res.intercept=} -- {res.slope=}")

    plt.scatter(x, y, marker=".", color="blue")
    linx = np.linspace(0, max(x), 100)
    plt.plot(
        linx, [res.slope * i + res.intercept for i in linx], "--", color="blue", alpha=0.7, label="Linear Best Fit"
    )

    popt, _ = curve_fit(quadratic, x, y)
    print(f"Fitted quadratic coeffs: {popt}")
    plt.plot(linx, quadratic(np.array(linx), *popt), "--", color="green", alpha=0.7, label="Quadratic Best Fit")

    # factory = zne.inference.RichardsonFactory(scale_factors=x)
    # [factory.push({"scale_factor": s}, val) for s, val in zip(x, y)]
    # richardson_extrapolation = factory.reduce()
    # print("richardson_extrapolation", richardson_extrapolation)

    plt.scatter([0], [theoretical_HOP], marker="o", color="red", label="Ideal HOP", zorder=5)
    plt.scatter(
        [0],
        [res.intercept],
        marker="*",
        color="gold",
        label="Zero Noise Linear Fit Intercept",
        zorder=10,
        alpha=1,
        s=60,
    )
    plt.scatter(
        [0], [popt[2]], marker="+", color="gold", label="Zero Noise Quadratic Fit Intercept", zorder=10, alpha=1, s=60
    )
    # plt.scatter([0], [richardson_extrapolation], marker="x", color="gold", label="Zero Noise Richardson Extrapolation", zorder=10, alpha=1, s=60)

    plt.plot(linx, [0.5 for _ in linx], "--", label="Decohered Noise HOP", color="black")

    plt.ylabel("Heavy Output Probability")
    plt.xlabel("$\\lambda$")
    plt.grid()

    fig = plt.gcf()
    fig.set_size_inches(7, 3.2)
    plt.legend(ncol=2, prop={"size": 9})
    plt.tight_layout()
    plt.show()
    plt.savefig(f"ZNE_QV_{unoptimization_strategy}_{num_qubits}.pdf")
    plt.close()


def plot_qaoa(
    num_qubits: int = 12,
    unoptimization_strategy: str = "P_c",
    unoptimization_rounds: int = 35,
    seed: int = 1,
    shots: int = 1_000_000,
) -> None:
    G = nx.random_regular_graph(3, num_qubits, seed=seed)
    p = 2

    all_possible_cuts = [bin(k)[2:].rjust(num_qubits, "0") for k in range(2**num_qubits)]
    all_costs = [calculate_max_cut_cost(cut, G) for cut in all_possible_cuts]

    maxcost = max(all_costs)
    print(f"{maxcost=}")

    # Random Angles, which do not work well:
    # betas = [random.uniform(0, math.pi) for i in range(p)]
    # gammas = [random.uniform(0, math.pi) for i in range(p)]

    # Fixed angles from https://arxiv.org/abs/2107.00677
    # gammas = [0.616]
    # betas = [0.393]
    gammas = [0.488, 0.898]
    betas = [0.555, 0.293]

    qc = create_qaoa_circuit(G, p, betas, gammas, num_qubits)
    print(f"Pre-transpiled gate operations: {qc.count_ops()}")

    qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3)
    print(f"Post-transpiled gate operations: {qc.count_ops()}")

    init = dict(qc.count_ops())["u3"] + dict(qc.count_ops())["cx"]

    shots = 10_000_000
    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    cuts = measure_sample_cuts(counts, G)
    no_noise_cut_value = np.mean(cuts)
    print(f"No noise: {no_noise_cut_value}")

    x, y = [], []
    for idx in range(1, unoptimization_rounds):
        print(f"Running iteration {idx} out of {unoptimization_rounds}")

        scaled = unoptimize_circuit(copy.deepcopy(qc), iterations=idx, strategy=unoptimization_strategy)
        scaled = transpile(scaled, basis_gates=["u3", "cx"], optimization_level=3)
        scaled_count = dict(scaled.count_ops())["u3"] + dict(scaled.count_ops())["cx"]
        scale_factor = scaled_count / float(init)

        backend = AerSimulator()
        noise_model = depolarizing_noise_model(error=0.001)
        shots = 1_000_000

        result = backend.run(scaled, noise_model=noise_model, shots=shots).result()
        counts = result.get_counts()

        cuts = measure_sample_cuts(counts, G)
        experimental_prob = np.mean(cuts)

        print(f"{scale_factor=}, {experimental_prob=}")
        x.append(scale_factor)
        y.append(experimental_prob)

    print(f"Scaled factors: {x}")
    print(f"Experimental probs: {y}")

    res = linregress(x, y)
    print(f"Zero-noise limit: {res.intercept}")

    print(f"{res.slope}")

    linx = np.linspace(0, max(x), 100)
    popt, _ = curve_fit(quadratic, x, y)
    print(f"Fitted quadratic coeffs: {popt}")

    plt.scatter(x, y, marker=".", color="blue")
    plt.plot(
        linx, [res.slope * i + res.intercept for i in linx], "--", color="blue", alpha=0.7, label="Linear Best Fit"
    )

    plt.plot(linx, quadratic(np.array(linx), *popt), "--", color="green", alpha=0.7, label="Quadratic Best Fit")
    plt.scatter([0], [no_noise_cut_value], marker="o", color="red", label="Noiseless Cut Value", zorder=5)
    plt.scatter(
        [0],
        [res.intercept],
        marker="*",
        color="gold",
        label="Zero Noise Linear Fit Intercept",
        zorder=10,
        alpha=1,
        s=60,
    )
    plt.scatter(
        [0], [popt[2]], marker="+", color="gold", label="Zero Noise Quadratic Fit Intercept", zorder=10, alpha=1, s=60
    )

    if unoptimization_strategy == "P_c":
        plt.title("Concatenated Strategy")
    if unoptimization_strategy == "P_r":
        plt.title("Random Strategy")

    plt.ylabel("Cut Value")
    plt.xlabel("$\\lambda$")
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(7, 3.2)
    plt.legend(ncol=2, prop={"size": 9})
    plt.tight_layout()
    plt.savefig(f"ZNE_QAOA_{unoptimization_strategy}_{num_qubits}.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    unoptmization_strategy = "P_c"

    num_qubits = 10
    plot_quantum_volume(num_qubits, unoptimization_strategy=unoptmization_strategy)

    num_qubits = 12
    plot_qaoa(num_qubits, unoptimization_strategy=unoptmization_strategy)
