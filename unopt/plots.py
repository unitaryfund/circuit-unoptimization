"""Plots for diagonostics of circuit unoptimization."""

import matplotlib.pyplot as plt
from unopt.benchmark import BenchResults


def plot_circuit_depths_from_results(results: BenchResults) -> None:
    """Plot circuit depths for ZNE+Fold vs ZNE+Unopt from `BenchResults`.

    Args:
        results: Instance of `BenchResults` returned by `bench`.
    """
    plt.figure(figsize=(10, 6))

    folded_depths = results.zne_fold_circuit_depths
    unopt_depths = results.zne_unopt_circuit_depths
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
