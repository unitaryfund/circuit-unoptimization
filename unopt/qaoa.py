"""Quantum Approximate Optimization Algorithm (QAOA) utilities."""

from qiskit import QuantumCircuit
import networkx as nx


def calculate_max_cut_cost(cut_vec: str, G: nx.Graph) -> int:
    """Compute the Max-Cut cost for a given bitstring representation of a cut.

    Args:
        cut_vec: A binary string representing a partition of the graph nodes.
        G: The input graph, represented as a NetworkX Graph.

    Returns:
        The weight of the cut, i.e., the number of edges crossing between partitions.
    """
    cut_weight = 0
    for i, j in G.edges():
        if cut_vec[i] != cut_vec[j]:
            cut_weight += 1
    return cut_weight


def create_qaoa_circuit(G: nx.Graph, p: int, betas: list[float], gammas: list[float], n: int) -> QuantumCircuit:
    """Create a QAOA quantum circuit for the Max-Cut problem.

    Args:
        G: The input graph, represented as a NetworkX Graph.
        p: The number of QAOA layers (depth of the circuit).
        betas: A list of beta parameters for the QAOA ansatz.
        gammas: A list of gamma parameters for the QAOA ansatz.
        n: The number of qubits (equal to the number of nodes in G).

    Returns:
        A Qiskit QuantumCircuit implementing the QAOA ansatz for Max-Cut.
    """
    qc = QuantumCircuit(n)

    # Apply Hadamard gates to all qubits (initialize in superposition)
    for i in range(n):
        qc.h(i)
    qc.barrier()

    # Apply p layers of QAOA
    for p_idx in range(p):
        # Apply cost Hamiltonian (based on graph edges)
        for i, j in G.edges():
            qc.cx(i, j)
            qc.rz(-1 * gammas[p_idx], j)
            qc.cx(i, j)
        qc.barrier()

        # Apply mixing Hamiltonian (single-qubit X rotations)
        for i in range(n):
            qc.rx(betas[p_idx], i)

    qc.measure_all()
    return qc


def measure_sample_cuts(counts: dict[str, int], G: nx.Graph) -> list[int]:
    """Convert measurement outcomes into Max-Cut costs based on sampled bitstrings.

    Args:
        counts: A dictionary mapping measured bitstrings to their frequency.
        G: The input graph, represented as a NetworkX Graph.

    Returns:
        A list of cut values, where each value corresponds to a sampled bitstring.
    """
    cuts = []
    for bitstring, count in counts.items():
        cut_value = calculate_max_cut_cost(bitstring[::-1], G)  # Reverse for endianness
        cuts.extend([cut_value] * count)  # Append cut values according to measurement frequency
    return cuts
