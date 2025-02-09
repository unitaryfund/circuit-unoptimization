"""Tests for quantum approximate optimization algorithm (QAOA) functions."""

import pytest
import networkx as nx
from unopt.qaoa import calculate_max_cut_cost


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Creates a simple 4-node fully connected graph for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    return G


def test_calculate_max_cut_cost(sample_graph: nx.Graph) -> None:
    """Tests that the max cut cost is correctly computed."""
    # No cut.
    assert calculate_max_cut_cost("0000", sample_graph) == 0
    # Optimal cut.
    assert calculate_max_cut_cost("0101", sample_graph) == 4
    assert calculate_max_cut_cost("0011", sample_graph) == 4
    assert calculate_max_cut_cost("0110", sample_graph) == 4
    assert calculate_max_cut_cost("1010", sample_graph) == 4
