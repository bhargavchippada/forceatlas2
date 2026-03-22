"""Shared test fixtures for ForceAtlas2 tests."""

import numpy as np
import pytest
import scipy.sparse


@pytest.fixture
def small_dense_graph():
    """A small 5-node dense adjacency matrix."""
    return np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=float)


@pytest.fixture
def small_sparse_graph(small_dense_graph):
    """Same graph as small_dense_graph but in scipy sparse format."""
    return scipy.sparse.csr_matrix(small_dense_graph)


@pytest.fixture
def weighted_graph():
    """A small weighted graph."""
    return np.array([
        [0, 2.5, 0, 0],
        [2.5, 0, 1.0, 3.0],
        [0, 1.0, 0, 0.5],
        [0, 3.0, 0.5, 0],
    ], dtype=float)


@pytest.fixture
def single_node_graph():
    """A graph with a single node and no edges."""
    return np.array([[0]], dtype=float)


@pytest.fixture
def two_node_graph():
    """A graph with two connected nodes."""
    return np.array([
        [0, 1],
        [1, 0],
    ], dtype=float)


@pytest.fixture
def disconnected_graph():
    """A graph with two disconnected components."""
    return np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)


@pytest.fixture
def complete_graph_5():
    """Complete graph on 5 nodes."""
    g = np.ones((5, 5), dtype=float)
    np.fill_diagonal(g, 0)
    return g


@pytest.fixture
def networkx_graph():
    """A NetworkX graph for integration tests."""
    import networkx as nx
    G = nx.random_geometric_graph(30, 0.3, seed=42)
    return G


@pytest.fixture
def triangle_graph():
    """A 3-node triangle graph."""
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


@pytest.fixture
def triangle_positions():
    """Equilateral triangle-ish positions."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
