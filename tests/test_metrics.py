"""Tests for fa2.metrics module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fa2 import ForceAtlas2
from fa2.metrics import edge_crossing_count, neighborhood_preservation, stress

# triangle_graph and triangle_positions fixtures are in conftest.py


@pytest.fixture
def path_graph():
    """0 - 1 - 2 - 3"""
    G = np.zeros((4, 4), dtype=float)
    G[0, 1] = G[1, 0] = 1
    G[1, 2] = G[2, 1] = 1
    G[2, 3] = G[3, 2] = 1
    return G


@pytest.fixture
def path_positions():
    """Nodes in a line — perfect for a path graph."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])


# ── stress ──


class TestStress:
    def test_perfect_layout_low_stress(self, path_graph, path_positions):
        """A path graph with linearly spaced positions should have low stress."""
        s = stress(path_graph, path_positions)
        assert s < 0.1  # Near-perfect

    def test_single_node(self):
        G = np.array([[0]], dtype=float)
        s = stress(G, np.array([[0.0, 0.0]]))
        assert s == 0.0

    def test_two_nodes(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        s = stress(G, np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert s == 0.0

    def test_stress_is_float(self, triangle_graph, triangle_positions):
        s = stress(triangle_graph, triangle_positions)
        assert isinstance(s, float)
        assert s >= 0.0

    def test_with_dict_positions(self, triangle_graph):
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 0.866)}
        s = stress(triangle_graph, pos)
        assert isinstance(s, float)

    def test_with_sparse_matrix(self, path_positions):
        G = csr_matrix(([1, 1, 1, 1, 1, 1], ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4))
        s = stress(G, path_positions)
        assert isinstance(s, float)
        assert s < 0.1

    def test_with_networkx(self):
        nx = pytest.importorskip("networkx")
        G = nx.path_graph(4)
        pos = {i: (float(i), 0.0) for i in range(4)}
        s = stress(G, pos)
        assert isinstance(s, float)
        assert s < 0.1

    def test_bad_layout_higher_stress(self, path_graph):
        """Scrambled positions should have higher stress than ordered ones."""
        good = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        bad = np.array([[3, 0], [0, 0], [1, 0], [2, 0]], dtype=float)
        s_good = stress(path_graph, good)
        s_bad = stress(path_graph, bad)
        assert s_bad > s_good

    def test_disconnected_graph(self):
        """Disconnected nodes should not crash."""
        G = np.zeros((3, 3), dtype=float)
        G[0, 1] = G[1, 0] = 1
        pos = np.array([[0, 0], [1, 0], [5, 5]], dtype=float)
        s = stress(G, pos)
        assert isinstance(s, float)


# ── edge_crossing_count ──


class TestEdgeCrossingCount:
    def test_no_crossings_path(self, path_graph, path_positions):
        count = edge_crossing_count(path_graph, path_positions)
        assert count == 0

    def test_no_crossings_triangle(self, triangle_graph, triangle_positions):
        count = edge_crossing_count(triangle_graph, triangle_positions)
        assert count == 0

    def test_crossing_detected(self):
        """Two edges that cross: (0,2) and (1,3) in a square with diagonal."""
        G = np.zeros((4, 4), dtype=float)
        G[0, 2] = G[2, 0] = 1  # diagonal
        G[1, 3] = G[3, 1] = 1  # other diagonal
        pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        count = edge_crossing_count(G, pos)
        assert count == 1

    def test_shared_endpoint_not_counted(self):
        """Edges sharing an endpoint should not be counted as crossing."""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        count = edge_crossing_count(G, pos)
        assert count == 0

    def test_with_networkx(self):
        nx = pytest.importorskip("networkx")
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3)])
        pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
        count = edge_crossing_count(G, pos)
        assert count == 1

    def test_3d_raises(self, triangle_graph):
        pos_3d = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
        with pytest.raises(ValueError, match="2D"):
            edge_crossing_count(triangle_graph, pos_3d)

    def test_empty_graph(self):
        G = np.zeros((3, 3), dtype=float)
        pos = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        assert edge_crossing_count(G, pos) == 0

    def test_with_sparse(self):
        G = csr_matrix(([1, 1, 1, 1], ([0, 2, 1, 3], [2, 0, 3, 1])), shape=(4, 4))
        pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        count = edge_crossing_count(G, pos)
        assert count == 1


# ── neighborhood_preservation ──


class TestNeighborhoodPreservation:
    def test_perfect_path(self, path_graph, path_positions):
        """Path with linear positions should have high NP."""
        np_score = neighborhood_preservation(path_graph, path_positions, k=2)
        assert np_score >= 0.8

    def test_single_node(self):
        G = np.array([[0]], dtype=float)
        np_score = neighborhood_preservation(G, np.array([[0, 0]]))
        assert np_score == 1.0

    def test_returns_float_in_range(self, triangle_graph, triangle_positions):
        np_score = neighborhood_preservation(triangle_graph, triangle_positions, k=2)
        assert isinstance(np_score, float)
        assert 0.0 <= np_score <= 1.0

    def test_k_capped_at_n_minus_1(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0]], dtype=float)
        # k=100 but only 2 nodes, should cap to k=1
        np_score = neighborhood_preservation(G, pos, k=100)
        assert isinstance(np_score, float)

    def test_with_networkx(self):
        nx = pytest.importorskip("networkx")
        G = nx.path_graph(5)
        pos = {i: (float(i), 0.0) for i in range(5)}
        np_score = neighborhood_preservation(G, pos, k=2)
        assert np_score >= 0.5

    def test_with_dict_positions(self, triangle_graph):
        pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.866)}
        np_score = neighborhood_preservation(triangle_graph, pos, k=2)
        assert 0.0 <= np_score <= 1.0

    def test_scrambled_lower_than_ordered(self, path_graph):
        good = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        bad = np.array([[3, 3], [0, 0], [1, 1], [2, 2]], dtype=float)
        np_good = neighborhood_preservation(path_graph, good, k=2)
        np_bad = neighborhood_preservation(path_graph, bad, k=2)
        assert np_good >= np_bad

    def test_k_zero_raises(self, triangle_graph, triangle_positions):
        with pytest.raises(ValueError, match="k must be positive"):
            neighborhood_preservation(triangle_graph, triangle_positions, k=0)

    def test_k_negative_raises(self, triangle_graph, triangle_positions):
        with pytest.raises(ValueError, match="k must be positive"):
            neighborhood_preservation(triangle_graph, triangle_positions, k=-1)


# ── Validation ──


class TestValidation:
    def test_position_graph_size_mismatch_stress(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)  # 3 positions, 2 nodes
        with pytest.raises(ValueError, match="positions has 3.*graph has 2"):
            stress(G, pos)

    def test_position_graph_size_mismatch_np(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        with pytest.raises(ValueError, match="positions has 3.*graph has 2"):
            neighborhood_preservation(G, pos, k=1)

    def test_all_nodes_same_position(self):
        """All nodes at origin — layout_mean is 0, should not crash."""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        pos = np.array([[0, 0], [0, 0], [0, 0]], dtype=float)
        s = stress(G, pos)
        assert isinstance(s, float)

    def test_stress_single_node_zero_denom(self):
        """Single node means no pairs — denom is 0, should return 0."""
        G = np.array([[0]], dtype=float)
        pos = np.array([[5.0, 5.0]])
        assert stress(G, pos) == 0.0

    def test_stress_with_list_positions(self, path_graph):
        """Ensure list positions path is covered."""
        pos = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        s = stress(path_graph, pos)
        assert isinstance(s, float)

    def test_np_disconnected_graph(self):
        """Disconnected nodes in neighborhood_preservation."""
        G = np.zeros((4, 4), dtype=float)
        G[0, 1] = G[1, 0] = 1
        pos = np.array([[0, 0], [1, 0], [5, 5], [6, 6]], dtype=float)
        np_score = neighborhood_preservation(G, pos, k=2)
        assert isinstance(np_score, float)

    def test_stress_with_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Ring(5)
        pos = [(float(i), 0.0) for i in range(5)]
        s = stress(G, pos)
        assert isinstance(s, float)

    def test_np_with_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Ring(5)
        pos = [(float(i), 0.0) for i in range(5)]
        np_score = neighborhood_preservation(G, pos, k=2)
        assert isinstance(np_score, float)

    def test_crossing_with_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph(4, [(0, 2), (1, 3)])
        pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        count = edge_crossing_count(G, pos)
        assert count == 1


# ── Integration with ForceAtlas2 ──


class TestMetricsIntegration:
    def test_fa2_layout_measurable(self):
        """Metrics work on actual ForceAtlas2 output."""
        nx = pytest.importorskip("networkx")
        G = nx.karate_club_graph()
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(G, iterations=100)

        s = stress(G, pos)
        crossings = edge_crossing_count(G, pos)
        np_score = neighborhood_preservation(G, pos, k=5)

        assert isinstance(s, float) and s > 0
        assert isinstance(crossings, int) and crossings >= 0
        assert 0.0 <= np_score <= 1.0

    def test_more_iterations_lower_stress(self):
        """More iterations should produce lower stress (seeded, deterministic)."""
        nx = pytest.importorskip("networkx")
        G = nx.karate_club_graph()

        pos_10 = ForceAtlas2(verbose=False, seed=42).forceatlas2_networkx_layout(G, iterations=10)
        pos_200 = ForceAtlas2(verbose=False, seed=42).forceatlas2_networkx_layout(G, iterations=200)

        s_10 = stress(G, pos_10)
        s_200 = stress(G, pos_200)

        assert s_200 < s_10, f"Expected 200-iter stress ({s_200}) < 10-iter stress ({s_10})"
