"""Benchmarks for ForceAtlas2 performance profiling.

Run with: pytest tests/test_benchmark.py --benchmark-only
"""

import numpy as np
import pytest
import scipy.sparse

from fa2 import ForceAtlas2


def _random_symmetric_graph(n, density=0.1, seed=42):
    """Create a random symmetric adjacency matrix."""
    rng = np.random.RandomState(seed)
    G = rng.random((n, n))
    G = (G + G.T) / 2
    G[G < (1 - density)] = 0
    np.fill_diagonal(G, 0)
    return G


def _random_sparse_graph(n, density=0.05, seed=42):
    """Create a random sparse symmetric matrix."""
    G = _random_symmetric_graph(n, density, seed)
    return scipy.sparse.csr_matrix(G)


class TestBenchmarkSmall:
    """Benchmarks with small graphs (50 nodes)."""

    @pytest.fixture
    def graph_50(self):
        return _random_symmetric_graph(50, density=0.2)

    def test_bench_50_nodes_100_iters(self, benchmark, graph_50):
        fa = ForceAtlas2(verbose=False, seed=42)
        result = benchmark(fa.forceatlas2, graph_50, iterations=100)
        assert len(result) == 50

    def test_bench_50_nodes_no_barneshut(self, benchmark, graph_50):
        fa = ForceAtlas2(verbose=False, seed=42, barnesHutOptimize=False)
        result = benchmark(fa.forceatlas2, graph_50, iterations=100)
        assert len(result) == 50

    def test_bench_50_nodes_linlog(self, benchmark, graph_50):
        fa = ForceAtlas2(verbose=False, seed=42, linLogMode=True)
        result = benchmark(fa.forceatlas2, graph_50, iterations=100)
        assert len(result) == 50


class TestBenchmarkMedium:
    """Benchmarks with medium graphs (200 nodes)."""

    @pytest.fixture
    def graph_200(self):
        return _random_symmetric_graph(200, density=0.05)

    @pytest.fixture
    def sparse_graph_200(self):
        return _random_sparse_graph(200, density=0.05)

    def test_bench_200_nodes_dense(self, benchmark, graph_200):
        fa = ForceAtlas2(verbose=False, seed=42)
        result = benchmark(fa.forceatlas2, graph_200, iterations=50)
        assert len(result) == 200

    def test_bench_200_nodes_sparse(self, benchmark, sparse_graph_200):
        fa = ForceAtlas2(verbose=False, seed=42)
        result = benchmark(fa.forceatlas2, sparse_graph_200, iterations=50)
        assert len(result) == 200


class TestBenchmarkLarge:
    """Benchmarks with larger graphs (500 nodes)."""

    @pytest.fixture
    def graph_500(self):
        return _random_sparse_graph(500, density=0.02)

    def test_bench_500_nodes(self, benchmark, graph_500):
        fa = ForceAtlas2(verbose=False, seed=42)
        result = benchmark(fa.forceatlas2, graph_500, iterations=20)
        assert len(result) == 500

    def test_bench_500_nodes_strong_gravity(self, benchmark, graph_500):
        fa = ForceAtlas2(verbose=False, seed=42, strongGravityMode=True)
        result = benchmark(fa.forceatlas2, graph_500, iterations=20)
        assert len(result) == 500


class TestBenchmarkNetworkX:
    """Benchmarks with NetworkX graphs."""

    @pytest.fixture
    def nx_graph(self):
        import networkx as nx
        return nx.random_geometric_graph(100, 0.15, seed=42)

    def test_bench_networkx_100(self, benchmark, nx_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        result = benchmark(fa.forceatlas2_networkx_layout, nx_graph, iterations=50)
        assert len(result) == 100
