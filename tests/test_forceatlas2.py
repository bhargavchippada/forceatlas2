"""Integration tests for ForceAtlas2 layout algorithm."""

import numpy as np
import pytest
import scipy.sparse

from fa2 import ForceAtlas2


class TestForceAtlas2Init:
    def test_default_params(self):
        fa = ForceAtlas2(verbose=False)
        assert fa.barnesHutOptimize is True
        assert fa.gravity == 1.0
        assert fa.scalingRatio == 2.0

    def test_adjust_sizes_raises(self):
        with pytest.raises(NotImplementedError, match="adjustSizes"):
            ForceAtlas2(adjustSizes=True)

    def test_multithreaded_raises(self):
        with pytest.raises(NotImplementedError, match="multiThreaded"):
            ForceAtlas2(multiThreaded=True)

    def test_negative_scaling_ratio_raises(self):
        with pytest.raises(ValueError, match="scalingRatio"):
            ForceAtlas2(scalingRatio=-1.0)

    def test_negative_gravity_raises(self):
        with pytest.raises(ValueError, match="gravity"):
            ForceAtlas2(gravity=-1.0)

    def test_zero_barnes_hut_theta_raises(self):
        with pytest.raises(ValueError, match="barnesHutTheta"):
            ForceAtlas2(barnesHutTheta=0)

    def test_negative_edge_weight_influence_raises(self):
        with pytest.raises(ValueError, match="edgeWeightInfluence"):
            ForceAtlas2(edgeWeightInfluence=-0.5)

    def test_zero_jitter_tolerance_raises(self):
        with pytest.raises(ValueError, match="jitterTolerance"):
            ForceAtlas2(jitterTolerance=0)

    def test_negative_jitter_tolerance_raises(self):
        with pytest.raises(ValueError, match="jitterTolerance"):
            ForceAtlas2(jitterTolerance=-1.0)


class TestTimer:
    def test_stop_without_start(self):
        from fa2.forceatlas2 import Timer
        t = Timer()
        t.stop()  # must not raise
        assert t.total_time == 0.0

    def test_display_output(self, capsys):
        from fa2.forceatlas2 import Timer
        t = Timer(name="TestTimer")
        t.start()
        t.stop()
        t.display()
        captured = capsys.readouterr()
        assert "TestTimer" in captured.out
        assert "seconds" in captured.out


class TestInitGraph:
    def test_dense_matrix(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False)
        nodes, edges = fa.init(small_dense_graph)
        assert len(nodes) == 5
        assert len(edges) > 0

    def test_sparse_matrix(self, small_sparse_graph):
        fa = ForceAtlas2(verbose=False)
        nodes, edges = fa.init(small_sparse_graph)
        assert len(nodes) == 5
        assert len(edges) > 0

    def test_non_square_raises(self):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ValueError, match="not square"):
            fa.init(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_non_symmetric_raises(self):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ValueError, match="not symmetric"):
            fa.init(np.array([[0, 1], [0, 0]], dtype=float))

    def test_sparse_non_symmetric_raises(self):
        fa = ForceAtlas2(verbose=False)
        G = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
        with pytest.raises(ValueError, match="not symmetric"):
            fa.init(G)

    def test_sparse_non_square_raises(self):
        fa = ForceAtlas2(verbose=False)
        G = scipy.sparse.csr_matrix(np.ones((2, 3)))
        with pytest.raises(ValueError, match="not square"):
            fa.init(G)

    def test_pos_not_ndarray_raises(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="numpy ndarray or None"):
            fa.init(small_dense_graph, pos=[[0, 0]] * 5)

    def test_sparse_self_loop_warns(self):
        import warnings
        G = scipy.sparse.csr_matrix(np.array([[1.0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        fa = ForceAtlas2(verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fa.init(G)
            assert any("self-loop" in str(warning.message).lower() for warning in w)

    def test_invalid_type_raises(self):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="numpy ndarray or scipy sparse"):
            fa.init([[1, 0], [0, 1]])

    def test_pos_wrong_shape_raises(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False)
        pos = np.array([[1, 2, 3]] * 5)  # Wrong: (5, 3) instead of (5, 2)
        with pytest.raises(ValueError, match="shape"):
            fa.init(small_dense_graph, pos=pos)

    def test_pos_wrong_count_raises(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False)
        pos = np.array([[1, 2]] * 3)  # Wrong: 3 positions for 5 nodes
        with pytest.raises(ValueError, match="shape"):
            fa.init(small_dense_graph, pos=pos)

    def test_initial_positions(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = np.array([[i, i] for i in range(5)], dtype=float)
        nodes, edges = fa.init(small_dense_graph, pos=pos)
        assert nodes[0].x == 0.0
        assert nodes[0].y == 0.0
        assert nodes[3].x == 3.0

    def test_weighted_graph_edges(self, weighted_graph):
        fa = ForceAtlas2(verbose=False)
        nodes, edges = fa.init(weighted_graph)
        weights = [e.weight for e in edges]
        assert 2.5 in weights
        assert 0.5 in weights


class TestForceAtlas2Layout:
    def test_basic_layout(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(small_dense_graph, iterations=50)
        assert len(pos) == 5
        assert all(len(p) == 2 for p in pos)

    def test_sparse_layout(self, small_sparse_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(small_sparse_graph, iterations=50)
        assert len(pos) == 5

    def test_seed_reproducibility(self, small_dense_graph):
        fa1 = ForceAtlas2(verbose=False, seed=42)
        pos1 = fa1.forceatlas2(small_dense_graph, iterations=100)

        fa2 = ForceAtlas2(verbose=False, seed=42)
        pos2 = fa2.forceatlas2(small_dense_graph, iterations=100)

        for p1, p2 in zip(pos1, pos2):
            assert p1[0] == pytest.approx(p2[0])
            assert p1[1] == pytest.approx(p2[1])

    def test_different_seeds_different_results(self, small_dense_graph):
        fa1 = ForceAtlas2(verbose=False, seed=42)
        pos1 = fa1.forceatlas2(small_dense_graph, iterations=100)

        fa2 = ForceAtlas2(verbose=False, seed=99)
        pos2 = fa2.forceatlas2(small_dense_graph, iterations=100)

        # At least some positions should differ
        diffs = [abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) for p1, p2 in zip(pos1, pos2)]
        assert max(diffs) > 0.01

    def test_single_node(self, single_node_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(single_node_graph, iterations=10)
        assert len(pos) == 1

    def test_two_nodes(self, two_node_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(two_node_graph, iterations=50)
        assert len(pos) == 2
        # Nodes should not be at the same position
        dist = ((pos[0][0] - pos[1][0])**2 + (pos[0][1] - pos[1][1])**2)**0.5
        assert dist > 0

    def test_complete_graph(self, complete_graph_5):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(complete_graph_5, iterations=50)
        assert len(pos) == 5

    def test_disconnected_graph(self, disconnected_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(disconnected_graph, iterations=100)
        assert len(pos) == 4

    def test_without_barnes_hut(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42, barnesHutOptimize=False)
        pos = fa.forceatlas2(small_dense_graph, iterations=50)
        assert len(pos) == 5

    def test_strong_gravity(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42, strongGravityMode=True)
        pos = fa.forceatlas2(small_dense_graph, iterations=50)
        assert len(pos) == 5

    def test_dissuade_hubs(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42, outboundAttractionDistribution=True)
        pos = fa.forceatlas2(small_dense_graph, iterations=50)
        assert len(pos) == 5

    def test_linlog_mode(self, small_dense_graph):
        fa = ForceAtlas2(verbose=False, seed=42, linLogMode=True)
        pos = fa.forceatlas2(small_dense_graph, iterations=50)
        assert len(pos) == 5

    def test_linlog_produces_different_layout(self, small_dense_graph):
        fa_lin = ForceAtlas2(verbose=False, seed=42, linLogMode=False)
        pos_lin = fa_lin.forceatlas2(small_dense_graph, iterations=200)

        fa_log = ForceAtlas2(verbose=False, seed=42, linLogMode=True)
        pos_log = fa_log.forceatlas2(small_dense_graph, iterations=200)

        diffs = [abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) for p1, p2 in zip(pos_lin, pos_log)]
        assert max(diffs) > 0.01

    def test_edge_weight_influence(self, weighted_graph):
        fa = ForceAtlas2(verbose=False, seed=42, edgeWeightInfluence=0.5)
        pos = fa.forceatlas2(weighted_graph, iterations=50)
        assert len(pos) == 4

    def test_callbacks(self, small_dense_graph):
        history = []

        def record(iteration, nodes):
            history.append(iteration)

        fa = ForceAtlas2(verbose=False, seed=42)
        fa.forceatlas2(small_dense_graph, iterations=10, callbacks=[record])
        assert history == list(range(10))

    def test_multiple_callbacks(self, small_dense_graph):
        counts = [0, 0]

        def cb1(i, nodes):
            counts[0] += 1

        def cb2(i, nodes):
            counts[1] += 1

        fa = ForceAtlas2(verbose=False, seed=42)
        fa.forceatlas2(small_dense_graph, iterations=5, callbacks=[cb1, cb2])
        assert counts == [5, 5]


class TestNetworkXLayout:
    def test_basic_layout(self, networkx_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(networkx_graph, iterations=50)
        assert len(pos) == len(networkx_graph.nodes())
        assert all(len(v) == 2 for v in pos.values())

    def test_with_initial_positions(self, networkx_graph):
        import networkx as nx
        init_pos = nx.spring_layout(networkx_graph, seed=42)
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(networkx_graph, pos=init_pos, iterations=50)
        assert len(pos) == len(networkx_graph.nodes())

    def test_invalid_graph_type(self):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="networkx Graph"):
            fa.forceatlas2_networkx_layout("not a graph")

    def test_weight_attr(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edge(0, 1, my_weight=2.0)
        G.add_edge(1, 2, my_weight=0.5)
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(G, iterations=50, weight_attr="my_weight")
        assert len(pos) == 3

    def test_reproducibility(self, networkx_graph):
        fa1 = ForceAtlas2(verbose=False, seed=42)
        pos1 = fa1.forceatlas2_networkx_layout(networkx_graph, iterations=50)

        fa2 = ForceAtlas2(verbose=False, seed=42)
        pos2 = fa2.forceatlas2_networkx_layout(networkx_graph, iterations=50)

        for node in pos1:
            assert pos1[node][0] == pytest.approx(pos2[node][0])
            assert pos1[node][1] == pytest.approx(pos2[node][1])

    def test_directed_graph_raises(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ValueError, match="undirected"):
            fa.forceatlas2_networkx_layout(G)

    def test_partial_pos_raises(self, networkx_graph):
        fa = ForceAtlas2(verbose=False, seed=42)
        partial_pos = {list(networkx_graph.nodes())[0]: (0.0, 0.0)}
        with pytest.raises(ValueError, match="missing an entry"):
            fa.forceatlas2_networkx_layout(networkx_graph, pos=partial_pos)

    def test_callbacks_with_networkx(self, networkx_graph):
        called = [False]

        def cb(i, nodes):
            called[0] = True

        fa = ForceAtlas2(verbose=False, seed=42)
        fa.forceatlas2_networkx_layout(networkx_graph, iterations=5, callbacks=[cb])
        assert called[0]


class TestIgraphLayout:
    def test_basic_layout(self):
        import igraph
        G = igraph.Graph.Famous("Petersen")
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=50)
        assert len(layout) == G.vcount()

    def test_weighted_layout(self):
        import igraph
        G = igraph.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)], directed=False)
        G.es["weight"] = [2.0, 1.0, 0.5]
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=50, weight_attr="weight")
        assert len(layout) == 4

    def test_edgeless_graph(self):
        import igraph
        G = igraph.Graph(n=3, edges=[], directed=False)
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=10)
        assert len(layout) == 3

    def test_single_node(self):
        import igraph
        G = igraph.Graph(n=1, edges=[], directed=False)
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=10)
        assert len(layout) == 1

    def test_directed_graph_raises(self):
        import igraph
        G = igraph.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ValueError, match="undirected"):
            fa.forceatlas2_igraph_layout(G)

    def test_invalid_type_raises(self):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="igraph Graph"):
            fa.forceatlas2_igraph_layout("not a graph")

    def test_with_initial_positions(self):
        import igraph
        G = igraph.Graph.Famous("Petersen")
        pos = np.array([[i * 0.1, i * 0.2] for i in range(G.vcount())])
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, pos=pos, iterations=50)
        assert len(layout) == G.vcount()

    def test_reproducibility(self):
        import igraph
        G = igraph.Graph.Famous("Petersen")
        fa1 = ForceAtlas2(verbose=False, seed=42)
        l1 = fa1.forceatlas2_igraph_layout(G, iterations=50)

        fa2 = ForceAtlas2(verbose=False, seed=42)
        l2 = fa2.forceatlas2_igraph_layout(G, iterations=50)

        for c1, c2 in zip(l1.coords, l2.coords):
            assert c1 == pytest.approx(c2)


class TestSelfLoops:
    def test_self_loop_warns(self):
        import warnings
        G = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fa.forceatlas2(G, iterations=5)
            assert any("self-loop" in str(warning.message).lower() for warning in w)


class TestLargerGraphs:
    """Test with larger graphs to catch scaling issues."""

    def test_medium_graph(self):
        """100-node graph should complete without errors."""
        size = 100
        # Create a random sparse symmetric matrix
        rng = np.random.RandomState(42)
        G = rng.random((size, size))
        G = (G + G.T) / 2
        G[G < 0.95] = 0
        np.fill_diagonal(G, 0)

        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == size

    def test_sparse_medium_graph(self):
        """100-node sparse graph."""
        size = 100
        rng = np.random.RandomState(42)
        G = rng.random((size, size))
        G = (G + G.T) / 2
        G[G < 0.95] = 0
        np.fill_diagonal(G, 0)
        G_sparse = scipy.sparse.csr_matrix(G)

        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(G_sparse, iterations=20)
        assert len(pos) == size

    def test_zero_weight_edges(self):
        """Graph with some zero-weight edges should work."""
        G = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3


class TestVerboseMode:
    def test_verbose_with_barneshut(self, small_dense_graph, capsys):
        fa = ForceAtlas2(verbose=True, seed=42, barnesHutOptimize=True)
        pos = fa.forceatlas2(small_dense_graph, iterations=2)
        captured = capsys.readouterr()
        assert "BarnesHut" in captured.out
        assert "Repulsion" in captured.out
        assert len(pos) == 5

    def test_verbose_without_barneshut(self, small_dense_graph, capsys):
        fa = ForceAtlas2(verbose=True, seed=42, barnesHutOptimize=False)
        fa.forceatlas2(small_dense_graph, iterations=2)
        captured = capsys.readouterr()
        assert "Repulsion" in captured.out
        assert "BarnesHut" not in captured.out


class TestNetworkXEdgeCases:
    def test_pos_not_dict_raises(self, networkx_graph):
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="dictionary or None"):
            fa.forceatlas2_networkx_layout(networkx_graph, pos=[(0, 0)] * 30)

    def test_networkx_not_installed(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "networkx", None)
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ImportError, match="networkx is required"):
            fa.forceatlas2_networkx_layout("anything")

    def test_networkx_empty_graph(self):
        import networkx as nx
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(G, iterations=5)
        assert len(pos) == 2


class TestIgraphEdgeCases:
    def test_igraph_not_installed(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "igraph", None)
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(ImportError, match="igraph is required"):
            fa.forceatlas2_igraph_layout("anything")

    def test_igraph_pos_invalid_type_raises(self):
        import igraph
        G = igraph.Graph.Famous("Petersen")
        fa = ForceAtlas2(verbose=False)
        with pytest.raises(TypeError, match="list, numpy array, or None"):
            fa.forceatlas2_igraph_layout(G, pos="bad_pos")

    def test_igraph_pos_as_list(self):
        import igraph
        G = igraph.Graph.Famous("Petersen")
        pos_list = [[i * 0.1, i * 0.2] for i in range(G.vcount())]
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, pos=pos_list, iterations=5)
        assert len(layout) == G.vcount()
