"""Tests for fa2.easy module."""

import json
import os
import tempfile

import pytest

from fa2.easy import layout, visualize

# ── layout() ──


class TestLayout:
    def test_edge_tuples_unweighted(self):
        pos = layout([(0, 1), (1, 2), (2, 0)], seed=42)
        assert len(pos) == 3
        assert all(isinstance(v, tuple) and len(v) == 2 for v in pos.values())

    def test_edge_tuples_weighted(self):
        pos = layout([(0, 1, 5.0), (1, 2, 1.0), (2, 0, 0.5)], seed=42)
        assert len(pos) == 3

    def test_edge_dicts(self):
        edges = [
            {"source": "A", "target": "B", "weight": 5.0},
            {"source": "B", "target": "C"},
            {"source": "A", "target": "C", "weight": 0.1},
        ]
        pos = layout(edges, seed=42)
        assert set(pos.keys()) == {"A", "B", "C"}

    def test_adjacency_dict(self):
        edges = {"A": ["B", "C"], "B": ["A", "D"], "C": ["A"], "D": ["B"]}
        pos = layout(edges, seed=42)
        assert set(pos.keys()) == {"A", "B", "C", "D"}

    def test_string_node_ids(self):
        pos = layout([("alpha", "beta"), ("beta", "gamma")], seed=42)
        assert set(pos.keys()) == {"alpha", "beta", "gamma"}

    def test_empty_edges(self):
        pos = layout([])
        assert pos == {}

    def test_mode_default(self):
        pos = layout([(0, 1), (1, 2)], mode="default", seed=42)
        assert len(pos) == 3

    def test_mode_community(self):
        pos = layout([(0, 1), (1, 2), (2, 0)], mode="community", seed=42)
        assert len(pos) == 3

    def test_mode_hub_dissuade(self):
        pos = layout([(0, 1), (0, 2), (0, 3)], mode="hub-dissuade", seed=42)
        assert len(pos) == 4

    def test_mode_compact(self):
        pos = layout([(0, 1), (1, 2)], mode="compact", seed=42)
        assert len(pos) == 3

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            layout([(0, 1)], mode="nonexistent")

    def test_dim_3(self):
        pos = layout([(0, 1), (1, 2)], dim=3, seed=42)
        assert all(len(v) == 3 for v in pos.values())

    def test_seed_reproducibility(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        p1 = layout(edges, seed=42)
        p2 = layout(edges, seed=42)
        for k in p1:
            assert p1[k] == p2[k]

    def test_node_sizes(self):
        pos = layout([(0, 1), (1, 2)], node_sizes={0: 5.0, 1: 3.0, 2: 8.0}, seed=42)
        assert len(pos) == 3

    def test_node_positions(self):
        pos = layout([(0, 1)], node_positions={0: (0.0, 0.0), 1: (10.0, 10.0)}, seed=42)
        assert len(pos) == 2

    def test_custom_iterations(self):
        pos = layout([(0, 1)], iterations=10, seed=42)
        assert len(pos) == 2

    def test_self_loops_skipped(self):
        pos = layout([(0, 0), (0, 1), (1, 1)], seed=42)
        assert len(pos) == 2  # self-loops don't create new nodes but edges are skipped

    def test_duplicate_edges_deduplicated(self):
        pos = layout([(0, 1), (1, 0), (0, 1)], seed=42)
        assert len(pos) == 2

    def test_invalid_edge_length(self):
        with pytest.raises(ValueError, match="2-tuple or 3-tuple"):
            layout([(0, 1, 2, 3)])

    def test_large_graph(self):
        """50 nodes, fast smoke test."""
        edges = [(i, (i + 1) % 50) for i in range(50)]
        pos = layout(edges, iterations=10, seed=42)
        assert len(pos) == 50


# ── visualize() ──


class TestVisualize:
    def test_matplotlib_output(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        fig = visualize([(0, 1), (1, 2)], output="matplotlib", seed=42)
        assert fig is not None
        plt.close(fig)

    def test_json_output(self):
        result = visualize([(0, 1), (1, 2)], output="json", seed=42)
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result

    def test_png_output(self):
        pytest.importorskip("matplotlib")
        data = visualize([(0, 1), (1, 2)], output="png", seed=42)
        assert isinstance(data, bytes)
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_svg_output(self):
        pytest.importorskip("matplotlib")
        data = visualize([(0, 1), (1, 2)], output="svg", seed=42)
        assert isinstance(data, bytes)
        assert b"<svg" in data

    def test_png_to_file(self):
        pytest.importorskip("matplotlib")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            visualize([(0, 1), (1, 2)], output="png", path=path, seed=42)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_json_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            visualize([(0, 1), (1, 2)], output="json", path=path, seed=42)
            with open(path) as f:
                data = json.load(f)
            assert len(data["nodes"]) == 3
        finally:
            os.unlink(path)

    def test_invalid_output(self):
        with pytest.raises(ValueError, match="Unknown output format"):
            visualize([(0, 1)], output="pdf")

    def test_empty_edges(self):
        result = visualize([], output="json")
        assert result == {}

    def test_with_title(self):
        pytest.importorskip("matplotlib.pyplot")
        import matplotlib.pyplot as plt
        fig = visualize([(0, 1), (1, 2)], output="matplotlib", title="Test", seed=42)
        assert fig is not None
        plt.close(fig)

    def test_community_mode_render(self):
        pytest.importorskip("matplotlib")
        data = visualize(
            [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)],
            mode="community", output="png", seed=42,
        )
        assert isinstance(data, bytes)
