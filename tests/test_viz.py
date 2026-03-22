"""Tests for fa2.viz module."""

import json
import os
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fa2 import ForceAtlas2

# Skip all tests if matplotlib not available
plt = pytest.importorskip("matplotlib.pyplot")

from fa2.viz import export_layout, plot_layout  # noqa: E402

# triangle_graph and triangle_positions fixtures are in conftest.py


@pytest.fixture
def nx_graph():
    nx = pytest.importorskip("networkx")
    return nx.karate_club_graph()


@pytest.fixture
def nx_positions(nx_graph):
    fa = ForceAtlas2(verbose=False, seed=42)
    return fa.forceatlas2_networkx_layout(nx_graph, iterations=50)


# ── plot_layout ──


class TestPlotLayout:
    def test_basic_2d(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions)
        assert fig is not None
        plt.close(fig)

    def test_with_networkx(self, nx_graph, nx_positions):
        fig = plot_layout(nx_graph, nx_positions)
        assert fig is not None
        plt.close(fig)

    def test_with_title(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions, title="Test Graph")
        assert fig is not None
        plt.close(fig)

    def test_color_by_degree(self, nx_graph, nx_positions):
        fig = plot_layout(nx_graph, nx_positions, color_by_degree=True)
        assert fig is not None
        plt.close(fig)

    def test_custom_node_color(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions, node_color="red")
        assert fig is not None
        plt.close(fig)

    def test_custom_node_size(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions, node_size=100)
        assert fig is not None
        plt.close(fig)

    def test_show_labels(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions, show_labels=True)
        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self, triangle_graph, triangle_positions):
        fig = plot_layout(triangle_graph, triangle_positions, figsize=(6, 4))
        assert fig is not None
        plt.close(fig)

    def test_with_dict_positions(self, triangle_graph):
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 0.866)}
        fig = plot_layout(triangle_graph, pos)
        assert fig is not None
        plt.close(fig)

    def test_with_list_positions(self, triangle_graph):
        pos = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]
        fig = plot_layout(triangle_graph, pos)
        assert fig is not None
        plt.close(fig)

    def test_sparse_matrix(self, triangle_positions):
        G = csr_matrix(([1, 1, 1, 1, 1, 1], ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])), shape=(3, 3))
        fig = plot_layout(G, triangle_positions)
        assert fig is not None
        plt.close(fig)

    def test_3d_layout(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0.5]])
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_custom_ax(self, triangle_graph, triangle_positions):
        fig, ax = plt.subplots()
        result = plot_layout(triangle_graph, triangle_positions, ax=ax)
        assert result is fig
        plt.close(fig)

    def test_with_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Famous("Petersen")
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=50)
        fig = plot_layout(G, layout)
        assert fig is not None
        plt.close(fig)

    def test_unsupported_dim_raises(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        with pytest.raises(ValueError, match="2D and 3D"):
            plot_layout(G, pos)

    def test_non_zero_based_node_ids(self):
        """NetworkX graph with non-zero-based integer node IDs."""
        nx = pytest.importorskip("networkx")
        G = nx.Graph()
        G.add_edges_from([(10, 20), (20, 30)])
        pos = {10: (0.0, 0.0), 20: (1.0, 0.0), 30: (2.0, 0.0)}
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_string_node_ids(self):
        """NetworkX graph with string node IDs."""
        nx = pytest.importorskip("networkx")
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        pos = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)}
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_3d_with_2d_ax_raises(self):
        """Passing a 2D ax for a 3D layout should raise ValueError."""
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0, 0], [1, 1, 1]])
        fig_2d, ax_2d = plt.subplots()
        with pytest.raises(ValueError, match="3D Axes"):
            plot_layout(G, pos, ax=ax_2d)
        plt.close(fig_2d)

    def test_tight_layout_not_called_on_user_ax(self):
        """When user provides ax, tight_layout should not be called on their figure."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0]])
        result = plot_layout(G, pos, ax=ax1)
        assert result is fig  # Returns the user's figure
        plt.close(fig)


# ── export_layout ──


class TestExportLayout:
    def test_json_returns_dict(self, triangle_graph, triangle_positions):
        result = export_layout(triangle_graph, triangle_positions, fmt="json")
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 3
        # Check node structure
        node = result["nodes"][0]
        assert "id" in node
        assert "x" in node
        assert "y" in node

    def test_json_d3_compatible(self, triangle_graph, triangle_positions):
        result = export_layout(triangle_graph, triangle_positions, fmt="json")
        # Verify it's valid JSON-serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result

    def test_json_to_file(self, triangle_graph, triangle_positions):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = export_layout(triangle_graph, triangle_positions, fmt="json", path=path)
            assert result is None  # writes to file
            with open(path) as f:
                data = json.load(f)
            assert len(data["nodes"]) == 3
        finally:
            os.unlink(path)

    def test_json_with_networkx(self, nx_graph, nx_positions):
        result = export_layout(nx_graph, nx_positions, fmt="json")
        assert len(result["nodes"]) == 34
        assert len(result["edges"]) > 0

    def test_json_weighted(self):
        G = np.array([[0, 5, 0.1], [5, 0, 1], [0.1, 1, 0]], dtype=float)
        pos = np.array([[0, 0], [1, 0], [0.5, 1]])
        result = export_layout(G, pos, fmt="json")
        weights = [e.get("weight") for e in result["edges"] if "weight" in e]
        assert len(weights) > 0

    def test_json_3d(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = np.array([[0, 0, 0], [1, 1, 1]])
        result = export_layout(G, pos, fmt="json")
        assert "z" in result["nodes"][0]

    def test_png_returns_bytes(self, triangle_graph, triangle_positions):
        data = export_layout(triangle_graph, triangle_positions, fmt="png")
        assert isinstance(data, bytes)
        assert data[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes

    def test_png_to_file(self, triangle_graph, triangle_positions):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            export_layout(triangle_graph, triangle_positions, fmt="png", path=path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_svg_returns_bytes(self, triangle_graph, triangle_positions):
        data = export_layout(triangle_graph, triangle_positions, fmt="svg")
        assert isinstance(data, bytes)
        assert b"<svg" in data

    def test_gexf_export(self, triangle_graph, triangle_positions):
        pytest.importorskip("networkx")
        data = export_layout(triangle_graph, triangle_positions, fmt="gexf")
        assert isinstance(data, bytes)
        assert b"gexf" in data.lower()

    def test_graphml_export(self, triangle_graph, triangle_positions):
        pytest.importorskip("networkx")
        data = export_layout(triangle_graph, triangle_positions, fmt="graphml")
        assert isinstance(data, bytes)
        assert b"graphml" in data.lower()

    def test_gexf_to_file(self, triangle_graph, triangle_positions):
        pytest.importorskip("networkx")
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            path = f.name
        try:
            export_layout(triangle_graph, triangle_positions, fmt="gexf", path=path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_invalid_format(self, triangle_graph, triangle_positions):
        with pytest.raises(ValueError, match="Unsupported format"):
            export_layout(triangle_graph, triangle_positions, fmt="pdf")

    def test_case_insensitive_format(self, triangle_graph, triangle_positions):
        result = export_layout(triangle_graph, triangle_positions, fmt="JSON")
        assert isinstance(result, dict)


# ── Integration ──


class TestVizIntegration:
    def test_full_pipeline_networkx(self):
        """Layout → plot → export pipeline."""
        nx = pytest.importorskip("networkx")
        G = nx.karate_club_graph()
        fa = ForceAtlas2(verbose=False, seed=42)
        pos = fa.forceatlas2_networkx_layout(G, iterations=100)

        # Plot
        fig = plot_layout(G, pos, color_by_degree=True, title="Karate Club")
        assert fig is not None
        plt.close(fig)

        # Export JSON
        result = export_layout(G, pos, fmt="json")
        assert len(result["nodes"]) == 34

        # Export PNG
        png = export_layout(G, pos, fmt="png")
        assert len(png) > 1000  # Should be a real image

    def test_full_pipeline_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Famous("Petersen")
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=50)

        fig = plot_layout(G, layout, title="Petersen")
        assert fig is not None
        plt.close(fig)

        result = export_layout(G, layout, fmt="json")
        assert len(result["nodes"]) == 10

    def test_export_gexf_igraph(self):
        """Export igraph graph to GEXF format."""
        ig = pytest.importorskip("igraph")
        nx_mod = pytest.importorskip("networkx")  # noqa: F841
        G = ig.Graph.Ring(4)
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=20)
        data = export_layout(G, layout, fmt="gexf")
        assert isinstance(data, bytes)

    def test_export_png_igraph(self):
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Ring(4)
        fa = ForceAtlas2(verbose=False, seed=42)
        layout = fa.forceatlas2_igraph_layout(G, iterations=20)
        data = export_layout(G, layout, fmt="png")
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_plot_list_positions_raw_matrix(self):
        """Raw matrix with list positions."""
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = [(0.0, 0.0), (1.0, 1.0)]
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_networkx_ndarray_positions(self):
        """NetworkX graph with ndarray positions (not dict)."""
        nx = pytest.importorskip("networkx")
        G = nx.path_graph(3)
        pos = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_igraph_list_positions(self):
        """igraph graph with list positions (not Layout)."""
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Ring(3)
        pos = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_igraph_ndarray_positions(self):
        """igraph graph with ndarray positions."""
        ig = pytest.importorskip("igraph")
        G = ig.Graph.Ring(3)
        pos = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        fig = plot_layout(G, pos)
        assert fig is not None
        plt.close(fig)

    def test_export_graphml_networkx(self):
        """Export NetworkX graph to GraphML (hits the NX copy path)."""
        nx = pytest.importorskip("networkx")
        G = nx.path_graph(3)
        pos = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
        data = export_layout(G, pos, fmt="graphml")
        assert isinstance(data, bytes)
        assert b"graphml" in data.lower()

    def test_export_gexf_3d(self):
        """Export 3D positions to GEXF (hits z-coordinate path)."""
        nx = pytest.importorskip("networkx")
        G = nx.path_graph(3)
        pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
        data = export_layout(G, pos, fmt="gexf")
        assert isinstance(data, bytes)


class TestVizImportErrors:
    """Test ImportError handling for matplotlib."""

    def test_plot_layout_import_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        # Need to reimport to trigger the error
        import numpy as np

        from fa2.viz import plot_layout
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        with pytest.raises(ImportError, match="matplotlib"):
            plot_layout(G, pos)

    def test_export_gexf_import_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "networkx", None)
        import numpy as np

        from fa2.viz import export_layout
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        with pytest.raises(ImportError, match="networkx"):
            export_layout(G, pos, fmt="gexf")


class TestVizEdgeCases:
    """Cover remaining uncovered lines in viz.py."""

    def test_plot_3d_with_existing_ax(self):
        """Test 3D plot with pre-existing 3D axes (line 218)."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        from fa2.viz import plot_layout
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = {0: (0.0, 0.0, 0.0), 1: (1.0, 1.0, 1.0)}
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = plot_layout(G, pos, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_3d_wrong_ax_raises(self):
        """Test 3D plot with 2D axes raises ValueError (line 217-218)."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        from fa2.viz import plot_layout
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = {0: (0.0, 0.0, 0.0), 1: (1.0, 1.0, 1.0)}
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="3D"):
            plot_layout(G, pos, ax=ax)
        plt.close(fig)

    def test_plot_3d_color_by_degree(self):
        """3D plot with numeric colors (covers cmap branch in _plot_3d)."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        from fa2.viz import plot_layout
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0.5]])
        fig = plot_layout(G, pos, color_by_degree=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_3d_with_title(self):
        """Test 3D plot with title (line 232)."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        from fa2.viz import plot_layout
        G = np.array([[0, 1], [1, 0]], dtype=float)
        pos = {0: (0.0, 0.0, 0.0), 1: (1.0, 1.0, 1.0)}
        fig = plot_layout(G, pos, title="3D Test")
        assert fig is not None
        plt.close(fig)
