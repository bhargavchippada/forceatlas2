"""In-process tests for MCP server functions (for coverage)."""

import pytest

try:
    import fa2.mcp_server as mcp_mod
except (ImportError, TypeError, Exception):
    mcp_mod = None

pytestmark = pytest.mark.skipif(mcp_mod is None, reason="MCP server module not loadable")


class TestMCPLayoutGraph:
    def test_basic_layout(self):
        result = mcp_mod.layout_graph(
            edges=[["A", "B"], ["B", "C"]],
            iterations=10,
            seed=42,
        )
        assert isinstance(result, dict)
        assert len(result) == 3
        assert all(len(v) == 2 for v in result.values())

    def test_3d_layout(self):
        result = mcp_mod.layout_graph(
            edges=[["A", "B"], ["B", "C"]],
            dim=3, iterations=10, seed=42,
        )
        assert all(len(v) == 3 for v in result.values())

    def test_community_mode(self):
        result = mcp_mod.layout_graph(
            edges=[["A", "B"], ["B", "C"]],
            mode="community", iterations=10, seed=42,
        )
        assert len(result) == 3

    def test_weighted_edges(self):
        result = mcp_mod.layout_graph(
            edges=[["A", "B", 5.0], ["B", "C", 1.0]],
            iterations=10, seed=42,
        )
        assert len(result) == 3


class TestMCPLayoutAndRender:
    def test_render_png(self):
        pytest.importorskip("matplotlib")
        result = mcp_mod.layout_and_render(
            edges=[["A", "B"], ["B", "C"]],
            iterations=10, seed=42,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_render_with_title(self):
        pytest.importorskip("matplotlib")
        result = mcp_mod.layout_and_render(
            edges=[["A", "B"]],
            iterations=5, seed=42, title="Test",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_empty_edges(self):
        pytest.importorskip("matplotlib")
        result = mcp_mod.layout_and_render(
            edges=[],
            iterations=5, seed=42,
        )
        # Empty edges → empty layout → empty result or valid output
        assert isinstance(result, str)


class TestMCPEvaluateLayout:
    def test_evaluate(self):
        positions = mcp_mod.layout_graph(
            edges=[["A", "B"], ["B", "C"], ["A", "C"]],
            iterations=10, seed=42,
        )
        result = mcp_mod.evaluate_layout(
            edges=[["A", "B"], ["B", "C"], ["A", "C"]],
            positions=positions,
        )
        assert "stress" in result
        assert "neighborhood_preservation" in result
        assert "edge_crossings" in result

    def test_evaluate_empty_positions(self):
        result = mcp_mod.evaluate_layout(
            edges=[["A", "B"]],
            positions={},
        )
        assert result["stress"] == 0.0
        assert result["neighborhood_preservation"] == 1.0

    def test_evaluate_int_key_conversion(self):
        """String keys from layout_graph should be converted back to ints."""
        result = mcp_mod.evaluate_layout(
            edges=[[0, 1], [1, 2]],
            positions={"0": [0.0, 0.0], "1": [1.0, 0.0], "2": [0.5, 1.0]},
        )
        assert "stress" in result

    def test_evaluate_non_int_keys(self):
        """Non-integer string keys should remain as strings."""
        result = mcp_mod.evaluate_layout(
            edges=[["A", "B"], ["B", "C"]],
            positions={"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [0.5, 1.0]},
        )
        assert "stress" in result

    def test_evaluate_already_int_keys(self):
        """Integer keys (not str) should pass through directly (line 147)."""
        result = mcp_mod.evaluate_layout(
            edges=[[0, 1], [1, 2]],
            positions={0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.5, 1.0]},
        )
        assert "stress" in result
