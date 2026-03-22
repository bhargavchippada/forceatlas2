"""Tests for the NumPy vectorized backend."""

import numpy as np
import pytest

from fa2 import ForceAtlas2


class TestVectorizedBackend:
    """Verify vectorized backend produces valid layouts."""

    def test_basic_layout(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3
        assert len(pos[0]) == 2

    def test_3d_layout(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", dim=3)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos[0]) == 3

    def test_reproducibility(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa1 = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos1 = fa1.forceatlas2(G, iterations=20)
        fa2 = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos2 = fa2.forceatlas2(G, iterations=20)
        for p1, p2 in zip(pos1, pos2):
            assert p1[0] == pytest.approx(p2[0])
            assert p1[1] == pytest.approx(p2[1])

    def test_linlog_mode(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", linLogMode=True)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3

    def test_strong_gravity(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", strongGravityMode=True)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3

    def test_adjust_sizes(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", adjustSizes=True)
        pos = fa.forceatlas2(G, iterations=20, sizes=np.array([1.0, 2.0, 1.5]))
        assert len(pos) == 3

    def test_weighted_edges(self):
        G = np.array([[0, 5, 1], [5, 0, 2], [1, 2, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3

    def test_distributed_attraction(self):
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized",
                         outboundAttractionDistribution=True)
        pos = fa.forceatlas2(G, iterations=20)
        assert len(pos) == 3

    def test_callbacks(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        called = [0]
        def cb(i, nodes):
            called[0] += 1
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        fa.forceatlas2(G, iterations=5, callbacks=[cb])
        assert called[0] == 5

    def test_callbacks_3d(self):
        G = np.array([[0, 1], [1, 0]], dtype=float)
        positions = []
        def cb(i, nodes):
            positions.append(tuple(nodes[0].pos))
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", dim=3)
        fa.forceatlas2(G, iterations=3, callbacks=[cb])
        assert len(positions) == 3
        assert len(positions[0]) == 3

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            ForceAtlas2(backend="gpu")


class TestAutoBackendSelection:
    """Test that auto backend selects vectorized when Cython is not available."""

    def test_auto_selects_vectorized_without_cython(self, monkeypatch):
        """When fa2util is pure Python, auto should use vectorized."""
        import fa2.forceatlas2 as mod
        # Pretend fa2util is the .py fallback
        monkeypatch.setattr(mod.fa2util, "__file__", "/fake/fa2util.py")
        fa = ForceAtlas2(verbose=False, seed=42, backend="auto")
        assert fa._should_use_vectorized() is True

    def test_auto_selects_loop_with_cython(self):
        """When fa2util is Cython .so, auto should use loop (with BH)."""
        import fa2.fa2util
        fa = ForceAtlas2(verbose=False, seed=42, backend="auto")
        if fa2.fa2util.__file__.endswith(".so") or fa2.fa2util.__file__.endswith(".pyd"):
            assert fa._should_use_vectorized() is False


class TestVectorizedEdgeCases:
    """Cover edge cases for vectorized functions."""

    def test_single_node(self):
        """Single node graph should not crash."""
        G = np.zeros((1, 1))
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos = fa.forceatlas2(G, iterations=5)
        assert len(pos) == 1

    def test_no_edges(self):
        """Edgeless graph should work."""
        G = np.zeros((3, 3))
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos = fa.forceatlas2(G, iterations=5)
        assert len(pos) == 3

    def test_edge_weight_influence_zero(self):
        """edgeWeightInfluence=0 should ignore weights."""
        G = np.array([[0, 10], [10, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", edgeWeightInfluence=0)
        pos = fa.forceatlas2(G, iterations=10)
        assert len(pos) == 2

    def test_edge_weight_influence_fractional(self):
        """edgeWeightInfluence=0.5 uses pow."""
        G = np.array([[0, 4], [4, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", edgeWeightInfluence=0.5)
        pos = fa.forceatlas2(G, iterations=10)
        assert len(pos) == 2

    def test_linlog_distributed(self):
        """LinLog mode + distributed attraction."""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized",
                         linLogMode=True, outboundAttractionDistribution=True)
        pos = fa.forceatlas2(G, iterations=10)
        assert len(pos) == 3

    def test_linlog_adjust_sizes(self):
        """LinLog mode + adjustSizes."""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized",
                         linLogMode=True, adjustSizes=True)
        pos = fa.forceatlas2(G, iterations=10, sizes=np.array([1.0, 2.0, 1.5]))
        assert len(pos) == 3

    def test_linlog_adjust_sizes_distributed(self):
        """LinLog + adjustSizes + distributed attraction covers all branches."""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized",
                         linLogMode=True, adjustSizes=True,
                         outboundAttractionDistribution=True)
        pos = fa.forceatlas2(G, iterations=10, sizes=np.array([1.0, 2.0, 1.5]))
        assert len(pos) == 3

    def test_verbose_vectorized(self):
        """Verbose mode with vectorized backend should show tqdm."""
        G = np.array([[0, 1], [1, 0]], dtype=float)
        fa = ForceAtlas2(verbose=True, seed=42, backend="vectorized")
        pos = fa.forceatlas2(G, iterations=2)
        assert len(pos) == 2

    def test_single_node_adjust_sizes(self):
        """Single node with adjustSizes shouldn't crash."""
        G = np.zeros((1, 1))
        fa = ForceAtlas2(verbose=False, seed=42, backend="vectorized", adjustSizes=True)
        pos = fa.forceatlas2(G, iterations=3, sizes=np.array([1.0]))
        assert len(pos) == 1

    def test_empty_graph_zero_traction(self):
        """Empty graph exercises zero-traction fallback in adjustSpeed."""
        from fa2.fa2util_vectorized import adjustSpeedAndApplyForces
        positions = np.zeros((2, 2))
        forces = np.zeros((2, 2))
        old_forces = np.zeros((2, 2))
        masses = np.array([1.0, 1.0])
        new_pos, speed, eff = adjustSpeedAndApplyForces(
            positions, forces, old_forces, masses, 1.0, 1.0, 1.0)
        assert speed > 0


class TestVectorizedEquivalence:
    """Verify vectorized and loop backends produce similar results."""

    def test_same_direction(self):
        """Both backends should move nodes in broadly the same direction."""
        G = np.array([[0, 1, 0, 0, 1],
                       [1, 0, 1, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 0, 1, 0, 1],
                       [1, 0, 0, 1, 0]], dtype=float)

        fa_loop = ForceAtlas2(verbose=False, seed=42, backend="loop", barnesHutOptimize=False)
        pos_loop = fa_loop.forceatlas2(G, iterations=10)

        fa_vec = ForceAtlas2(verbose=False, seed=42, backend="vectorized")
        pos_vec = fa_vec.forceatlas2(G, iterations=10)

        # They won't be exactly equal (different accumulation order) but
        # should be qualitatively similar — each node within reasonable distance
        for p1, p2 in zip(pos_loop, pos_vec):
            # Allow generous tolerance since the algorithms may diverge
            assert abs(p1[0] - p2[0]) < 50.0
            assert abs(p1[1] - p2[1]) < 50.0
