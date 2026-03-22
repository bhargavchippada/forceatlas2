"""Unit tests for fa2util force computation functions."""



from fa2 import fa2util


class TestNode:
    def test_default_values(self):
        n = fa2util.Node()
        assert n.mass == 0.0
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.dx == 0.0
        assert n.dy == 0.0
        assert n.old_dx == 0.0
        assert n.old_dy == 0.0

    def test_set_values(self):
        n = fa2util.Node()
        n.mass = 5.0
        n.x = 1.0
        n.y = 2.0
        assert n.mass == 5.0
        assert n.x == 1.0
        assert n.y == 2.0


class TestEdge:
    def test_default_values(self):
        e = fa2util.Edge()
        assert e.node1 == -1
        assert e.node2 == -1
        assert e.weight == 0.0

    def test_set_values(self):
        e = fa2util.Edge()
        e.node1 = 0
        e.node2 = 1
        e.weight = 2.5
        assert e.node1 == 0
        assert e.node2 == 1
        assert e.weight == 2.5


def _make_node(x, y, mass=1.0):
    n = fa2util.Node()
    n.x = x
    n.y = y
    n.mass = mass
    return n


def _make_edge(n1, n2, weight=1.0):
    e = fa2util.Edge()
    e.node1 = n1
    e.node2 = n2
    e.weight = weight
    return e


class TestRepulsionViaApply:
    """Test repulsion behavior through the public apply_repulsion function."""

    def test_repulsion_pushes_apart(self):
        """Nodes at different positions should be pushed apart."""
        n1 = _make_node(0, 0)
        n2 = _make_node(1, 0)
        fa2util.apply_repulsion([n1, n2], coefficient=1.0)
        # n1 should be pushed left (negative dx)
        assert n1.dx < 0
        # n2 should be pushed right (positive dx)
        assert n2.dx > 0

    def test_repulsion_symmetric(self):
        """Repulsion forces should be equal and opposite."""
        n1 = _make_node(0, 0)
        n2 = _make_node(1, 0)
        fa2util.apply_repulsion([n1, n2], coefficient=1.0)
        assert abs(n1.dx + n2.dx) < 1e-10
        assert abs(n1.dy + n2.dy) < 1e-10

    def test_coincident_nodes_no_crash(self):
        """Nodes at the same position should not crash (division by zero guard)."""
        n1 = _make_node(0, 0)
        n2 = _make_node(0, 0)
        fa2util.apply_repulsion([n1, n2], coefficient=1.0)
        assert n1.dx == 0.0
        assert n1.dy == 0.0

    def test_higher_mass_stronger_force(self):
        """Higher mass nodes should experience stronger repulsion."""
        n1 = _make_node(0, 0, mass=1.0)
        n2 = _make_node(1, 0, mass=1.0)
        fa2util.apply_repulsion([n1, n2], coefficient=1.0)
        force_low = abs(n1.dx)

        n3 = _make_node(0, 0, mass=5.0)
        n4 = _make_node(1, 0, mass=5.0)
        fa2util.apply_repulsion([n3, n4], coefficient=1.0)
        force_high = abs(n3.dx)

        assert force_high > force_low


class TestGravityViaApply:
    """Test gravity behavior through the public apply_gravity function."""

    def test_gravity_pulls_toward_center(self):
        """Gravity should pull nodes toward the origin."""
        n = _make_node(5.0, 3.0)
        fa2util.apply_gravity([n], gravity=1.0, scalingRatio=2.0, useStrongGravity=False)
        assert n.dx < 0  # Pulled left toward 0
        assert n.dy < 0  # Pulled down toward 0

    def test_node_at_origin_no_force(self):
        """Node at origin should experience no gravity."""
        n = _make_node(0, 0)
        fa2util.apply_gravity([n], gravity=1.0, scalingRatio=2.0, useStrongGravity=False)
        assert n.dx == 0.0
        assert n.dy == 0.0

    def test_strong_gravity_pulls_toward_center(self):
        n = _make_node(5.0, 3.0)
        fa2util.apply_gravity([n], gravity=1.0, scalingRatio=2.0, useStrongGravity=True)
        assert n.dx < 0
        assert n.dy < 0

    def test_strong_gravity_on_axis(self):
        """Node on the X axis (y=0) should still receive strong gravity force."""
        n = _make_node(5.0, 0.0)
        fa2util.apply_gravity([n], gravity=1.0, scalingRatio=2.0, useStrongGravity=True)
        assert n.dx < 0  # Should be pulled toward center

    def test_strong_gravity_at_origin_no_force(self):
        """Node at origin should receive no strong gravity."""
        n = _make_node(0.0, 0.0)
        fa2util.apply_gravity([n], gravity=1.0, scalingRatio=2.0, useStrongGravity=True)
        assert n.dx == 0.0
        assert n.dy == 0.0


class TestLinAttraction:
    def test_attraction_pulls_together(self):
        """Connected nodes should be pulled together."""
        n1 = _make_node(0, 0)
        n2 = _make_node(1, 0)
        fa2util.linAttraction(n1, n2, 1.0, False, coefficient=1.0)
        # n1 at x=0, n2 at x=1: n1 should be pulled right, n2 left
        assert n1.dx > 0
        assert n2.dx < 0

    def test_attraction_symmetric(self):
        """Attraction forces should be equal and opposite."""
        n1 = _make_node(0, 0)
        n2 = _make_node(1, 0)
        fa2util.linAttraction(n1, n2, 1.0, False, coefficient=1.0)
        assert abs(n1.dx + n2.dx) < 1e-10

    def test_distributed_attraction(self):
        """With distributed attraction, force should be divided by mass."""
        n1 = _make_node(0, 0, mass=2.0)
        n2 = _make_node(1, 0, mass=2.0)
        fa2util.linAttraction(n1, n2, 1.0, True, coefficient=1.0)

        n3 = _make_node(0, 0, mass=2.0)
        n4 = _make_node(1, 0, mass=2.0)
        fa2util.linAttraction(n3, n4, 1.0, False, coefficient=1.0)

        # Distributed should be weaker (divided by mass)
        assert abs(n1.dx) < abs(n3.dx)


class TestLogAttraction:
    def test_log_attraction_pulls_together(self):
        n1 = _make_node(0, 0)
        n2 = _make_node(1, 0)
        fa2util.logAttraction(n1, n2, 1.0, False, coefficient=1.0)
        assert n1.dx > 0
        assert n2.dx < 0

    def test_log_attraction_uses_log_formula(self):
        """Force should equal log(1+d), not d (linear) or 1/d (old wrong formula)."""
        from math import log
        n1 = _make_node(0, 0)
        n2 = _make_node(10, 0)
        fa2util.logAttraction(n1, n2, 1.0, False, coefficient=1.0)
        # dx should be log(1+10) = log(11) ≈ 2.3979, not 10 (linear) or 1 (old wrong)
        expected = log(11)
        assert abs(n1.dx - expected) < 1e-6

    def test_log_attraction_distributed(self):
        """Distributed mode divides force by source node mass."""
        n1 = _make_node(0, 0, mass=2.0)
        n2 = _make_node(5, 0, mass=2.0)
        fa2util.logAttraction(n1, n2, 1.0, True, coefficient=1.0)

        n3 = _make_node(0, 0, mass=2.0)
        n4 = _make_node(5, 0, mass=2.0)
        fa2util.logAttraction(n3, n4, 1.0, False, coefficient=1.0)

        # Distributed should be weaker (divided by n1.mass=2)
        assert abs(n1.dx) < abs(n3.dx)
        assert abs(n1.dx * 2 - n3.dx) < 1e-10  # Exactly half

    def test_coincident_nodes_no_crash(self):
        n1 = _make_node(0, 0)
        n2 = _make_node(0, 0)
        fa2util.logAttraction(n1, n2, 1.0, False, coefficient=1.0)
        assert n1.dx == 0.0


class TestApplyRepulsion:
    def test_all_nodes_get_forces(self):
        nodes = [_make_node(i, i) for i in range(4)]
        fa2util.apply_repulsion(nodes, coefficient=1.0)
        for n in nodes:
            assert n.dx != 0.0 or n.dy != 0.0


class TestApplyGravity:
    def test_standard_gravity(self):
        nodes = [_make_node(5, 5), _make_node(-3, -3)]
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=2.0, useStrongGravity=False)
        assert nodes[0].dx < 0  # Pulled toward center
        assert nodes[1].dx > 0

    def test_strong_gravity(self):
        nodes = [_make_node(5, 5), _make_node(-3, -3)]
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=2.0, useStrongGravity=True)
        assert nodes[0].dx < 0
        assert nodes[1].dx > 0


class TestApplyAttraction:
    def test_standard_mode(self):
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 1.0, linLogMode=False)
        assert nodes[0].dx > 0  # Pulled toward node 1

    def test_linlog_mode(self):
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 1.0, linLogMode=True)
        assert nodes[0].dx > 0

    def test_edge_weight_influence_zero(self):
        """With edgeWeightInfluence=0, all edges should have equal effect."""
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1, weight=100.0)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 0.0, linLogMode=False)
        dx_heavy = nodes[0].dx

        nodes2 = [_make_node(0, 0), _make_node(5, 0)]
        edges2 = [_make_edge(0, 1, weight=0.01)]
        fa2util.apply_attraction(nodes2, edges2, False, 1.0, 0.0, linLogMode=False)
        dx_light = nodes2[0].dx

        assert abs(dx_heavy - dx_light) < 1e-10

    def test_edge_weight_influence_zero_linlog(self):
        """With edgeWeightInfluence=0 + linLogMode, weights should be ignored too."""
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1, weight=100.0)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 0.0, linLogMode=True)
        dx_heavy = nodes[0].dx

        nodes2 = [_make_node(0, 0), _make_node(5, 0)]
        edges2 = [_make_edge(0, 1, weight=0.01)]
        fa2util.apply_attraction(nodes2, edges2, False, 1.0, 0.0, linLogMode=True)
        dx_light = nodes2[0].dx

        assert abs(dx_heavy - dx_light) < 1e-10

    def test_edge_weight_influence_fractional(self):
        """edgeWeightInfluence between 0 and 1 uses pow(weight, influence)."""
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1, weight=4.0)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 0.5, linLogMode=False)
        # pow(4, 0.5) = 2.0, so force should be 2x the unit-weight case
        dx_pow = nodes[0].dx

        nodes2 = [_make_node(0, 0), _make_node(5, 0)]
        edges2 = [_make_edge(0, 1, weight=1.0)]
        fa2util.apply_attraction(nodes2, edges2, False, 1.0, 1.0, linLogMode=False)
        dx_unit = nodes2[0].dx

        assert abs(dx_pow - 2 * dx_unit) < 1e-10

    def test_edge_weight_influence_fractional_linlog(self):
        """pow branch with linLogMode=True."""
        nodes = [_make_node(0, 0), _make_node(5, 0)]
        edges = [_make_edge(0, 1, weight=4.0)]
        fa2util.apply_attraction(nodes, edges, False, 1.0, 0.5, linLogMode=True)
        assert nodes[0].dx > 0  # Should pull toward node 1


class TestRegion:
    def test_creation(self):
        nodes = [_make_node(0, 0, mass=1.0), _make_node(1, 1, mass=1.0)]
        r = fa2util.Region(nodes)
        assert r.mass == 2.0
        assert abs(r.massCenterX - 0.5) < 1e-10
        assert abs(r.massCenterY - 0.5) < 1e-10

    def test_single_node_region(self):
        nodes = [_make_node(3, 4, mass=2.0)]
        r = fa2util.Region(nodes)
        assert r.mass == 2.0
        assert r.massCenterX == 3.0
        assert r.massCenterY == 4.0
        assert r.size == 0.0

    def test_build_subregions(self):
        nodes = [_make_node(i, j, mass=1.0) for i in range(3) for j in range(3)]
        r = fa2util.Region(nodes)
        r.buildSubRegions()
        assert len(r.subregions) > 0

    def test_apply_force_on_nodes(self):
        nodes = [_make_node(i, i, mass=1.0) for i in range(5)]
        r = fa2util.Region(nodes)
        r.buildSubRegions()
        r.applyForceOnNodes(nodes, theta=1.2, coefficient=2.0)
        # Most nodes should have some force applied (center node may cancel)
        nonzero = sum(1 for n in nodes if n.dx != 0.0 or n.dy != 0.0)
        assert nonzero >= len(nodes) - 1


class TestAdjustSpeedAndApplyForces:
    def test_basic_operation(self):
        nodes = [_make_node(i, i, mass=1.0) for i in range(3)]
        # Set some forces
        for i, n in enumerate(nodes):
            n.dx = float(i)
            n.dy = float(i)
            n.old_dx = float(i) * 0.5
            n.old_dy = float(i) * 0.5

        result = fa2util.adjustSpeedAndApplyForces(nodes, speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0)
        assert 'speed' in result
        assert 'speedEfficiency' in result
        assert result['speed'] > 0

    def test_positions_change(self):
        nodes = [_make_node(0, 0, mass=1.0), _make_node(5, 5, mass=1.0)]
        nodes[0].dx = 1.0
        nodes[0].dy = 1.0
        nodes[1].dx = -1.0
        nodes[1].dy = -1.0

        old_x0 = nodes[0].x
        fa2util.adjustSpeedAndApplyForces(nodes, speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0)
        assert nodes[0].x != old_x0

    def test_empty_nodes_no_crash(self):
        """Should handle empty node list without crashing. speed increases by maxRise."""
        result = fa2util.adjustSpeedAndApplyForces([], speed=1.0, speedEfficiency=1.0, jitterTolerance=1.0)
        # With no nodes: totalSwinging=0, targetSpeed=inf, maxRise=0.5
        # speed = 1.0 + min(inf - 1.0, 0.5 * 1.0) = 1.5
        assert result['speed'] == 1.5
        # speedEfficiency *= 1.3 (speed < 1000, no swinging)
        assert result['speedEfficiency'] == 1.3
