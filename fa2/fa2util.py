# CPU-intensive ForceAtlas2 routines. Pure Python fallback for when the
# Cython extension (fa2util.pyx -> fa2util.so) is not compiled.
#
# IF YOU MODIFY THIS FILE, YOU MUST ALSO MODIFY fa2util.pyx TO KEEP
# BOTH IMPLEMENTATIONS IN SYNC!
#
# Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

from math import log, sqrt


class Node:
    """Graph node with N-dimensional position and force vectors."""

    __slots__ = ('mass', 'dim', 'pos', 'force', 'old_force', 'size')

    def __init__(self, dim=2):
        self.mass = 0.0
        self.dim = dim
        self.pos = [0.0] * dim
        self.force = [0.0] * dim
        self.old_force = [0.0] * dim
        self.size = 0.0  # For adjustSizes (node radius)

    # Backward-compatible properties for dim=2 code paths and tests.
    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, v):
        self.pos[0] = v

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, v):
        self.pos[1] = v

    @property
    def dx(self):
        return self.force[0]

    @dx.setter
    def dx(self, v):
        self.force[0] = v

    @property
    def dy(self):
        return self.force[1]

    @dy.setter
    def dy(self, v):
        self.force[1] = v

    @property
    def old_dx(self):
        return self.old_force[0]

    @old_dx.setter
    def old_dx(self, v):
        self.old_force[0] = v

    @property
    def old_dy(self):
        return self.old_force[1]

    @old_dy.setter
    def old_dy(self, v):
        self.old_force[1] = v


class Edge:
    """Undirected graph edge with source/target node indices and weight."""

    __slots__ = ('node1', 'node2', 'weight')

    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


def linRepulsion(n1, n2, coefficient=0):
    """Apply linear repulsion between two nodes.

    Force: ``F = coefficient * m1 * m2 / distance``, applied as a factor
    divided by distance² (the ``distance²`` in code is the factor/distance
    optimization pattern from Gephi).

    Parameters
    ----------
    n1, n2 : Node
        The two nodes to repel.
    coefficient : float
        Repulsion coefficient (typically ``scalingRatio``).
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    distance2 = sum(v * v for v in dist)

    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        for d in range(dim):
            n1.force[d] += dist[d] * factor
            n2.force[d] -= dist[d] * factor


def linRepulsion_region(n, r, coefficient=0):
    """Apply one-sided repulsion between a node and a Barnes-Hut region.

    Parameters
    ----------
    n : Node
        The node being repelled.
    r : Region
        The Barnes-Hut region (aggregated mass center).
    coefficient : float
        Repulsion coefficient.
    """
    dim = n.dim
    dist = [n.pos[d] - r.massCenter[d] for d in range(dim)]
    distance2 = sum(v * v for v in dist)

    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        for d in range(dim):
            n.force[d] += dist[d] * factor


def linRepulsion_antiCollision(n1, n2, coefficient=0):
    """Anti-collision repulsion: subtract node sizes from distance.

    When nodes overlap (``d_adj < 0``), applies a strong constant push
    (``100x`` normal force). Otherwise uses size-adjusted distance.
    Reference: Gephi ``ForceFactory.java`` ``linRepulsion_antiCollision``.

    Parameters
    ----------
    n1, n2 : Node
        The two nodes. Must have ``size`` attribute set.
    coefficient : float
        Repulsion coefficient.
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    euclidean = sqrt(sum(v * v for v in dist))
    distance = euclidean - n1.size - n2.size

    if euclidean > 0:
        if distance <= 0:
            # Overlap or touching: strong constant repulsive force
            factor = 100.0 * coefficient * n1.mass * n2.mass
        else:
            # No overlap: standard repulsion with size-adjusted distance
            factor = coefficient * n1.mass * n2.mass / (distance * distance)
        # Direction is still based on euclidean dist vector (unit = dist/euclidean)
        for d in range(dim):
            n1.force[d] += dist[d] * factor
            n2.force[d] -= dist[d] * factor


def linRepulsion_antiCollision_region(n, r, coefficient=0):
    """Anti-collision repulsion between a node and a Barnes-Hut region.

    One-sided version for Barnes-Hut traversal. The region has no size.

    Parameters
    ----------
    n : Node
        The node being repelled.
    r : Region
        The Barnes-Hut region.
    coefficient : float
        Repulsion coefficient.
    """
    dim = n.dim
    dist = [n.pos[d] - r.massCenter[d] for d in range(dim)]
    euclidean = sqrt(sum(v * v for v in dist))
    distance = euclidean - n.size

    if euclidean > 0:
        if distance <= 0:
            factor = 100.0 * coefficient * n.mass * r.mass
        else:
            factor = coefficient * n.mass * r.mass / (distance * distance)
        for d in range(dim):
            n.force[d] += dist[d] * factor


def linGravity(n, g):
    """Apply linear gravity toward the origin.

    Force: ``F = mass * g / distance``. Weakens with distance.

    Parameters
    ----------
    n : Node
        The node to attract toward origin.
    g : float
        Gravity strength.
    """
    dim = n.dim
    distance2 = sum(n.pos[d] * n.pos[d] for d in range(dim))

    if distance2 > 0:
        distance = sqrt(distance2)
        factor = n.mass * g / distance
        for d in range(dim):
            n.force[d] -= n.pos[d] * factor


def strongGravity(n, g, coefficient=0):
    """Apply strong (distance-independent) gravity toward origin.

    Force: ``F = coefficient * mass * g``. Does not weaken with distance.

    Parameters
    ----------
    n : Node
        The node to attract toward origin.
    g : float
        Gravity strength.
    coefficient : float
        Scaling coefficient (typically ``scalingRatio``).
    """
    if any(n.pos[d] != 0 for d in range(n.dim)):
        factor = coefficient * n.mass * g
        for d in range(n.dim):
            n.force[d] -= n.pos[d] * factor


def linAttraction(n1, n2, e, distributedAttraction, coefficient=0):
    """Apply linear attraction between connected nodes.

    Force: ``F = -coefficient * edgeWeight * distance``.

    Parameters
    ----------
    n1, n2 : Node
        Source and target nodes.
    e : float
        Edge weight (after ``edgeWeightInfluence`` exponent).
    distributedAttraction : bool
        If True, divide by source node mass (dissuade hubs).
    coefficient : float
        Attraction coefficient.
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    for d in range(dim):
        n1.force[d] += dist[d] * factor
        n2.force[d] -= dist[d] * factor


def logAttraction(n1, n2, e, distributedAttraction, coefficient=0):
    """Apply logarithmic attraction for LinLog mode.

    Force: ``F = -coefficient * edgeWeight * log(1 + distance)``.
    Reference: Jacomy et al. 2014 Formula 3; Gephi ``ForceFactory.java``.

    Parameters
    ----------
    n1, n2 : Node
        Source and target nodes.
    e : float
        Edge weight.
    distributedAttraction : bool
        If True, divide by source node mass.
    coefficient : float
        Attraction coefficient.
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    distance = sqrt(sum(v * v for v in dist))

    if distance > 0:
        log_factor = log(1 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        for d in range(dim):
            n1.force[d] += dist[d] * factor
            n2.force[d] -= dist[d] * factor


def linAttraction_antiCollision(n1, n2, e, distributedAttraction, coefficient=0):
    """Linear attraction with anti-collision: zero force when overlapping.

    Reference: Gephi ``ForceFactory.java`` ``linAttraction_antiCollision``.

    Parameters
    ----------
    n1, n2 : Node
        Source and target nodes.
    e : float
        Edge weight.
    distributedAttraction : bool
        If True, divide by source node mass.
    coefficient : float
        Attraction coefficient.
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    euclidean = sqrt(sum(v * v for v in dist))
    distance = euclidean - n1.size - n2.size

    if distance > 0:
        if not distributedAttraction:
            factor = -coefficient * e
        else:
            factor = -coefficient * e / n1.mass
        for d in range(dim):
            n1.force[d] += dist[d] * factor
            n2.force[d] -= dist[d] * factor


def logAttraction_antiCollision(n1, n2, e, distributedAttraction, coefficient=0):
    """Logarithmic attraction with anti-collision: zero force when overlapping.

    Parameters
    ----------
    n1, n2 : Node
        Source and target nodes.
    e : float
        Edge weight.
    distributedAttraction : bool
        If True, divide by source node mass.
    coefficient : float
        Attraction coefficient.
    """
    dim = n1.dim
    dist = [n1.pos[d] - n2.pos[d] for d in range(dim)]
    euclidean = sqrt(sum(v * v for v in dist))
    distance = euclidean - n1.size - n2.size

    if distance > 0:
        log_factor = log(1 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        for d in range(dim):
            n1.force[d] += dist[d] * factor
            n2.force[d] -= dist[d] * factor


def apply_repulsion(nodes, coefficient, adjustSizes=False):
    """Apply repulsion forces between all pairs of nodes.

    Parameters
    ----------
    nodes : list of Node
        All graph nodes.
    coefficient : float
        Repulsion coefficient (``scalingRatio``).
    adjustSizes : bool
        Use anti-collision repulsion.
    """
    repulse_fn = linRepulsion_antiCollision if adjustSizes else linRepulsion
    for i, n1 in enumerate(nodes):
        for n2 in nodes[:i]:
            repulse_fn(n1, n2, coefficient)


def apply_gravity(nodes, gravity, scalingRatio, useStrongGravity=False):
    """Apply gravity forces to all nodes.

    Parameters
    ----------
    nodes : list of Node
    gravity : float
        Gravity strength.
    scalingRatio : float
        Used as coefficient for strong gravity.
    useStrongGravity : bool
        Use distance-independent strong gravity.
    """
    if not useStrongGravity:
        for n in nodes:
            linGravity(n, gravity)
    else:
        for n in nodes:
            strongGravity(n, gravity, scalingRatio)


def apply_attraction(nodes, edges, distributedAttraction, coefficient, edgeWeightInfluence, linLogMode=False,
                     adjustSizes=False):
    """Apply attraction forces along all edges.

    Selects the appropriate attraction function based on ``linLogMode``
    and ``adjustSizes``. Optimizes for ``edgeWeightInfluence`` of 0 or 1
    to avoid slow ``pow()`` calls.

    Parameters
    ----------
    nodes : list of Node
    edges : list of Edge
    distributedAttraction : bool
        Divide by source mass (dissuade hubs).
    coefficient : float
        Attraction coefficient.
    edgeWeightInfluence : float
        Exponent applied to edge weights.
    linLogMode : bool
        Use logarithmic attraction.
    adjustSizes : bool
        Use anti-collision attraction.
    """
    if adjustSizes:
        attract_fn = logAttraction_antiCollision if linLogMode else linAttraction_antiCollision
    else:
        attract_fn = logAttraction if linLogMode else linAttraction
    if edgeWeightInfluence == 0:
        for edge in edges:
            attract_fn(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            attract_fn(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    else:
        for edge in edges:
            attract_fn(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                       distributedAttraction, coefficient)


class Region:
    """Barnes-Hut spatial tree node.

    Generalizes the quadtree to ``2^dim`` partitioning for N-dimensional
    layouts. Each region stores aggregated mass and center of mass for
    its child nodes, enabling O(n log n) repulsion approximation.

    Parameters
    ----------
    nodes : list of Node
        Nodes contained in this region.
    """

    def __init__(self, nodes):
        self.mass = 0.0
        self.massCenter = []  # N-dimensional center of mass
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self._dim = nodes[0].dim if nodes else 2
        self.massCenter = [0.0] * self._dim
        self.updateMassAndGeometry()

    def updateMassAndGeometry(self):
        if len(self.nodes) == 1:
            n = self.nodes[0]
            self.mass = n.mass
            self.massCenter = list(n.pos)
            self.size = 0.0
        elif len(self.nodes) > 1:
            dim = self._dim
            self.mass = 0.0
            massSum = [0.0] * dim
            for n in self.nodes:
                self.mass += n.mass
                for d in range(dim):
                    massSum[d] += n.pos[d] * n.mass
            if self.mass > 0:
                self.massCenter = [s / self.mass for s in massSum]

            self.size = 0.0
            for n in self.nodes:
                distance = sqrt(sum((n.pos[d] - self.massCenter[d]) ** 2 for d in range(dim)))
                self.size = max(self.size, 2 * distance)

    def buildSubRegions(self):
        if len(self.nodes) > 1:
            dim = self._dim
            # Partition nodes into 2^dim buckets using bitmask on each dimension
            num_buckets = 1 << dim  # 2^dim
            buckets = [[] for _ in range(num_buckets)]
            for n in self.nodes:
                bucket = 0
                for d in range(dim):
                    if n.pos[d] >= self.massCenter[d]:
                        bucket |= (1 << d)
                buckets[bucket].append(n)

            for bucket_nodes in buckets:
                if len(bucket_nodes) > 0:
                    if len(bucket_nodes) < len(self.nodes):
                        subregion = Region(bucket_nodes)
                        self.subregions.append(subregion)
                    else:
                        # All nodes in one bucket — split into individual regions
                        for n in bucket_nodes:
                            subregion = Region([n])
                            self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()

    def applyForce(self, n, theta, coefficient=0, adjustSizes=False):
        if len(self.nodes) == 0:
            return
        repulse_fn = linRepulsion_antiCollision_region if adjustSizes else linRepulsion_region
        if len(self.nodes) < 2:
            if self.nodes[0] is not n:
                repulse_fn(n, self, coefficient)
        else:
            dim = self._dim
            distance = sqrt(sum((n.pos[d] - self.massCenter[d]) ** 2 for d in range(dim)))
            if distance * theta > self.size:
                repulse_fn(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce(n, theta, coefficient, adjustSizes)

    def applyForceOnNodes(self, nodes, theta, coefficient=0, adjustSizes=False):
        for n in nodes:
            self.applyForce(n, theta, coefficient, adjustSizes)


def adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, jitterTolerance, adjustSizes=False):
    """Adjust simulation speed and apply accumulated forces to node positions.

    Uses swing/traction measurement to adaptively control speed.
    High swing (oscillation) reduces speed; high traction (consistent
    movement) increases it.

    Parameters
    ----------
    nodes : list of Node
        All graph nodes with accumulated forces.
    speed : float
        Current simulation speed.
    speedEfficiency : float
        Current speed efficiency factor.
    jitterTolerance : float
        How much swinging is tolerated.
    adjustSizes : bool
        Cap per-node speed based on node size.

    Returns
    -------
    dict
        ``{'speed': float, 'speedEfficiency': float}``
    """
    totalSwinging = 0.0
    totalEffectiveTraction = 0.0
    for n in nodes:
        dim = n.dim
        swinging = sqrt(sum((n.old_force[d] - n.force[d]) ** 2 for d in range(dim)))
        totalSwinging += n.mass * swinging
        totalEffectiveTraction += .5 * n.mass * sqrt(
            sum((n.old_force[d] + n.force[d]) ** 2 for d in range(dim)))

    estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
    minJT = sqrt(estimatedOptimalJitterTolerance)
    maxJT = 10
    if len(nodes) > 0 and totalEffectiveTraction > 0:
        jt = jitterTolerance * max(minJT,
                                   min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                       len(nodes) * len(nodes))))
    else:
        jt = jitterTolerance * minJT

    minSpeedEfficiency = 0.05

    if totalEffectiveTraction > 0 and totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .5
        jt = max(jt, jitterTolerance)

    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .7
    elif speed < 1000:
        speedEfficiency *= 1.3

    maxRise = .5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Apply forces to positions
    for n in nodes:
        dim = n.dim
        swinging = n.mass * sqrt(sum((n.old_force[d] - n.force[d]) ** 2 for d in range(dim)))
        factor = speed / (1.0 + sqrt(speed * swinging))
        # Gephi caps per-node speed when adjustSizes is on to prevent large nodes overshooting
        if adjustSizes and n.size > 0:
            factor = min(factor, 10.0 / n.size)
        for d in range(dim):
            n.pos[d] += n.force[d] * factor

    return {'speed': speed, 'speedEfficiency': speedEfficiency}


# Aliases for compatibility with Cython module which has Node2D/NodeND classes.
# In pure Python, the single Node class handles all dimensions.
Node2D = Node
NodeND = Node


# Warn if running pure Python fallback (no compiled extension)
if __file__.endswith(".py"):
    import warnings

    warnings.warn(
        "Running pure Python fa2util (no compiled extension found). "
        "Rebuild for 10-100x speedup: pip install cython && pip install --no-build-isolation fa2",
        stacklevel=2,
    )
