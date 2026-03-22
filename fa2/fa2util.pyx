# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""CPU-intensive ForceAtlas2 routines optimized with Cython.

Two node types for optimal performance:
- Node2D: scalar fields (x, y, dx, dy, old_dx, old_dy) for C-speed 2D layouts
- NodeND: list fields (pos, force, old_force) for N-dimensional layouts

The Node() factory returns the right type based on dim.

Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
Available under the GPLv3
"""

from libc.math cimport log, sqrt


# ═══════════════════════════════════════════════════════════════════════
# Node classes
# ═══════════════════════════════════════════════════════════════════════

cdef class Node2D:
    """2D graph node with scalar position/force fields (C-speed access)."""
    cdef public double mass
    cdef public double x, y
    cdef public double dx, dy
    cdef public double old_dx, old_dy
    cdef public double size
    cdef public int dim

    def __init__(self):
        self.mass = 0.0
        self.dim = 2
        self.x = 0.0
        self.y = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.size = 0.0

    @property
    def pos(self):
        return [self.x, self.y]

    @pos.setter
    def pos(self, val):
        self.x = <double>val[0]
        self.y = <double>val[1]

    @property
    def force(self):
        return [self.dx, self.dy]

    @force.setter
    def force(self, val):
        self.dx = <double>val[0]
        self.dy = <double>val[1]

    @property
    def old_force(self):
        return [self.old_dx, self.old_dy]

    @old_force.setter
    def old_force(self, val):
        self.old_dx = <double>val[0]
        self.old_dy = <double>val[1]


cdef class NodeND:
    """N-dimensional graph node with list-based position/force fields."""
    cdef public double mass
    cdef public int dim
    cdef public list pos
    cdef public list force
    cdef public list old_force
    cdef public double size

    def __init__(self, int dim=3):
        self.mass = 0.0
        self.dim = dim
        self.pos = [0.0] * dim
        self.force = [0.0] * dim
        self.old_force = [0.0] * dim
        self.size = 0.0

    @property
    def x(self):
        return <double>self.pos[0]
    @x.setter
    def x(self, double v):
        self.pos[0] = v

    @property
    def y(self):
        return <double>self.pos[1]
    @y.setter
    def y(self, double v):
        self.pos[1] = v

    @property
    def dx(self):
        return <double>self.force[0]
    @dx.setter
    def dx(self, double v):
        self.force[0] = v

    @property
    def dy(self):
        return <double>self.force[1]
    @dy.setter
    def dy(self, double v):
        self.force[1] = v

    @property
    def old_dx(self):
        return <double>self.old_force[0]
    @old_dx.setter
    def old_dx(self, double v):
        self.old_force[0] = v

    @property
    def old_dy(self):
        return <double>self.old_force[1]
    @old_dy.setter
    def old_dy(self, double v):
        self.old_force[1] = v


def Node(int dim=2):
    """Factory: returns Node2D for dim=2, NodeND for dim>2."""
    if dim == 2:
        return Node2D()
    return NodeND(dim)


cdef class Edge:
    """Represents a graph edge with source, target, and weight."""
    cdef public int node1, node2
    cdef public double weight

    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# ═══════════════════════════════════════════════════════════════════════
# 2D force functions (scalar fields — C-speed)
# ═══════════════════════════════════════════════════════════════════════

cdef void linRepulsion_2d(Node2D n1, Node2D n2, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double distance2 = xDist * xDist + yDist * yDist
    cdef double factor
    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


cdef void linRepulsion_region_2d(Node2D n, Region r, double coefficient):
    cdef double xDist = n.x - <double>r.massCenter[0]
    cdef double yDist = n.y - <double>r.massCenter[1]
    cdef double distance2 = xDist * xDist + yDist * yDist
    cdef double factor
    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        n.dx += xDist * factor
        n.dy += yDist * factor


cdef void linRepulsion_antiCollision_2d(Node2D n1, Node2D n2, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double euclidean = sqrt(xDist * xDist + yDist * yDist)
    cdef double distance = euclidean - n1.size - n2.size
    cdef double factor
    if euclidean > 0:
        if distance <= 0:
            factor = 100.0 * coefficient * n1.mass * n2.mass
        else:
            factor = coefficient * n1.mass * n2.mass / (distance * distance)
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


cdef void linRepulsion_antiCollision_region_2d(Node2D n, Region r, double coefficient):
    cdef double xDist = n.x - <double>r.massCenter[0]
    cdef double yDist = n.y - <double>r.massCenter[1]
    cdef double euclidean = sqrt(xDist * xDist + yDist * yDist)
    cdef double distance = euclidean - n.size
    cdef double factor
    if euclidean > 0:
        if distance <= 0:
            factor = 100.0 * coefficient * n.mass * r.mass
        else:
            factor = coefficient * n.mass * r.mass / (distance * distance)
        n.dx += xDist * factor
        n.dy += yDist * factor


cdef void linGravity_2d(Node2D n, double g):
    cdef double distance = sqrt(n.x * n.x + n.y * n.y)
    cdef double factor
    if distance > 0:
        factor = n.mass * g / distance
        n.dx -= n.x * factor
        n.dy -= n.y * factor


cdef void strongGravity_2d(Node2D n, double g, double coefficient):
    cdef double factor
    if n.x != 0.0 or n.y != 0.0:
        factor = coefficient * n.mass * g
        n.dx -= n.x * factor
        n.dy -= n.y * factor


cdef void linAttraction_2d(Node2D n1, Node2D n2, double e, bint distributedAttraction, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double factor
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


cdef void logAttraction_2d(Node2D n1, Node2D n2, double e, bint distributedAttraction, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double distance = sqrt(xDist * xDist + yDist * yDist)
    cdef double factor, log_factor
    if distance > 0:
        log_factor = log(1.0 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


cdef void linAttraction_antiCollision_2d(Node2D n1, Node2D n2, double e, bint distributedAttraction, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double euclidean = sqrt(xDist * xDist + yDist * yDist)
    cdef double distance = euclidean - n1.size - n2.size
    cdef double factor
    if distance > 0:
        if not distributedAttraction:
            factor = -coefficient * e
        else:
            factor = -coefficient * e / n1.mass
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


cdef void logAttraction_antiCollision_2d(Node2D n1, Node2D n2, double e, bint distributedAttraction, double coefficient):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double euclidean = sqrt(xDist * xDist + yDist * yDist)
    cdef double distance = euclidean - n1.size - n2.size
    cdef double factor, log_factor
    if distance > 0:
        log_factor = log(1.0 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


# ═══════════════════════════════════════════════════════════════════════
# N-dimensional force functions (list fields)
# ═══════════════════════════════════════════════════════════════════════

cdef void linRepulsion_nd(NodeND n1, NodeND n2, double coefficient):
    cdef int dim = n1.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double factor, dist_d
    for d in range(dim):
        dist_d = <double>n1.pos[d] - <double>n2.pos[d]
        distance2 += dist_d * dist_d
    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        for d in range(dim):
            dist_d = <double>n1.pos[d] - <double>n2.pos[d]
            n1.force[d] = <double>n1.force[d] + dist_d * factor
            n2.force[d] = <double>n2.force[d] - dist_d * factor


cdef void linRepulsion_region_nd(NodeND n, Region r, double coefficient):
    cdef int dim = n.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double factor, dist_d
    for d in range(dim):
        dist_d = <double>n.pos[d] - <double>r.massCenter[d]
        distance2 += dist_d * dist_d
    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        for d in range(dim):
            dist_d = <double>n.pos[d] - <double>r.massCenter[d]
            n.force[d] = <double>n.force[d] + dist_d * factor


cdef void linRepulsion_antiCollision_nd(NodeND n1, NodeND n2, double coefficient):
    cdef int dim = n1.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double euclidean, distance, factor, dist_d
    for d in range(dim):
        dist_d = <double>n1.pos[d] - <double>n2.pos[d]
        distance2 += dist_d * dist_d
    euclidean = sqrt(distance2)
    distance = euclidean - n1.size - n2.size
    if euclidean > 0:
        if distance <= 0:
            factor = 100.0 * coefficient * n1.mass * n2.mass
        else:
            factor = coefficient * n1.mass * n2.mass / (distance * distance)
        for d in range(dim):
            dist_d = <double>n1.pos[d] - <double>n2.pos[d]
            n1.force[d] = <double>n1.force[d] + dist_d * factor
            n2.force[d] = <double>n2.force[d] - dist_d * factor


cdef void linRepulsion_antiCollision_region_nd(NodeND n, Region r, double coefficient):
    cdef int dim = n.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double euclidean, distance, factor, dist_d
    for d in range(dim):
        dist_d = <double>n.pos[d] - <double>r.massCenter[d]
        distance2 += dist_d * dist_d
    euclidean = sqrt(distance2)
    distance = euclidean - n.size
    if euclidean > 0:
        if distance <= 0:
            factor = 100.0 * coefficient * n.mass * r.mass
        else:
            factor = coefficient * n.mass * r.mass / (distance * distance)
        for d in range(dim):
            dist_d = <double>n.pos[d] - <double>r.massCenter[d]
            n.force[d] = <double>n.force[d] + dist_d * factor


cdef void linGravity_nd(NodeND n, double g):
    cdef int dim = n.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double distance, factor
    for d in range(dim):
        distance2 += (<double>n.pos[d]) * (<double>n.pos[d])
    if distance2 > 0:
        distance = sqrt(distance2)
        factor = n.mass * g / distance
        for d in range(dim):
            n.force[d] = <double>n.force[d] - <double>n.pos[d] * factor


cdef void strongGravity_nd(NodeND n, double g, double coefficient):
    cdef int dim = n.dim
    cdef int d
    cdef double factor
    cdef bint any_nonzero = False
    for d in range(dim):
        if <double>n.pos[d] != 0.0:
            any_nonzero = True
            break
    if any_nonzero:
        factor = coefficient * n.mass * g
        for d in range(dim):
            n.force[d] = <double>n.force[d] - <double>n.pos[d] * factor


cpdef void linAttraction(n1, n2, double e, bint distributedAttraction, double coefficient=0):
    """Linear attraction along an edge (dispatches 2D vs ND)."""
    cdef int dim
    cdef int d
    cdef double factor, dist_d
    if isinstance(n1, Node2D):
        linAttraction_2d(<Node2D>n1, <Node2D>n2, e, distributedAttraction, coefficient)
    else:
        dim = (<NodeND>n1).dim
        if not distributedAttraction:
            factor = -coefficient * e
        else:
            factor = -coefficient * e / (<NodeND>n1).mass
        for d in range(dim):
            dist_d = <double>(<NodeND>n1).pos[d] - <double>(<NodeND>n2).pos[d]
            (<NodeND>n1).force[d] = <double>(<NodeND>n1).force[d] + dist_d * factor
            (<NodeND>n2).force[d] = <double>(<NodeND>n2).force[d] - dist_d * factor


cpdef void logAttraction(n1, n2, double e, bint distributedAttraction, double coefficient=0):
    """Logarithmic attraction along an edge (dispatches 2D vs ND)."""
    if isinstance(n1, Node2D):
        logAttraction_2d(<Node2D>n1, <Node2D>n2, e, distributedAttraction, coefficient)
    else:
        _logAttraction_nd(<NodeND>n1, <NodeND>n2, e, distributedAttraction, coefficient)


cdef void _logAttraction_nd(NodeND n1, NodeND n2, double e, bint distributedAttraction, double coefficient):
    cdef int dim = n1.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double distance, factor, log_factor, dist_d
    for d in range(dim):
        dist_d = <double>n1.pos[d] - <double>n2.pos[d]
        distance2 += dist_d * dist_d
    distance = sqrt(distance2)
    if distance > 0:
        log_factor = log(1.0 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        for d in range(dim):
            dist_d = <double>n1.pos[d] - <double>n2.pos[d]
            n1.force[d] = <double>n1.force[d] + dist_d * factor
            n2.force[d] = <double>n2.force[d] - dist_d * factor


cpdef void linAttraction_antiCollision(n1, n2, double e, bint distributedAttraction, double coefficient=0):
    if isinstance(n1, Node2D):
        linAttraction_antiCollision_2d(<Node2D>n1, <Node2D>n2, e, distributedAttraction, coefficient)
    else:
        _linAttraction_antiCollision_nd(<NodeND>n1, <NodeND>n2, e, distributedAttraction, coefficient)


cdef void _linAttraction_antiCollision_nd(NodeND n1, NodeND n2, double e, bint distributedAttraction, double coefficient):
    cdef int dim = n1.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double euclidean, distance, factor, dist_d
    for d in range(dim):
        dist_d = <double>n1.pos[d] - <double>n2.pos[d]
        distance2 += dist_d * dist_d
    euclidean = sqrt(distance2)
    distance = euclidean - n1.size - n2.size
    if distance > 0:
        if not distributedAttraction:
            factor = -coefficient * e
        else:
            factor = -coefficient * e / n1.mass
        for d in range(dim):
            dist_d = <double>n1.pos[d] - <double>n2.pos[d]
            n1.force[d] = <double>n1.force[d] + dist_d * factor
            n2.force[d] = <double>n2.force[d] - dist_d * factor


cpdef void logAttraction_antiCollision(n1, n2, double e, bint distributedAttraction, double coefficient=0):
    if isinstance(n1, Node2D):
        logAttraction_antiCollision_2d(<Node2D>n1, <Node2D>n2, e, distributedAttraction, coefficient)
    else:
        _logAttraction_antiCollision_nd(<NodeND>n1, <NodeND>n2, e, distributedAttraction, coefficient)


cdef void _logAttraction_antiCollision_nd(NodeND n1, NodeND n2, double e, bint distributedAttraction, double coefficient):
    cdef int dim = n1.dim
    cdef int d
    cdef double distance2 = 0.0
    cdef double euclidean, distance, factor, log_factor, dist_d
    for d in range(dim):
        dist_d = <double>n1.pos[d] - <double>n2.pos[d]
        distance2 += dist_d * dist_d
    euclidean = sqrt(distance2)
    distance = euclidean - n1.size - n2.size
    if distance > 0:
        log_factor = log(1.0 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        for d in range(dim):
            dist_d = <double>n1.pos[d] - <double>n2.pos[d]
            n1.force[d] = <double>n1.force[d] + dist_d * factor
            n2.force[d] = <double>n2.force[d] - dist_d * factor


# ═══════════════════════════════════════════════════════════════════════
# Batch force application (dispatch once, then C-speed inner loop)
# ═══════════════════════════════════════════════════════════════════════

cpdef void apply_repulsion(list nodes, double coefficient, bint adjustSizes=False):
    """Apply repulsion forces between all node pairs."""
    cdef int i, j, n_nodes
    cdef Node2D n2d_1, n2d_2
    cdef NodeND nnd_1, nnd_2

    n_nodes = len(nodes)
    if n_nodes > 0 and isinstance(nodes[0], Node2D):
        for i in range(n_nodes):
            n2d_1 = <Node2D>nodes[i]
            for j in range(i):
                n2d_2 = <Node2D>nodes[j]
                if adjustSizes:
                    linRepulsion_antiCollision_2d(n2d_1, n2d_2, coefficient)
                else:
                    linRepulsion_2d(n2d_1, n2d_2, coefficient)
    else:
        for i in range(n_nodes):
            nnd_1 = <NodeND>nodes[i]
            for j in range(i):
                nnd_2 = <NodeND>nodes[j]
                if adjustSizes:
                    linRepulsion_antiCollision_nd(nnd_1, nnd_2, coefficient)
                else:
                    linRepulsion_nd(nnd_1, nnd_2, coefficient)


cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, bint useStrongGravity=False):
    """Apply gravitational forces to all nodes."""
    cdef Node2D n2d
    cdef NodeND nnd

    if len(nodes) > 0 and isinstance(nodes[0], Node2D):
        if not useStrongGravity:
            for n2d in nodes:
                linGravity_2d(n2d, gravity)
        else:
            for n2d in nodes:
                strongGravity_2d(n2d, gravity, scalingRatio)
    else:
        if not useStrongGravity:
            for nnd in nodes:
                linGravity_nd(nnd, gravity)
        else:
            for nnd in nodes:
                strongGravity_nd(nnd, gravity, scalingRatio)


cpdef void apply_attraction(list nodes, list edges, bint distributedAttraction, double coefficient,
                            double edgeWeightInfluence, bint linLogMode=False, bint adjustSizes=False):
    """Apply attraction forces along all edges."""
    cdef Edge edge
    cdef double w
    cdef Node2D n2d_1, n2d_2
    cdef bint is2d = len(nodes) > 0 and isinstance(nodes[0], Node2D)

    # Macro for edge iteration with weight handling
    # Uses 2D or ND attraction function based on node type
    if is2d:
        if adjustSizes:
            if linLogMode:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        logAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        logAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        logAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
            else:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        linAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        linAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        linAttraction_antiCollision_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
        else:
            if linLogMode:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        logAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        logAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        logAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
            else:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        linAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        linAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        linAttraction_2d(<Node2D>nodes[edge.node1], <Node2D>nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
    else:
        # ND path — uses cpdef dispatchers
        if adjustSizes:
            if linLogMode:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        logAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        logAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        logAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
            else:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        linAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        linAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        linAttraction_antiCollision(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
        else:
            if linLogMode:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        logAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        logAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        logAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)
            else:
                if edgeWeightInfluence == 0:
                    for edge in edges:
                        linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
                elif edgeWeightInfluence == 1:
                    for edge in edges:
                        linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
                else:
                    for edge in edges:
                        linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), distributedAttraction, coefficient)


# ═══════════════════════════════════════════════════════════════════════
# Barnes-Hut spatial tree
# ═══════════════════════════════════════════════════════════════════════

cdef class Region:
    """Barnes-Hut tree node (single class, dispatches to 2D/ND force functions)."""
    cdef public double mass
    cdef public list massCenter
    cdef public double size
    cdef public list nodes
    cdef public list subregions
    cdef int _dim
    cdef bint _is2d

    def __init__(self, list nodes):
        self.mass = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        if nodes:
            self._dim = nodes[0].dim
            self._is2d = isinstance(nodes[0], Node2D)
            self.massCenter = [0.0] * self._dim
        else:
            self._dim = 2
            self._is2d = True
            self.massCenter = [0.0, 0.0]
        self.updateMassAndGeometry()

    cdef void updateMassAndGeometry(self):
        cdef int dim = self._dim
        cdef int d
        cdef double distance, distance2
        cdef Node2D n2d
        cdef NodeND nnd
        cdef list massSum

        if len(self.nodes) == 1:
            if self._is2d:
                n2d = <Node2D>self.nodes[0]
                self.mass = n2d.mass
                self.massCenter = [n2d.x, n2d.y]
            else:
                nnd = <NodeND>self.nodes[0]
                self.mass = nnd.mass
                self.massCenter = list(nnd.pos)
            self.size = 0.0
        elif len(self.nodes) > 1:
            self.mass = 0.0
            massSum = [0.0] * dim

            if self._is2d:
                for n2d in self.nodes:
                    self.mass += n2d.mass
                    massSum[0] = <double>massSum[0] + n2d.x * n2d.mass
                    massSum[1] = <double>massSum[1] + n2d.y * n2d.mass
            else:
                for nnd in self.nodes:
                    self.mass += nnd.mass
                    for d in range(dim):
                        massSum[d] = <double>massSum[d] + <double>nnd.pos[d] * nnd.mass

            if self.mass > 0:
                for d in range(dim):
                    self.massCenter[d] = <double>massSum[d] / self.mass

            self.size = 0.0
            if self._is2d:
                for n2d in self.nodes:
                    distance = sqrt((n2d.x - <double>self.massCenter[0]) ** 2 +
                                    (n2d.y - <double>self.massCenter[1]) ** 2)
                    if 2 * distance > self.size:
                        self.size = 2 * distance
            else:
                for nnd in self.nodes:
                    distance2 = 0.0
                    for d in range(dim):
                        distance2 += (<double>nnd.pos[d] - <double>self.massCenter[d]) ** 2
                    distance = sqrt(distance2)
                    if 2 * distance > self.size:
                        self.size = 2 * distance

    cpdef void buildSubRegions(self):
        cdef int dim = self._dim
        cdef int d, bucket_idx, num_buckets
        cdef Node2D n2d
        cdef NodeND nnd
        cdef Region subregion
        cdef list buckets, bucket_nodes

        if len(self.nodes) > 1:
            num_buckets = 1 << dim
            buckets = [[] for _ in range(num_buckets)]

            if self._is2d:
                for n2d in self.nodes:
                    bucket_idx = 0
                    if n2d.x >= <double>self.massCenter[0]:
                        bucket_idx |= 1
                    if n2d.y >= <double>self.massCenter[1]:
                        bucket_idx |= 2
                    (<list>buckets[bucket_idx]).append(n2d)
            else:
                for nnd in self.nodes:
                    bucket_idx = 0
                    for d in range(dim):
                        if <double>nnd.pos[d] >= <double>self.massCenter[d]:
                            bucket_idx |= (1 << d)
                    (<list>buckets[bucket_idx]).append(nnd)

            for bucket_nodes in buckets:
                if len(bucket_nodes) > 0:
                    if len(bucket_nodes) < len(self.nodes):
                        subregion = Region(bucket_nodes)
                        self.subregions.append(subregion)
                    else:
                        for node in bucket_nodes:
                            subregion = Region([node])
                            self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()

    cdef void applyForce_2d(self, Node2D n, double theta, double coefficient, bint adjustSizes):
        cdef double distance
        cdef Region subregion

        if len(self.nodes) == 0:
            return
        if len(self.nodes) < 2:
            if self.nodes[0] is not n:
                if adjustSizes:
                    linRepulsion_antiCollision_region_2d(n, self, coefficient)
                else:
                    linRepulsion_region_2d(n, self, coefficient)
        else:
            distance = sqrt((n.x - <double>self.massCenter[0]) ** 2 +
                            (n.y - <double>self.massCenter[1]) ** 2)
            if distance * theta > self.size:
                if adjustSizes:
                    linRepulsion_antiCollision_region_2d(n, self, coefficient)
                else:
                    linRepulsion_region_2d(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce_2d(n, theta, coefficient, adjustSizes)

    cdef void applyForce_nd(self, NodeND n, double theta, double coefficient, bint adjustSizes):
        cdef int dim = self._dim
        cdef int d
        cdef double distance, distance2
        cdef Region subregion

        if len(self.nodes) == 0:
            return
        if len(self.nodes) < 2:
            if self.nodes[0] is not n:
                if adjustSizes:
                    linRepulsion_antiCollision_region_nd(n, self, coefficient)
                else:
                    linRepulsion_region_nd(n, self, coefficient)
        else:
            distance2 = 0.0
            for d in range(dim):
                distance2 += (<double>n.pos[d] - <double>self.massCenter[d]) ** 2
            distance = sqrt(distance2)
            if distance * theta > self.size:
                if adjustSizes:
                    linRepulsion_antiCollision_region_nd(n, self, coefficient)
                else:
                    linRepulsion_region_nd(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce_nd(n, theta, coefficient, adjustSizes)

    cpdef applyForceOnNodes(self, list nodes, double theta, double coefficient=0, bint adjustSizes=False):
        cdef Node2D n2d
        cdef NodeND nnd
        if self._is2d:
            for n2d in nodes:
                self.applyForce_2d(n2d, theta, coefficient, adjustSizes)
        else:
            for nnd in nodes:
                self.applyForce_nd(nnd, theta, coefficient, adjustSizes)


# ═══════════════════════════════════════════════════════════════════════
# Speed adjustment and force application
# ═══════════════════════════════════════════════════════════════════════

cpdef dict adjustSpeedAndApplyForces(list nodes, double speed, double speedEfficiency, double jitterTolerance, bint adjustSizes=False):
    """Adjust speed adaptively and apply accumulated forces to node positions."""
    cdef double totalSwinging = 0.0
    cdef double totalEffectiveTraction = 0.0
    cdef double swinging, swing2, tract2, diff_x, diff_y, sum_x, sum_y
    cdef double factor
    cdef Node2D n2d
    cdef NodeND nnd
    cdef int d, dim
    cdef double diff_d, sum_d
    cdef bint is2d = len(nodes) > 0 and isinstance(nodes[0], Node2D)

    # Phase 1: compute swing and traction
    if is2d:
        for n2d in nodes:
            diff_x = n2d.old_dx - n2d.dx
            diff_y = n2d.old_dy - n2d.dy
            sum_x = n2d.old_dx + n2d.dx
            sum_y = n2d.old_dy + n2d.dy
            swinging = sqrt(diff_x * diff_x + diff_y * diff_y)
            totalSwinging += n2d.mass * swinging
            totalEffectiveTraction += 0.5 * n2d.mass * sqrt(sum_x * sum_x + sum_y * sum_y)
    else:
        for nnd in nodes:
            dim = nnd.dim
            swing2 = 0.0
            tract2 = 0.0
            for d in range(dim):
                diff_d = <double>nnd.old_force[d] - <double>nnd.force[d]
                sum_d = <double>nnd.old_force[d] + <double>nnd.force[d]
                swing2 += diff_d * diff_d
                tract2 += sum_d * sum_d
            swinging = sqrt(swing2)
            totalSwinging += nnd.mass * swinging
            totalEffectiveTraction += 0.5 * nnd.mass * sqrt(tract2)

    # Phase 2: adaptive speed
    cdef double estimatedOptimalJitterTolerance = 0.05 * sqrt(len(nodes))
    cdef double minJT = sqrt(estimatedOptimalJitterTolerance)
    cdef double maxJT = 10
    cdef double jt
    if len(nodes) > 0 and totalEffectiveTraction > 0:
        jt = jitterTolerance * max(minJT,
                                   min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                       len(nodes) * len(nodes))))
    else:
        jt = jitterTolerance * minJT

    cdef double minSpeedEfficiency = 0.05

    if totalEffectiveTraction > 0 and totalSwinging / totalEffectiveTraction > 2.0:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= 0.5
        jt = max(jt, jitterTolerance)

    cdef double targetSpeed
    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= 0.7
    elif speed < 1000:
        speedEfficiency *= 1.3

    cdef double maxRise = 0.5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    # Phase 3: apply forces
    if is2d:
        for n2d in nodes:
            diff_x = n2d.old_dx - n2d.dx
            diff_y = n2d.old_dy - n2d.dy
            swinging = n2d.mass * sqrt(diff_x * diff_x + diff_y * diff_y)
            factor = speed / (1.0 + sqrt(speed * swinging))
            if adjustSizes and n2d.size > 0:
                if factor > 10.0 / n2d.size:
                    factor = 10.0 / n2d.size
            n2d.x += n2d.dx * factor
            n2d.y += n2d.dy * factor
    else:
        for nnd in nodes:
            dim = nnd.dim
            swing2 = 0.0
            for d in range(dim):
                diff_d = <double>nnd.old_force[d] - <double>nnd.force[d]
                swing2 += diff_d * diff_d
            swinging = nnd.mass * sqrt(swing2)
            factor = speed / (1.0 + sqrt(speed * swinging))
            if adjustSizes and nnd.size > 0:
                if factor > 10.0 / nnd.size:
                    factor = 10.0 / nnd.size
            for d in range(dim):
                nnd.pos[d] = <double>nnd.pos[d] + <double>nnd.force[d] * factor

    return {'speed': speed, 'speedEfficiency': speedEfficiency}
