# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""CPU-intensive ForceAtlas2 routines optimized with Cython.

This module contains the core force computation functions used by the
ForceAtlas2 algorithm. When compiled with Cython, these provide a
10-100x speedup over the pure Python fallback.

Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
Available under the GPLv3
"""

from libc.math cimport log, sqrt


cdef class Node:
    """Represents a graph node with position and force vectors."""
    cdef public double mass
    cdef public double old_dx, old_dy
    cdef public double dx, dy
    cdef public double x, y

    def __init__(self):
        self.mass = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0


cdef class Edge:
    """Represents a graph edge with source, target, and weight."""
    cdef public int node1, node2
    cdef public double weight

    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# Force computation functions from ForceFactory.java

cdef void linRepulsion(Node n1, Node n2, double coefficient=0):
    """Linear repulsion between two nodes."""
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


cdef void linRepulsion_region(Node n, Region r, double coefficient=0):
    """Linear repulsion between a node and a region (Barnes-Hut)."""
    cdef double xDist = n.x - r.massCenterX
    cdef double yDist = n.y - r.massCenterY
    cdef double distance2 = xDist * xDist + yDist * yDist
    cdef double factor

    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        n.dx += xDist * factor
        n.dy += yDist * factor


cdef void linGravity(Node n, double g):
    """Linear gravity — attracts node toward the origin."""
    cdef double xDist = n.x
    cdef double yDist = n.y
    cdef double distance = sqrt(xDist * xDist + yDist * yDist)
    cdef double factor

    if distance > 0:
        factor = n.mass * g / distance
        n.dx -= xDist * factor
        n.dy -= yDist * factor


cdef void strongGravity(Node n, double g, double coefficient=0):
    """Strong gravity — distance-independent attraction toward origin."""
    cdef double xDist = n.x
    cdef double yDist = n.y
    cdef double factor

    if xDist != 0 or yDist != 0:
        factor = coefficient * n.mass * g
        n.dx -= xDist * factor
        n.dy -= yDist * factor


cpdef void linAttraction(Node n1, Node n2, double e, bint distributedAttraction, double coefficient=0):
    """Linear attraction along an edge."""
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


cpdef void logAttraction(Node n1, Node n2, double e, bint distributedAttraction, double coefficient=0):
    """Logarithmic attraction along an edge (LinLog mode).

    F = -coefficient * edgeWeight * log(1 + distance)
    Factor (force/distance) = -coefficient * edgeWeight * log(1 + distance) / distance
    Reference: Jacomy et al. 2014 Formula 3; Gephi ForceFactory.java logAttraction
    """
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double distance = sqrt(xDist * xDist + yDist * yDist)
    cdef double factor
    cdef double log_factor

    if distance > 0:
        log_factor = log(1 + distance) / distance
        if not distributedAttraction:
            factor = -coefficient * e * log_factor
        else:
            factor = -coefficient * e * log_factor / n1.mass
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


# Batch force application functions

cpdef void apply_repulsion(list nodes, double coefficient):
    """Apply repulsion forces between all node pairs."""
    cdef int i = 0
    cdef int j
    cdef Node n1, n2
    for n1 in nodes:
        j = i
        for n2 in nodes:
            if j == 0:
                break
            linRepulsion(n1, n2, coefficient)
            j -= 1
        i += 1


cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, bint useStrongGravity=False):
    """Apply gravitational forces to all nodes."""
    cdef Node n
    if not useStrongGravity:
        for n in nodes:
            linGravity(n, gravity)
    else:
        for n in nodes:
            strongGravity(n, gravity, scalingRatio)


cpdef void apply_attraction(list nodes, list edges, bint distributedAttraction, double coefficient,
                            double edgeWeightInfluence, bint linLogMode=False):
    """Apply attraction forces along all edges."""
    cdef Edge edge
    if linLogMode:
        if edgeWeightInfluence == 0:
            for edge in edges:
                logAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
        elif edgeWeightInfluence == 1:
            for edge in edges:
                logAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
        else:
            for edge in edges:
                logAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                              distributedAttraction, coefficient)
    else:
        if edgeWeightInfluence == 0:
            for edge in edges:
                linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
        elif edgeWeightInfluence == 1:
            for edge in edges:
                linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
        else:
            for edge in edges:
                linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence),
                              distributedAttraction, coefficient)


# Barnes-Hut optimization

cdef class Region:
    """Barnes-Hut tree node for spatial approximation of repulsion forces."""
    cdef public double mass
    cdef public double massCenterX, massCenterY
    cdef public double size
    cdef public list nodes
    cdef public list subregions

    def __init__(self, list nodes):
        self.mass = 0.0
        self.massCenterX = 0.0
        self.massCenterY = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.updateMassAndGeometry()

    cdef void updateMassAndGeometry(self):
        cdef double massSumX = 0
        cdef double massSumY = 0
        cdef double distance
        cdef Node n

        if len(self.nodes) == 1:
            n = self.nodes[0]
            self.mass = n.mass
            self.massCenterX = n.x
            self.massCenterY = n.y
            self.size = 0.0
        elif len(self.nodes) > 1:
            self.mass = 0
            for n in self.nodes:
                self.mass += n.mass
                massSumX += n.x * n.mass
                massSumY += n.y * n.mass
            if self.mass > 0:
                self.massCenterX = massSumX / self.mass
                self.massCenterY = massSumY / self.mass

            self.size = 0.0
            for n in self.nodes:
                distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
                self.size = max(self.size, 2 * distance)

    cpdef void buildSubRegions(self):
        cdef list topleftNodes
        cdef list bottomleftNodes
        cdef list toprightNodes
        cdef list bottomrightNodes
        cdef Node n
        cdef Region subregion

        if len(self.nodes) > 1:
            topleftNodes = []
            bottomleftNodes = []
            toprightNodes = []
            bottomrightNodes = []

            for n in self.nodes:
                if n.x < self.massCenterX:
                    if n.y < self.massCenterY:
                        bottomleftNodes.append(n)
                    else:
                        topleftNodes.append(n)
                else:
                    if n.y < self.massCenterY:
                        bottomrightNodes.append(n)
                    else:
                        toprightNodes.append(n)

            if len(topleftNodes) > 0:
                if len(topleftNodes) < len(self.nodes):
                    subregion = Region(topleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in topleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomleftNodes) > 0:
                if len(bottomleftNodes) < len(self.nodes):
                    subregion = Region(bottomleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(toprightNodes) > 0:
                if len(toprightNodes) < len(self.nodes):
                    subregion = Region(toprightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in toprightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomrightNodes) > 0:
                if len(bottomrightNodes) < len(self.nodes):
                    subregion = Region(bottomrightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomrightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()

    cdef void applyForce(self, Node n, double theta, double coefficient=0):
        cdef double distance
        cdef Region subregion

        if len(self.nodes) == 0:
            return
        if len(self.nodes) < 2:
            if self.nodes[0] is not n:
                linRepulsion_region(n, self, coefficient)
        else:
            distance = sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
            if distance * theta > self.size:
                linRepulsion_region(n, self, coefficient)
            else:
                for subregion in self.subregions:
                    subregion.applyForce(n, theta, coefficient)

    cpdef applyForceOnNodes(self, list nodes, double theta, double coefficient=0):
        cdef Node n
        for n in nodes:
            self.applyForce(n, theta, coefficient)


cpdef dict adjustSpeedAndApplyForces(list nodes, double speed, double speedEfficiency, double jitterTolerance):
    """Adjust speed adaptively and apply accumulated forces to node positions."""
    cdef double totalSwinging = 0.0
    cdef double totalEffectiveTraction = 0.0
    cdef Node n
    cdef double swinging

    for n in nodes:
        swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        totalSwinging += n.mass * swinging
        totalEffectiveTraction += .5 * n.mass * sqrt(
            (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))

    cdef double estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
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
            speedEfficiency *= .5
        jt = max(jt, jitterTolerance)

    cdef double targetSpeed
    if totalSwinging == 0:
        targetSpeed = float('inf')
    else:
        targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

    if totalSwinging > jt * totalEffectiveTraction:
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= .7
    elif speed < 1000:
        speedEfficiency *= 1.3

    cdef double maxRise = .5
    speed = speed + min(targetSpeed - speed, maxRise * speed)

    cdef double factor
    for n in nodes:
        swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        factor = speed / (1.0 + sqrt(speed * swinging))
        n.x = n.x + (n.dx * factor)
        n.y = n.y + (n.dy * factor)

    return {'speed': speed, 'speedEfficiency': speedEfficiency}
