# This file allows separating the most CPU intensive routines from the
# main code.  This allows them to be optimized with Cython.  If you
# don't have Cython, this will run normally.  However, if you use
# Cython, you'll get speed boosts from 10-100x automatically.
#
# The only catch is that IF YOU MODIFY THIS FILE, YOU MUST ALSO MODIFY
# fa2util.pxd TO REFLECT ANY CHANGES IN FUNCTION DEFINITIONS!
#
# Copyright 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

from math import sqrt


# This will substitute for the nLayout object
class Node:
    def __init__(self):
        self.mass = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0


# This is not in the original java code, but it makes it easier to
# deal with edges.
class Edge:
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


class Region:
    def __init__(self, nodes):
        self.mass = 0.0
        self.massCenterX = 0.0
        self.massCenterY = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []


# Here are some functions from ForceFactory.java
# =============================================

# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` (and optionally `n2`).  It does
# not return anything.
def linRepulsion(n1, n2, coefficient=0):
    xDist = n1.x - n2.x
    yDist = n1.y - n2.y
    distance2 = xDist * xDist + yDist * yDist  # Distance squared

    if distance2 > 0:
        factor = coefficient * n1.mass * n2.mass / distance2
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


def linRepulsionRegion(n, r, coefficient=0):
    xDist = n.x - r.massCenterX
    yDist = n.y - r.massCenterY
    distance2 = xDist * xDist + yDist * yDist

    if distance2 > 0:
        factor = coefficient * n.mass * r.mass / distance2
        n.dx += xDist * factor
        n.dy += yDist * factor


# Gravity repulsion function.  For some reason, gravity was included
# within the linRepulsion function in the original gephi java code,
# which doesn't make any sense (considering a. gravity is unrelated to
# nodes repelling each other, and b. gravity is actually an
# attraction).
def linGravity(n, g, coefficient=0):
    xDist = n.x
    yDist = n.y
    distance = sqrt(xDist * xDist + yDist * yDist)

    if distance > 0:
        factor = coefficient * n.mass * g / distance
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Strong gravity force function.  `n` should be a node, and `g`
# should be a constant by which to apply the force.
def strongGravity(n, g, coefficient=0):
    xDist = n.x
    yDist = n.y

    if xDist != 0 and yDist != 0:
        factor = coefficient * n.mass * g
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Attraction function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` (and optionally `n2`).  It does
# not return anything.
def linAttraction(n1, n2, e, coefficient=0):
    xDist = n1.x - n2.x
    yDist = n1.y - n2.y
    factor = -coefficient * e
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


# The following functions iterate through the nodes or edges and apply
# the forces directly to the node objects.  These iterations are here
# instead of the main file because Python is slow with loops.  Where
# relevant, they also contain the logic to select which version of the
# force function to use.
def apply_repulsion(nodes, coefficient):
    for i in range(0, len(nodes)):
        for j in range(0, i):
            linRepulsion(nodes[i], nodes[j], coefficient)


def apply_gravity(nodes, gravity, scalingRatio, useStrongGravity=False):
    if not useStrongGravity:
        for i in range(0, len(nodes)):
            linGravity(nodes[i], gravity / scalingRatio, scalingRatio)
    else:
        for i in range(0, len(nodes)):
            strongGravity(nodes[i], gravity / scalingRatio, scalingRatio)


def apply_attraction(nodes, edges, coefficient, edgeWeightInfluence):
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    if edgeWeightInfluence == 0:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], 1, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, coefficient)
    else:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], pow(edge.weight, edgeWeightInfluence), coefficient)


def updateMassAndGeometry(nodes):
    region = Region(nodes)
    if len(nodes) > 1:
        region.mass = 0
        massSumX = 0
        massSumY = 0
        for n in nodes:
            region.mass += n.mass
            massSumX += n.x * n.mass
            massSumY += n.y * n.mass
        region.massCenterX = massSumX / region.mass;
        region.massCenterY = massSumY / region.mass;

        region.size = 0.0;
        for n in nodes:
            distance = sqrt((n.x - region.massCenterX) ** 2 + (n.y - region.massCenterY) ** 2)
            region.size = max(region.size, 2 * distance)
    return region


def buildSubRegions(region):
    if len(region.nodes) > 1:

        leftNodes = []
        rightNodes = []
        for n in region.nodes:
            if n.x < region.massCenterX:
                leftNodes.append(n)
            else:
                rightNodes.append(n)

        topleftNodes = []
        bottomleftNodes = []
        for n in leftNodes:
            if n.y < region.massCenterY:
                topleftNodes.append(n)
            else:
                bottomleftNodes.append(n)

        toprightNodes = []
        bottomrightNodes = []
        for n in rightNodes:
            if n.y < region.massCenterY:
                toprightNodes.append(n)
            else:
                bottomrightNodes.append(n)

        if len(topleftNodes) > 0:
            if len(topleftNodes) < len(region.nodes):
                subregion = updateMassAndGeometry(topleftNodes)
                region.subregions.append(subregion)
            else:
                for n in topleftNodes:
                    subregion = updateMassAndGeometry([n])
                    region.subregions.append(subregion)

        if len(bottomleftNodes) > 0:
            if len(bottomleftNodes) < len(region.nodes):
                subregion = updateMassAndGeometry(bottomleftNodes)
                region.subregions.append(subregion)
            else:
                for n in bottomleftNodes:
                    subregion = updateMassAndGeometry([n])
                    region.subregions.append(subregion)

        if len(toprightNodes) > 0:
            if len(toprightNodes) < len(region.nodes):
                subregion = updateMassAndGeometry(toprightNodes)
                region.subregions.append(subregion)
            else:
                for n in toprightNodes:
                    subregion = updateMassAndGeometry([n])
                    region.subregions.append(subregion)

        if len(bottomrightNodes) > 0:
            if len(bottomrightNodes) < len(region.nodes):
                subregion = updateMassAndGeometry(bottomrightNodes)
                region.subregions.append(subregion)
            else:
                for n in bottomrightNodes:
                    subregion = updateMassAndGeometry([n])
                    region.subregions.append(subregion)

        for subregion in region.subregions:
            buildSubRegions(subregion)

def applyForce(n, r, theta, coefficient=0):
    if len(r.nodes) < 2:
        linRepulsion(n, r.nodes[0], coefficient)
    else:
        distance = sqrt((n.x - r.massCenterX) ** 2 + (n.y - r.massCenterY) ** 2)
        if distance * theta > r.size:
            linRepulsionRegion(n, r, coefficient)
        else:
            for subregion in r.subregions:
                applyForce(n, subregion, theta, coefficient)


try:
    import cython

    if not cython.compiled:
        print("Warning: uncompiled fa2util module.  Compile with cython for a 10-100x speed boost.")
except:
    print("No cython detected.  Install cython and compile the fa2util module for a 10-100x speed boost.")
