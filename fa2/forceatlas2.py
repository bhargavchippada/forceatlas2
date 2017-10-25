# This is a python implementation of the ForceAtlas2 plugin from Gephi
# intended to be used with networkx, but is in theory independent of
# it since it only relies on the adjacency matrix.  This
# implementation is based directly on the Gephi plugin:
#
# https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java
#
# This is in contrast to an implementation I found in R (which uses
# strange parametrization) and to one in Python (which is based on the
# spring_layout).  The python implementation did not seem to work, and
# I did not try the R implementation because I didn't want to spend a
# long time setting up a messy pipe system only to find that the
# algorithm was as wrong as the python one.  
#
# For simplicity and for keeping code in sync with upstream, I have
# reused as many of the variable/function names as possible, even when
# they are in a more java-like style (e.g. camalcase)
#
# I wrote this because I was unable to find a graph layout algorithm
# in Python that clearly showed modular structure.
#
# NOTES: Currently, this only works for unweighted, undirected graphs.
#
# Copyright 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

import random
import time
from math import sqrt

import numpy
import scipy

from . import fa2util


# Default parameters
# ==================


# Given an adjacency matrix, this function computes the node positions
# according to the ForceAtlas2 layout algorithm.  It takes the same
# arguments that one would give to the ForceAtlas2 algorithm in Gephi.
# Not all of them are implemented.  See below for a description of
# each parameter and whether or not it has been implemented.
#
# This function will return a list of X-Y coordinate tuples, ordered
# in the same way as the rows/columns in the input matrix.
#
# The only reason you would want to run this directly is if you don't
# use networkx.  In this case, you'll likely need to convert the
# output to a more usable format.  If you do use networkx, use the
# "forceatlas2_networkx_layout" function below.

class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = 0.0
        self.total_time = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.total_time += (time.time() - self.start_time)

    def print(self):
        print(self.name, " took total ", self.total_time, " seconds")


class ForceAtlas2:
    def __init__(self,
                 # Behavior alternatives
                 outboundAttractionDistribution=False,  # "Dissuade hubs" # NOT (fully) IMPLEMENTED
                 linLogMode=False,  # NOT IMPLEMENTED
                 adjustSizes=False,  # "Prevent overlap" # NOT IMPLEMENTED
                 edgeWeightInfluence=1.0,

                 # Performance
                 jitterTolerance=1.0,  # "Tolerance"
                 barnesHutOptimize=False,  # NOT IMPLEMENTED
                 barnesHutTheta=1.2,  # NOT IMPLEMENTED

                 # Tuning
                 scalingRatio=2.0,
                 strongGravityMode=False,
                 gravity=1.0):
        assert outboundAttractionDistribution == linLogMode == adjustSizes == False, "You selected a feature that has not been implemented yet..."
        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.adjustSizes = adjustSizes
        self.edgeWeightInfluence = edgeWeightInfluence
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity

    def init(self,
             G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
             pos=None  # Array of initial positions
             ):
        isSparse = False
        if isinstance(G, numpy.ndarray):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert numpy.all(G.T == G), "G is not symmetric.  Currently only undirected graphs are supported"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
            G = G.tolil()
            isSparse = True
        else:
            assert False, "G is not numpy ndarray or scipy sparse matrix"

        # Put nodes into a data structure we can understand
        nodes = []
        for i in range(0, G.shape[0]):
            n = fa2util.Node()
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + numpy.count_nonzero(G[i])
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we can understand
        edges = []
        es = numpy.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            if e[1] <= e[0]: continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]  # The index of the first node in `nodes`
            edge.node2 = e[1]  # The index of the second node in `nodes`
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    def forceatlas2(self,
                    G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
                    pos=None,  # Array of initial positions
                    iterations=100  # Number of times to iterate the main loop
                    ):
        # Initializing, initAlgo()
        # ================================================================

        # speed and speedEfficiency describe a scaling factor of dx and dy
        # before x and y are adjusted.  These are modified as the
        # algorithm runs to help ensure convergence.
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos)
        # ================================================================

        # Main loop, i.e. goAlgo()
        # ========================

        barnes_hut_timer = Timer(name="BarnesHut")
        repulsion_timer = Timer(name="Repulsion")
        gravity_timer = Timer(name="Gravity")
        attraction_timer = Timer(name="Attraction")
        # Each iteration of this loop reseprensts a call to goAlgo().
        for _i in range(0, iterations):
            for n in nodes:
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0

            barnes_hut_timer.start()
            # Barnes Hut optimization
            if self.barnesHutOptimize:
                rootRegion = fa2util.updateMassAndGeometry(nodes)
                fa2util.buildSubRegions(rootRegion)
            barnes_hut_timer.stop()

            if self.outboundAttractionDistribution:
                outboundAttCompensation = numpy.mean([n.mass for n in nodes])

            # I did not implement parallelization here.  Also, Barnes Hut
            # optimization would change this from "linRepulsion" to another
            # version of the function.

            repulsion_timer.start()
            if self.barnesHutOptimize:
                for n in nodes:
                    fa2util.applyForce(n, rootRegion, self.barnesHutTheta, self.scalingRatio)
            else:
                fa2util.apply_repulsion(nodes, self.scalingRatio)
            repulsion_timer.stop()

            gravity_timer.start()
            fa2util.apply_gravity(nodes, self.gravity, self.scalingRatio, useStrongGravity=self.strongGravityMode)
            gravity_timer.stop()

            # If other forms of attraction were implemented they would be
            # selected here.
            attraction_timer.start()
            fa2util.apply_attraction(nodes, edges, 1.0, self.edgeWeightInfluence)
            attraction_timer.stop()

            # Auto adjust speed.
            totalSwinging = 0.0  # How much irregular movement
            totalEffectiveTraction = 0.0  # How much useful movement
            for n in nodes:
                swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
                totalSwinging += n.mass * swinging
                totalEffectiveTraction += .5 * n.mass * sqrt(
                    (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))

            # Optimize jitter tolerance.  The 'right' jitter tolerance for
            # this network. Bigger networks need more tolerance. Denser
            # networks need less tolerance. Totally empiric.
            estimatedOptimalJitterTolerance = .05 * sqrt(len(nodes))
            minJT = sqrt(estimatedOptimalJitterTolerance)
            maxJT = 10
            jt = self.jitterTolerance * max(minJT,
                                            min(maxJT, estimatedOptimalJitterTolerance * totalEffectiveTraction / (
                                                len(nodes) * len(nodes))))

            minSpeedEfficiency = 0.05

            # Protective against erratic behavior
            if totalSwinging / totalEffectiveTraction > 2.0:
                if speedEfficiency > minSpeedEfficiency:
                    speedEfficiency *= .5
                jt = max(jt, self.jitterTolerance)

            targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging

            if totalSwinging > jt * totalEffectiveTraction:
                if speedEfficiency > minSpeedEfficiency:
                    speedEfficiency *= .7
            elif speed < 1000:
                speedEfficiency *= 1.3

            # But the speed shoudn't rise too much too quickly, since it would
            # make the convergence drop dramatically.
            maxRise = .5
            speed = speed + min(targetSpeed - speed, maxRise * speed)

            # Apply forces.
            #
            # Need to add a case if adjustSizes ("prevent overlap") is
            # implemented.
            for n in nodes:
                swinging = n.mass * sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
                factor = speed / (1.0 + sqrt(speed * swinging))
                n.x = n.x + (n.dx * factor)
                n.y = n.y + (n.dy * factor)

        barnes_hut_timer.print()
        repulsion_timer.print()
        gravity_timer.print()
        attraction_timer.print()
        return [(n.x, n.y) for n in nodes]

    # A layout for NetworkX.
    #
    # This function returns a NetworkX layout, which is really just a
    # dictionary of node positions (2D X-Y tuples) indexed by the node
    # name.
    def forceatlas2_networkx_layout(self, G, pos=None, iterations=100):
        import networkx
        assert isinstance(G, networkx.classes.graph.Graph), "Not a networkx graph"
        assert isinstance(pos, dict) or (pos is None), "pos must be specified as a dictionary, as in networkx"
        M = networkx.to_scipy_sparse_matrix(G, dtype='f', format='lil')
        if pos is None:
            l = self.forceatlas2(M, pos=None, iterations=iterations)
        else:
            poslist = numpy.asarray([pos[i] for i in G.nodes()])
            l = self.forceatlas2(M, pos=poslist, iterations=iterations)
        return dict(zip(G.nodes(), l))
