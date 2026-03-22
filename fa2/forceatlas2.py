"""ForceAtlas2 layout algorithm for Python.

The fastest Python implementation of the ForceAtlas2 plugin from Gephi,
intended for use with NetworkX, igraph, or raw adjacency matrices.
Based on the Gephi Java implementation:

    https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java

Currently supports weighted undirected graphs only.

Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
Available under the GPLv3
"""

import random
import time
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm

from . import fa2util

__all__ = ["ForceAtlas2", "Timer"]


class Timer:
    """Simple timer for profiling force computation phases."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self._start_time: Optional[float] = None
        self.total_time = 0.0

    def start(self):
        self._start_time = time.time()

    def stop(self):
        if self._start_time is None:
            return
        self.total_time += time.time() - self._start_time
        self._start_time = None

    def display(self):
        print(f"{self.name} took {self.total_time:.2f} seconds")


class ForceAtlas2:
    """ForceAtlas2 layout algorithm implementation.

    A force-directed graph layout algorithm optimized for quality and performance.
    Supports Barnes-Hut approximation for O(n log n) complexity.

    .. todo:: Implement ``adjustSizes`` (prevent node overlap). Gephi uses
       anti-collision repulsion (distance minus node sizes, with strong constant
       repulsive force when overlapping) and anti-collision attraction (zero
       force when overlapping). See Gephi ForceFactory.java
       ``linRepulsion_antiCollision`` and ``linAttraction_antiCollision``.
       (GitHub issue #7, PRs #9, #36)

    .. todo:: Implement ``multiThreaded`` parallel force computation. Gephi
       parallelizes repulsion + gravity (not attraction) using a thread pool
       with 8*threadCount chunks. Python's GIL limits benefit for CPU-bound
       work, but this could use multiprocessing or release the GIL in Cython.

    .. todo:: Support N-dimensional layout (3D+). The algorithm generalizes
       naturally to N dimensions — replace (x, y) with an N-vector, and
       distance/force calculations extend accordingly. NetworkX's FA2 and
       the Rust ``forceatlas2`` crate already support this. (GitHub issue #14)

    .. todo:: Add ``normalizeEdgeWeights`` parameter (Gephi feature). Min-max
       normalizes edge weights to [0, 1] before applying edgeWeightInfluence.
       Processing order: raw weight -> inversion -> normalization -> exponent.

    .. todo:: Add ``invertedEdgeWeightsMode`` parameter (Gephi feature).
       Inverts edge weights (w = 1/w) so stronger connections have lower values.

    .. todo:: Add ``inferSettings()`` auto-tuning (from Graphology/sigma.js).
       Automatically selects scalingRatio, gravity, and barnesHutTheta based
       on graph size and density.

    .. todo:: Vectorize force computation with NumPy to eliminate Python loops
       in the pure-Python fallback path. NetworkX's FA2 demonstrates this
       approach for significant speedup without Cython.

    Parameters
    ----------
    outboundAttractionDistribution : bool
        Distributes attraction along outbound edges. Hubs attract less and are
        pushed to the borders (aka "Dissuade Hubs" in Gephi).
    linLogMode : bool
        Use logarithmic attraction force. Produces more clustered layouts where
        communities are more tightly grouped.
    adjustSizes : bool
        Prevent node overlap. Not yet implemented.
    edgeWeightInfluence : float
        How much influence edge weights have. 0 means no influence (all edges
        equal), 1 means normal influence.
    jitterTolerance : float
        How much swinging is tolerated. Higher values give more speed but less
        precision. Values above 1 are discouraged.
    barnesHutOptimize : bool
        Use Barnes-Hut approximation for repulsion calculation.
        Reduces complexity from O(n^2) to O(n log n).
    barnesHutTheta : float
        Accuracy parameter for Barnes-Hut. Lower values are more accurate
        but slower. Default 1.2.
    scalingRatio : float
        Repulsion strength. Higher values produce more spread-out graphs.
    strongGravityMode : bool
        Use a stronger gravity model that is distance-independent.
    gravity : float
        Attraction to center. Prevents disconnected components from drifting away.
    seed : int or None
        Random seed for reproducible layouts. If None, layouts are non-deterministic.
    verbose : bool
        Show progress bar and timing information.
    """

    def __init__(
        self,
        # Behavior alternatives
        outboundAttractionDistribution: bool = False,
        linLogMode: bool = False,
        adjustSizes: bool = False,
        edgeWeightInfluence: float = 1.0,
        # Performance
        jitterTolerance: float = 1.0,
        barnesHutOptimize: bool = True,
        barnesHutTheta: float = 1.2,
        multiThreaded: bool = False,
        # Tuning
        scalingRatio: float = 2.0,
        strongGravityMode: bool = False,
        gravity: float = 1.0,
        # Reproducibility
        seed: Optional[int] = None,
        # Log
        verbose: bool = True,
    ):
        if adjustSizes:
            raise NotImplementedError(
                "adjustSizes (prevent overlap) is not yet implemented. "
                "Contributions welcome: https://github.com/bhargavchippada/forceatlas2"
            )
        if multiThreaded:
            raise NotImplementedError(
                "multiThreaded is not yet implemented. "
                "Contributions welcome: https://github.com/bhargavchippada/forceatlas2"
            )
        if scalingRatio <= 0:
            raise ValueError("scalingRatio must be positive")
        if gravity < 0:
            raise ValueError("gravity must be non-negative")
        if barnesHutTheta <= 0:
            raise ValueError("barnesHutTheta must be positive")
        if edgeWeightInfluence < 0:
            raise ValueError("edgeWeightInfluence must be non-negative")
        if jitterTolerance <= 0:
            raise ValueError("jitterTolerance must be positive")

        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.edgeWeightInfluence = edgeWeightInfluence
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity
        self.seed = seed
        self.verbose = verbose

    def init(
        self,
        G: Union[np.ndarray, scipy.sparse.spmatrix],
        pos: Optional[np.ndarray] = None,
    ) -> tuple[list[Any], list[Any]]:
        """Initialize nodes and edges from an adjacency matrix.

        Parameters
        ----------
        G : numpy.ndarray or scipy.sparse matrix
            Adjacency matrix (must be symmetric for undirected graphs).
        pos : numpy.ndarray, optional
            Initial positions as an (N, 2) array.

        Returns
        -------
        nodes : list of Node
        edges : list of Edge
        """
        isSparse = False
        if isinstance(G, np.ndarray):
            if G.shape[0] != G.shape[1]:
                raise ValueError(f"Adjacency matrix is not square: {G.shape}")
            if not np.allclose(G, G.T):
                raise ValueError("Adjacency matrix is not symmetric. Only undirected graphs are supported.")
        elif scipy.sparse.issparse(G):
            if G.shape[0] != G.shape[1]:
                raise ValueError(f"Adjacency matrix is not square: {G.shape}")
            diff = G - G.T
            if diff.nnz > 0 and not np.allclose(diff.data, 0):
                raise ValueError("Adjacency matrix is not symmetric. Only undirected graphs are supported.")
            G = G.tolil()
            isSparse = True
        else:
            raise TypeError(f"G must be a numpy ndarray or scipy sparse matrix, got {type(G).__name__}")

        if pos is not None:
            if not isinstance(pos, np.ndarray):
                raise TypeError("pos must be a numpy ndarray or None")
            if pos.shape != (G.shape[0], 2):
                raise ValueError(f"pos must have shape ({G.shape[0]}, 2), got {pos.shape}")

        # Warn about self-loops (they inflate node mass without creating edges)
        if isinstance(G, np.ndarray):
            if np.any(np.diag(G) != 0):
                import warnings
                warnings.warn(
                    "Adjacency matrix has non-zero diagonal (self-loops). "
                    "Self-loops inflate node mass but are excluded from edges, "
                    "which may produce unexpected layouts.",
                    stacklevel=2,
                )
        elif isSparse and G.diagonal().any():
            import warnings
            warnings.warn(
                "Adjacency matrix has non-zero diagonal (self-loops). "
                "Self-loops inflate node mass but are excluded from edges, "
                "which may produce unexpected layouts.",
                stacklevel=2,
            )

        # Seed RNG for reproducibility
        rng = random.Random(self.seed)

        # Build node list
        nodes = []
        for i in range(G.shape[0]):
            n = fa2util.Node()
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + np.count_nonzero(G[i])
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = rng.random()
                n.y = rng.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Build edge list
        edges = []
        es = np.asarray(G.nonzero()).T
        for e in es:
            if e[1] <= e[0]:
                continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]
            edge.node2 = e[1]
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    def forceatlas2(
        self,
        G: Union[np.ndarray, scipy.sparse.spmatrix],
        pos: Optional[np.ndarray] = None,
        iterations: int = 100,
        callbacks: Optional[list[Callable]] = None,
    ) -> list[tuple[float, float]]:
        """Compute ForceAtlas2 layout from an adjacency matrix.

        Parameters
        ----------
        G : numpy.ndarray or scipy.sparse matrix
            Adjacency matrix (symmetric, for undirected graphs).
        pos : numpy.ndarray, optional
            Initial node positions as an (N, 2) array.
        iterations : int
            Number of layout iterations.
        callbacks : list of callable, optional
            Functions called after each iteration with signature:
            ``callback(iteration, nodes)`` where nodes is the list of Node objects.

        Returns
        -------
        positions : list of (x, y) tuples
            Final positions for each node.
        """
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos)
        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = np.mean([n.mass for n in nodes])

        barneshut_timer = Timer(name="BarnesHut Approximation")
        repulsion_timer = Timer(name="Repulsion forces")
        gravity_timer = Timer(name="Gravitational forces")
        attraction_timer = Timer(name="Attraction forces")
        applyforces_timer = Timer(name="AdjustSpeedAndApplyForces step")

        niters = range(iterations)
        if self.verbose:
            niters = tqdm(niters)
        for _i in niters:
            for n in nodes:
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0

            # Barnes Hut optimization
            if self.barnesHutOptimize:
                barneshut_timer.start()
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()
                barneshut_timer.stop()

            # Repulsion forces
            repulsion_timer.start()
            if self.barnesHutOptimize:
                rootRegion.applyForceOnNodes(nodes, self.barnesHutTheta, self.scalingRatio)
            else:
                fa2util.apply_repulsion(nodes, self.scalingRatio)
            repulsion_timer.stop()

            # Gravitational forces
            gravity_timer.start()
            fa2util.apply_gravity(nodes, self.gravity, scalingRatio=self.scalingRatio,
                                  useStrongGravity=self.strongGravityMode)
            gravity_timer.stop()

            # Attraction forces
            attraction_timer.start()
            fa2util.apply_attraction(nodes, edges, self.outboundAttractionDistribution,
                                     outboundAttCompensation, self.edgeWeightInfluence,
                                     self.linLogMode)
            attraction_timer.stop()

            # Adjust speeds and apply forces
            applyforces_timer.start()
            values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance)
            speed = values['speed']
            speedEfficiency = values['speedEfficiency']
            applyforces_timer.stop()

            # Invoke callbacks
            if callbacks:
                for cb in callbacks:
                    cb(_i, nodes)

        if self.verbose:
            if self.barnesHutOptimize:
                barneshut_timer.display()
            repulsion_timer.display()
            gravity_timer.display()
            attraction_timer.display()
            applyforces_timer.display()

        return [(n.x, n.y) for n in nodes]

    def forceatlas2_networkx_layout(
        self,
        G,
        pos: Optional[dict] = None,
        iterations: int = 100,
        weight_attr: Optional[str] = None,
        callbacks: Optional[list[Callable]] = None,
    ) -> dict:
        """Compute ForceAtlas2 layout for a NetworkX graph.

        Parameters
        ----------
        G : networkx.Graph
            Input graph.
        pos : dict, optional
            Initial positions as ``{node: (x, y)}``.
        iterations : int
            Number of layout iterations.
        weight_attr : str, optional
            Edge attribute name to use as weight. None means unweighted.
        callbacks : list of callable, optional
            Functions called after each iteration.

        Returns
        -------
        positions : dict
            Dictionary of ``{node: (x, y)}`` positions.
        """
        try:
            import networkx
        except ImportError:
            raise ImportError(
                "networkx is required for forceatlas2_networkx_layout. "
                "Install it with: pip install 'fa2[networkx]'"
            ) from None

        try:
            import cynetworkx
        except ImportError:
            cynetworkx = None

        if not (
            isinstance(G, networkx.classes.graph.Graph)
            or (cynetworkx and isinstance(G, cynetworkx.classes.graph.Graph))
        ):
            raise TypeError("G must be a networkx Graph")
        if G.is_directed():
            raise ValueError(
                "Only undirected NetworkX graphs are supported. "
                "Convert with G.to_undirected() first."
            )
        if pos is not None and not isinstance(pos, dict):
            raise TypeError("pos must be a dictionary or None")

        # NetworkX 2.7+ has to_scipy_sparse_array; 3.x removed to_scipy_sparse_matrix
        M = networkx.to_scipy_sparse_array(G, dtype='f', format='lil', weight=weight_attr)

        if pos is None:
            layout = self.forceatlas2(M, pos=None, iterations=iterations, callbacks=callbacks)
        else:
            try:
                poslist = np.asarray([pos[i] for i in G.nodes()])
            except KeyError as exc:
                raise ValueError(
                    f"pos is missing an entry for node {exc}. "
                    "pos must contain a position for every node in G."
                ) from exc
            layout = self.forceatlas2(M, pos=poslist, iterations=iterations, callbacks=callbacks)
        return dict(zip(G.nodes(), layout))

    def forceatlas2_igraph_layout(
        self,
        G,
        pos=None,
        iterations: int = 100,
        weight_attr: Optional[str] = None,
        callbacks: Optional[list[Callable]] = None,
    ):
        """Compute ForceAtlas2 layout for an igraph graph.

        Parameters
        ----------
        G : igraph.Graph
            Input graph.
        pos : list or numpy.ndarray, optional
            Initial positions as an (N, 2) array or list of (x, y) tuples.
        iterations : int
            Number of layout iterations.
        weight_attr : str, optional
            Edge attribute name to use as weight.
        callbacks : list of callable, optional
            Functions called after each iteration.

        Returns
        -------
        layout : igraph.Layout
            An igraph Layout object.
        """
        try:
            import igraph
        except ImportError:
            raise ImportError(
                "igraph is required for forceatlas2_igraph_layout. "
                "Install it with: pip install 'fa2[igraph]'"
            ) from None
        from scipy.sparse import csr_matrix

        def to_sparse(graph, weight_attr=None):
            n = graph.vcount()
            edges = graph.get_edgelist()
            if weight_attr is None:
                weights = [1] * len(edges)
            else:
                weights = list(graph.es[weight_attr])

            if not graph.is_directed():
                edges.extend([(v, u) for u, v in edges])
                weights = weights + weights

            if not edges:
                return csr_matrix((n, n))
            return csr_matrix((weights, list(zip(*edges))), shape=(n, n))

        if not isinstance(G, igraph.Graph):
            raise TypeError("G must be an igraph Graph")
        if G.is_directed():
            raise ValueError(
                "Only undirected igraph graphs are supported. "
                "Convert with G.as_undirected() first."
            )
        if pos is not None and not isinstance(pos, (list, np.ndarray)):
            raise TypeError("pos must be a list, numpy array, or None")

        if isinstance(pos, list):
            pos = np.array(pos)

        adj = to_sparse(G, weight_attr)
        coords = self.forceatlas2(adj, pos=pos, iterations=iterations, callbacks=callbacks)

        return igraph.layout.Layout(coords, 2)
