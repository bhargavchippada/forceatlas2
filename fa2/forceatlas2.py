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
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm

from . import fa2util

__all__ = ["ForceAtlas2", "Timer"]


def _igraph_to_sparse(graph, weight_attr=None):
    """Convert an undirected igraph graph to a symmetric scipy sparse matrix."""
    from scipy.sparse import csr_matrix

    n = graph.vcount()
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1.0] * len(edges)
    else:
        weights = list(graph.es[weight_attr])

    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights = weights + weights

    if not edges:
        return csr_matrix((n, n))
    return csr_matrix((weights, list(zip(*edges))), shape=(n, n))


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

    New in v1.0.0:

    - N-dimensional layout support via ``dim`` parameter (default 2).
    - ``adjustSizes`` anti-collision mode (Gephi ForceFactory.java parity).
    - ``normalizeEdgeWeights`` and ``invertedEdgeWeightsMode`` parameters.
    - ``inferSettings()`` classmethod for auto-tuning.
    - ``backend`` parameter: ``"auto"``, ``"cython"``, ``"vectorized"``, ``"loop"``.
    - ``store_pos_as`` parameter for NetworkX/igraph wrappers.
    - ``size_attr`` parameter for NetworkX/igraph wrappers.

    Note: ``multiThreaded`` is accepted as a parameter but raises
    ``NotImplementedError``. Contributions welcome.

    Parameters
    ----------
    outboundAttractionDistribution : bool
        Distributes attraction along outbound edges. Hubs attract less and are
        pushed to the borders (aka "Dissuade Hubs" in Gephi).
    linLogMode : bool
        Use logarithmic attraction force. Produces more clustered layouts where
        communities are more tightly grouped.
    adjustSizes : bool
        Prevent node overlap using anti-collision forces (Gephi parity).
    edgeWeightInfluence : float
        How much influence edge weights have. 0 means no influence (all edges
        equal), 1 means normal influence.
    normalizeEdgeWeights : bool
        Min-max normalize edge weights to [0, 1] before applying
        edgeWeightInfluence. Applied after inversion if both are enabled.
    invertedEdgeWeightsMode : bool
        Invert edge weights (w = 1/w) so that stronger connections have
        lower values. Applied before normalization.
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
    dim : int
        Number of layout dimensions. Default 2 (standard 2D layout).
        Use 3 for 3D layouts. Both Cython and pure Python backends support
        arbitrary dimensions.
    backend : str
        Force computation backend. ``"auto"`` (default) uses Cython if compiled,
        otherwise NumPy vectorized. ``"cython"`` or ``"loop"`` forces the
        loop-based path. ``"vectorized"`` uses NumPy broadcasting (O(n^2), no
        Barnes-Hut).
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
        normalizeEdgeWeights: bool = False,
        invertedEdgeWeightsMode: bool = False,
        # Performance
        jitterTolerance: float = 1.0,
        barnesHutOptimize: bool = True,
        barnesHutTheta: float = 1.2,
        multiThreaded: bool = False,
        # Tuning
        scalingRatio: float = 2.0,
        strongGravityMode: bool = False,
        gravity: float = 1.0,
        # Layout
        dim: int = 2,
        # Backend
        backend: str = "auto",
        # Reproducibility
        seed: Optional[int] = None,
        # Log
        verbose: bool = True,
    ):
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
        if not isinstance(dim, int) or dim < 2:
            raise ValueError("dim must be an integer >= 2")
        if backend not in ("auto", "cython", "vectorized", "loop"):
            raise ValueError("backend must be 'auto', 'cython', 'vectorized', or 'loop'")

        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.adjustSizes = adjustSizes
        self.edgeWeightInfluence = edgeWeightInfluence
        self.normalizeEdgeWeights = normalizeEdgeWeights
        self.invertedEdgeWeightsMode = invertedEdgeWeightsMode
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity
        self.dim = dim
        self.backend = backend
        self.seed = seed
        self.verbose = verbose

    @classmethod
    def inferSettings(
        cls,
        G,
        **overrides,
    ) -> "ForceAtlas2":
        """Create a ForceAtlas2 instance with auto-tuned parameters.

        Analyzes graph characteristics (node count, edge count, density) and
        selects appropriate scalingRatio, gravity, barnesHutOptimize, and
        barnesHutTheta. Inspired by Graphology/sigma.js ``inferSettings``.

        Parameters
        ----------
        G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
            The graph to analyze.
        **overrides
            Any ForceAtlas2 parameter to override the inferred values.

        Returns
        -------
        ForceAtlas2
            A configured ForceAtlas2 instance.
        """
        # Extract node/edge counts from any supported graph type
        if isinstance(G, np.ndarray):
            n = G.shape[0]
            m = int(np.count_nonzero(G) / 2)  # symmetric → halve
        elif scipy.sparse.issparse(G):
            n = G.shape[0]
            m = int(G.nnz / 2)
        else:
            n = m = None
            # Try NetworkX
            try:
                import networkx
                if isinstance(G, networkx.Graph):
                    n = G.number_of_nodes()
                    m = G.number_of_edges()
            except ImportError:
                pass
            # Try igraph
            if n is None:
                try:
                    import igraph
                    if isinstance(G, igraph.Graph):
                        n = G.vcount()
                        m = G.ecount()
                except ImportError:
                    pass
            if n is None:
                raise TypeError(
                    "G must be a numpy ndarray, scipy sparse matrix, "
                    "networkx.Graph, or igraph.Graph"
                )

        n = max(n, 1)  # guard against empty graph

        # Heuristics inspired by Graphology/sigma.js inferSettings
        settings = {
            "scalingRatio": 2.0 * n if n > 100 else 10.0,
            "gravity": max(1.0, n / 500.0),
            "barnesHutOptimize": n > 2000,
            "barnesHutTheta": 1.2,
        }

        # For dense graphs, increase gravity to keep layout compact
        max_edges = n * (n - 1) / 2
        if max_edges > 0:
            density = m / max_edges
            if density > 0.5:
                settings["gravity"] = max(settings["gravity"], 5.0)

        # Apply user overrides
        settings.update(overrides)

        return cls(**settings)

    def init(
        self,
        G: Union[np.ndarray, scipy.sparse.spmatrix],
        pos: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
    ) -> tuple[list[Any], list[Any]]:
        """Initialize nodes and edges from an adjacency matrix.

        Parameters
        ----------
        G : numpy.ndarray or scipy.sparse matrix
            Adjacency matrix (must be symmetric for undirected graphs).
        pos : numpy.ndarray, optional
            Initial positions as an (N, dim) array.
        sizes : numpy.ndarray, optional
            Node sizes as a 1-D array of shape (N,). Used with adjustSizes
            to set per-node radii for anti-collision forces.

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
            if pos.shape != (G.shape[0], self.dim):
                raise ValueError(f"pos must have shape ({G.shape[0]}, {self.dim}), got {pos.shape}")

        if sizes is not None:
            if not isinstance(sizes, np.ndarray):
                raise TypeError("sizes must be a numpy ndarray or None")
            if sizes.shape != (G.shape[0],):
                raise ValueError(f"sizes must have shape ({G.shape[0]},), got {sizes.shape}")

        # Warn about self-loops (they inflate node mass without creating edges)
        has_self_loops = False
        if isinstance(G, np.ndarray):
            has_self_loops = bool(np.any(np.diag(G) != 0))
        elif isSparse:
            has_self_loops = bool(G.diagonal().any())
        if has_self_loops:
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
            n = fa2util.Node(dim=self.dim)
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + np.count_nonzero(G[i])
            if pos is None:
                if self.dim == 2:
                    n.x = rng.random()
                    n.y = rng.random()
                else:
                    for d in range(self.dim):
                        n.pos[d] = rng.random()
            else:
                if self.dim == 2:
                    n.x = float(pos[i][0])
                    n.y = float(pos[i][1])
                else:
                    for d in range(self.dim):
                        n.pos[d] = pos[i][d]
            if sizes is not None:
                n.size = float(sizes[i])
            elif self.adjustSizes:
                n.size = 1.0  # Default node radius
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
        sizes: Optional[np.ndarray] = None,
    ) -> list[tuple]:
        """Compute ForceAtlas2 layout from an adjacency matrix.

        Parameters
        ----------
        G : numpy.ndarray or scipy.sparse matrix
            Adjacency matrix (symmetric, for undirected graphs).
        pos : numpy.ndarray, optional
            Initial node positions as an (N, dim) array.
        iterations : int
            Number of layout iterations.
        callbacks : list of callable, optional
            Functions called after each iteration with signature:
            ``callback(iteration, nodes)`` where nodes is the list of Node objects.
        sizes : numpy.ndarray, optional
            Node sizes as a 1-D array of shape (N,). Used with adjustSizes.

        Returns
        -------
        positions : list of tuples
            Final positions for each node. Each tuple has ``dim`` elements.
        """
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos, sizes=sizes)
        self._transform_edge_weights(edges)

        # Determine backend
        use_vectorized = self._should_use_vectorized()

        if use_vectorized:
            return self._run_vectorized(nodes, edges, iterations, callbacks, speed, speedEfficiency)
        else:
            return self._run_loop(nodes, edges, iterations, callbacks, speed, speedEfficiency)

    def _transform_edge_weights(self, edges: list) -> None:
        """Apply inversion and/or normalization to edge weights in-place."""
        if not edges or not (self.invertedEdgeWeightsMode or self.normalizeEdgeWeights):
            return
        if self.invertedEdgeWeightsMode:
            for edge in edges:
                if edge.weight != 0:
                    edge.weight = 1.0 / edge.weight
        if self.normalizeEdgeWeights:
            w_min = min(edge.weight for edge in edges)
            w_max = max(edge.weight for edge in edges)
            w_range = w_max - w_min
            if w_range > 0:
                for edge in edges:
                    edge.weight = (edge.weight - w_min) / w_range
            else:
                for edge in edges:
                    edge.weight = 1.0

    def _should_use_vectorized(self) -> bool:
        """Determine whether to use the vectorized backend."""
        if self.backend == "vectorized":
            return True
        if self.backend in ("cython", "loop"):
            return False
        # "auto": use Cython/loop (with Barnes-Hut) when Cython is compiled.
        # Fall back to vectorized (NumPy) when only pure Python is available,
        # since it's 10-16x faster than pure Python loops.
        if fa2util.__file__.endswith(".py"):
            return True
        return False

    def _run_vectorized(self, nodes, edges, iterations, callbacks, speed, speedEfficiency):
        """Run layout using vectorized NumPy backend (no Barnes-Hut)."""
        from . import fa2util_vectorized as vec

        dim = self.dim
        n_nodes = len(nodes)

        # Convert to arrays
        if dim == 2:
            positions = np.array([(n.x, n.y) for n in nodes], dtype=np.float64)
        else:
            positions = np.array([n.pos for n in nodes], dtype=np.float64)
        masses = np.array([n.mass for n in nodes], dtype=np.float64)
        node_sizes = np.array([n.size for n in nodes], dtype=np.float64)

        # Edge arrays
        edge_sources = np.array([e.node1 for e in edges], dtype=np.int32)
        edge_targets = np.array([e.node2 for e in edges], dtype=np.int32)
        edge_weights = np.array([e.weight for e in edges], dtype=np.float64)

        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = np.mean(masses)

        old_forces = np.zeros_like(positions)
        forces = np.zeros_like(positions)

        niters = range(iterations)
        if self.verbose:
            niters = tqdm(niters)
        for _i in niters:
            old_forces[:] = forces
            forces[:] = 0.0

            # Repulsion (O(n^2) pairwise — no Barnes-Hut in vectorized mode)
            if self.adjustSizes:
                forces += vec.apply_repulsion_adjustSizes(positions, masses, node_sizes, self.scalingRatio)
            else:
                forces += vec.apply_repulsion(positions, masses, self.scalingRatio)

            # Gravity
            forces += vec.apply_gravity(positions, masses, self.gravity, self.scalingRatio,
                                  self.strongGravityMode)

            # Attraction
            forces += vec.apply_attraction(positions, edge_sources, edge_targets, edge_weights,
                                     masses, self.outboundAttractionDistribution,
                                     outboundAttCompensation, self.edgeWeightInfluence,
                                     self.linLogMode, self.adjustSizes, node_sizes)

            # Adjust speed and apply
            positions, speed, speedEfficiency = vec.adjustSpeedAndApplyForces(
                positions, forces, old_forces, masses, speed, speedEfficiency,
                self.jitterTolerance, self.adjustSizes, node_sizes)

            # Callbacks (sync positions back to Node objects)
            if callbacks:
                if dim == 2:
                    for j in range(n_nodes):
                        nodes[j].x = positions[j, 0]
                        nodes[j].y = positions[j, 1]
                else:
                    for j in range(n_nodes):
                        for d in range(dim):
                            nodes[j].pos[d] = positions[j, d]
                for cb in callbacks:
                    cb(_i, nodes)

        return [tuple(row) for row in positions]

    def _run_loop(self, nodes, edges, iterations, callbacks, speed, speedEfficiency):
        """Run layout using the standard loop-based backend (Cython or pure Python)."""
        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = np.mean([n.mass for n in nodes])

        barneshut_timer = Timer(name="BarnesHut Approximation")
        repulsion_timer = Timer(name="Repulsion forces")
        gravity_timer = Timer(name="Gravitational forces")
        attraction_timer = Timer(name="Attraction forces")
        applyforces_timer = Timer(name="AdjustSpeedAndApplyForces step")

        dim = self.dim
        niters = range(iterations)
        if self.verbose:
            niters = tqdm(niters)
        for _i in niters:
            if dim == 2:
                for n in nodes:
                    n.old_dx = n.dx
                    n.old_dy = n.dy
                    n.dx = 0.0
                    n.dy = 0.0
            else:
                for n in nodes:
                    for d in range(dim):
                        n.old_force[d] = n.force[d]
                        n.force[d] = 0.0

            # Barnes Hut optimization
            if self.barnesHutOptimize:
                barneshut_timer.start()
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()
                barneshut_timer.stop()

            # Repulsion forces
            repulsion_timer.start()
            if self.barnesHutOptimize:
                rootRegion.applyForceOnNodes(nodes, self.barnesHutTheta, self.scalingRatio,
                                             adjustSizes=self.adjustSizes)
            else:
                fa2util.apply_repulsion(nodes, self.scalingRatio, adjustSizes=self.adjustSizes)
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
                                  self.linLogMode, adjustSizes=self.adjustSizes)
            attraction_timer.stop()

            # Adjust speeds and apply forces
            applyforces_timer.start()
            values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance,
                                                       adjustSizes=self.adjustSizes)
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

        if dim == 2:
            return [(n.x, n.y) for n in nodes]
        return [tuple(n.pos) for n in nodes]

    def forceatlas2_networkx_layout(
        self,
        G,
        pos: Optional[dict] = None,
        iterations: int = 100,
        weight_attr: Optional[str] = None,
        callbacks: Optional[list[Callable]] = None,
        size_attr: Optional[str] = None,
        store_pos_as: Optional[str] = None,
    ) -> dict:
        """Compute ForceAtlas2 layout for a NetworkX graph.

        Parameters
        ----------
        G : networkx.Graph
            Input graph.
        pos : dict, optional
            Initial positions as ``{node: tuple}``.
        iterations : int
            Number of layout iterations.
        weight_attr : str, optional
            Edge attribute name to use as weight. None means unweighted.
        callbacks : list of callable, optional
            Functions called after each iteration.
        size_attr : str, optional
            Node attribute name for node sizes (used with adjustSizes).
        store_pos_as : str, optional
            If set, stores positions as node attributes under this key.

        Returns
        -------
        positions : dict
            Dictionary of ``{node: tuple}`` positions.
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
        M = networkx.to_scipy_sparse_array(G, dtype=float, format='lil', weight=weight_attr)

        # Extract node sizes if size_attr is provided
        sizes = None
        if size_attr is not None:
            sizes = np.array([G.nodes[n].get(size_attr, 1.0) for n in G.nodes()])

        if pos is None:
            layout = self.forceatlas2(M, pos=None, iterations=iterations, callbacks=callbacks, sizes=sizes)
        else:
            try:
                poslist = np.asarray([pos[i] for i in G.nodes()])
            except KeyError as exc:
                raise ValueError(
                    f"pos is missing an entry for node {exc}. "
                    "pos must contain a position for every node in G."
                ) from exc
            layout = self.forceatlas2(M, pos=poslist, iterations=iterations, callbacks=callbacks, sizes=sizes)
        result = dict(zip(G.nodes(), layout))

        if store_pos_as is not None:
            networkx.set_node_attributes(G, result, store_pos_as)

        return result

    def forceatlas2_igraph_layout(
        self,
        G,
        pos=None,
        iterations: int = 100,
        weight_attr: Optional[str] = None,
        callbacks: Optional[list[Callable]] = None,
        size_attr: Optional[str] = None,
        store_pos_as: Optional[str] = None,
    ) -> Any:
        """Compute ForceAtlas2 layout for an igraph graph.

        Parameters
        ----------
        G : igraph.Graph
            Input graph.
        pos : list or numpy.ndarray, optional
            Initial positions as an (N, dim) array or list of tuples.
        iterations : int
            Number of layout iterations.
        weight_attr : str, optional
            Edge attribute name to use as weight.
        callbacks : list of callable, optional
            Functions called after each iteration.
        size_attr : str, optional
            Vertex attribute name for node sizes (used with adjustSizes).
        store_pos_as : str, optional
            If set, stores positions as vertex attributes under this key.

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

        sizes = None
        if size_attr is not None:
            sizes = np.array(G.vs[size_attr])

        adj = _igraph_to_sparse(G, weight_attr)
        coords = self.forceatlas2(adj, pos=pos, iterations=iterations, callbacks=callbacks, sizes=sizes)

        layout = igraph.layout.Layout(coords, self.dim)

        if store_pos_as is not None:
            G.vs[store_pos_as] = coords

        return layout
