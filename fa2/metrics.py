"""Layout quality metrics for evaluating graph visualizations.

Provides quantitative measures to compare parameter settings, iteration
counts, or different layouts of the same graph.
"""

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import pdist, squareform

__all__ = ["stress", "edge_crossing_count", "neighborhood_preservation"]


def _to_position_array(positions, node_list=None):
    """Convert positions (dict, list, or ndarray) to an (N, dim) array."""
    if isinstance(positions, dict):
        if node_list is None:
            node_list = sorted(positions.keys(), key=str)
        return np.array([positions[n] for n in node_list], dtype=np.float64), node_list
    if isinstance(positions, list):
        return np.array(positions, dtype=np.float64), None
    return np.asarray(positions, dtype=np.float64), None


def _extract_graph(G):
    """Extract node_list and sparse/dense adjacency from any supported graph type.

    Returns (adj, node_list) where adj is scipy sparse or numpy array.
    """
    node_list = None
    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            node_list = list(G.nodes())
            adj = nx.to_scipy_sparse_array(G, nodelist=node_list, weight="weight")
            return adj, node_list
    except ImportError:
        pass

    try:
        import igraph
        if isinstance(G, igraph.Graph):
            node_list = list(range(G.vcount()))
            from fa2.forceatlas2 import _igraph_to_sparse
            adj = _igraph_to_sparse(G)
            return adj, node_list
    except ImportError:
        pass

    # numpy or scipy sparse
    return G, node_list


def stress(G, positions):
    """Compute normalized stress of a layout.

    Stress measures how well pairwise Euclidean distances in the layout
    preserve graph-theoretic distances (hop count). Lower is better.

    Uses unweighted shortest-path distance (hop count) as the graph-theoretic
    distance. Edge weights are ignored; only graph topology is used.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
        Adjacency matrix or graph object.
    positions : dict, list, or numpy.ndarray
        Node positions as ``{node: (x, y, ...)}``, list of tuples, or
        ``(N, dim)`` array.

    Returns
    -------
    float
        Normalized stress value. 0.0 = perfect distance preservation.
    """
    from scipy.sparse.csgraph import shortest_path

    adj, node_list = _extract_graph(G)
    pos, _ = _to_position_array(positions, node_list)
    n = pos.shape[0]

    if n < 2:
        return 0.0

    adj_n = adj.shape[0] if hasattr(adj, 'shape') else len(adj)
    if pos.shape[0] != adj_n:
        raise ValueError(f"positions has {pos.shape[0]} entries but graph has {adj_n} nodes")

    # Graph distances (shortest path, hop count)
    graph_dist = shortest_path(adj, directed=False, unweighted=True)
    # Replace inf (disconnected) with max finite off-diagonal distance + 1
    np.fill_diagonal(graph_dist, np.inf)  # exclude self-distance from max
    finite_mask = np.isfinite(graph_dist)
    if not finite_mask.all():
        max_d = graph_dist[finite_mask].max() if finite_mask.any() else 1.0
        graph_dist = np.where(finite_mask, graph_dist, max_d + 1.0)
    np.fill_diagonal(graph_dist, 0.0)  # restore diagonal for squareform

    # Layout distances using pdist (memory-efficient: returns vector, not matrix)
    layout_dist_vec = pdist(pos)
    graph_dist_vec = squareform(graph_dist, checks=False)

    # Normalize layout distances to same scale as graph distances
    graph_mean = graph_dist_vec.mean()
    layout_mean = layout_dist_vec.mean()
    if layout_mean > 0:
        layout_dist_vec = layout_dist_vec * (graph_mean / layout_mean)

    # Compute stress (denom > 0 guaranteed for n >= 2 since all distances >= 1)
    denom = np.dot(graph_dist_vec, graph_dist_vec)
    diff = layout_dist_vec - graph_dist_vec
    return float(np.dot(diff, diff) / denom)


def edge_crossing_count(G, positions):
    """Count the number of edge crossings in a 2D layout.

    Two edges cross if their line segments properly intersect (excluding
    shared endpoints). Only works for 2D layouts.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
        Adjacency matrix or graph object.
    positions : dict, list, or numpy.ndarray
        Node positions (must be 2D).

    Returns
    -------
    int
        Number of edge crossings.

    Raises
    ------
    ValueError
        If positions are not 2D.
    """
    adj, node_list = _extract_graph(G)
    pos, _ = _to_position_array(positions, node_list)

    if pos.shape[1] != 2:
        raise ValueError("edge_crossing_count only works for 2D layouts")

    # Extract edge list from sparse or dense (avoid todense for sparse)
    if issparse(adj):
        rows, cols = adj.nonzero()
    else:
        rows, cols = np.nonzero(np.asarray(adj))

    edges = [(int(r), int(c)) for r, c in zip(rows, cols) if c > r]

    n_edges = len(edges)
    crossings = 0

    for i in range(n_edges):
        a1, a2 = edges[i]
        p1, p2 = pos[a1], pos[a2]
        for j in range(i + 1, n_edges):
            b1, b2 = edges[j]
            # Skip if edges share an endpoint
            if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                continue
            p3, p4 = pos[b1], pos[b2]
            if _segments_intersect(p1, p2, p3, p4):
                crossings += 1

    return crossings


def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 properly intersects p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _cross(a, b, c):
    """2D cross product of vectors (b-a) and (c-a)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def neighborhood_preservation(G, positions, k=10):
    """Measure how well spatial neighbors match graph neighbors.

    For each node, compares its k-nearest graph neighbors (by hop count)
    with its k-nearest spatial neighbors. Returns the average overlap
    fraction.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
        Adjacency matrix or graph object.
    positions : dict, list, or numpy.ndarray
        Node positions.
    k : int
        Number of neighbors to compare. Must be positive.
        Capped at ``n - 1``.

    Returns
    -------
    float
        Neighborhood preservation score in [0, 1]. 1.0 = perfect.
        Isolated nodes (unreachable from all others) are assigned the
        maximum penalty distance and contribute 0 to the score.

    Raises
    ------
    ValueError
        If k <= 0.
    """
    from scipy.sparse.csgraph import shortest_path

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    adj, node_list = _extract_graph(G)
    pos, _ = _to_position_array(positions, node_list)
    n = pos.shape[0]

    if n < 2:
        return 1.0

    adj_n = adj.shape[0] if hasattr(adj, 'shape') else len(adj)
    if pos.shape[0] != adj_n:
        raise ValueError(f"positions has {pos.shape[0]} entries but graph has {adj_n} nodes")

    k = min(k, n - 1)

    # Graph distances (hop count)
    graph_dist = shortest_path(adj, directed=False, unweighted=True)
    finite_mask = np.isfinite(graph_dist)
    if not finite_mask.all():
        max_d = graph_dist[finite_mask].max() if finite_mask.any() else 1.0
        graph_dist = np.where(finite_mask, graph_dist, max_d + 1.0)

    # Layout distances — compute per-row to avoid O(n²) memory for large graphs
    total_overlap = 0.0
    for i in range(n):
        # k-nearest graph neighbors (exclude self)
        graph_neighbors = set(np.argsort(graph_dist[i])[1:k + 1])
        # k-nearest spatial neighbors (exclude self)
        dists_i = np.sum((pos - pos[i]) ** 2, axis=1)
        layout_neighbors = set(np.argsort(dists_i)[1:k + 1])
        total_overlap += len(graph_neighbors & layout_neighbors) / k

    return float(total_overlap / n)
