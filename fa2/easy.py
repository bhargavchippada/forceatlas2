"""Simplified API for ForceAtlas2 — no numpy knowledge required.

One-call functions that accept edge lists, adjacency dicts, or any
supported graph type and return plain Python dicts.

Example::

    from fa2.easy import layout, visualize

    positions = layout([("A", "B"), ("B", "C"), ("A", "C")])
    visualize([("A", "B"), ("B", "C")], output="png", path="graph.png")
"""

from typing import Optional, Union

import numpy as np
from scipy.sparse import coo_matrix

from .forceatlas2 import ForceAtlas2

__all__ = ["layout", "visualize"]

# Mode presets mapping mode name → ForceAtlas2 kwargs
_MODE_PRESETS = {
    "default": {},
    "community": {"linLogMode": True, "outboundAttractionDistribution": True},
    "hub-dissuade": {"outboundAttractionDistribution": True, "strongGravityMode": True},
    "compact": {"gravity": 5.0, "scalingRatio": 1.0},
}


def _parse_edges(edges):
    """Parse flexible edge input into (node_list, sparse_matrix, weights_provided).

    Accepted formats:
    - List of tuples: ``[(0, 1), (1, 2)]`` (unweighted)
    - List of 3-tuples: ``[(0, 1, 5.0), (1, 2, 1.0)]`` (weighted)
    - List of dicts: ``[{"source": "A", "target": "B", "weight": 5.0}, ...]``
    - Adjacency dict: ``{"A": ["B", "C"], "B": ["A"]}``

    Returns (node_list, sparse_matrix).
    """
    if isinstance(edges, dict):
        # Adjacency dict: {"A": ["B", "C"], ...}
        nodes = set()
        edge_list = []
        for src, targets in edges.items():
            nodes.add(src)
            for tgt in targets:
                nodes.add(tgt)
                edge_list.append((src, tgt, 1.0))
        node_list = sorted(nodes)
        return _edges_to_sparse(node_list, edge_list)

    if not edges:
        return [], coo_matrix((0, 0))

    first = edges[0]

    if isinstance(first, dict):
        # List of dicts: [{"source": "A", "target": "B", "weight": 5.0}, ...]
        edge_list = []
        nodes = set()
        for e in edges:
            src = e["source"]
            tgt = e["target"]
            w = float(e.get("weight", 1.0))
            nodes.add(src)
            nodes.add(tgt)
            edge_list.append((src, tgt, w))
        node_list = sorted(nodes)
        return _edges_to_sparse(node_list, edge_list)

    # List of tuples
    edge_list = []
    nodes = set()
    for e in edges:
        if len(e) == 2:
            src, tgt = e
            w = 1.0
        elif len(e) == 3:
            src, tgt, w = e
            w = float(w)
        else:
            raise ValueError(f"Edge must be a 2-tuple or 3-tuple, got {len(e)}-tuple: {e}")
        nodes.add(src)
        nodes.add(tgt)
        edge_list.append((src, tgt, w))
    node_list = sorted(nodes)
    return _edges_to_sparse(node_list, edge_list)


def _parse_edge_list(edges):
    """Parse edges into a list of (src, tgt, weight) triples with original IDs."""
    if isinstance(edges, dict):
        result = []
        for src, targets in edges.items():
            for tgt in targets:
                result.append((src, tgt, 1.0))
        return result

    if not edges:
        return []

    first = edges[0]
    if isinstance(first, dict):
        return [(e["source"], e["target"], float(e.get("weight", 1.0))) for e in edges]

    result = []
    for e in edges:
        if len(e) == 2:
            result.append((e[0], e[1], 1.0))
        elif len(e) >= 3:
            result.append((e[0], e[1], float(e[2])))
    return result


def _edges_to_sparse(node_list, edge_list):
    """Convert node list + edge triples to a symmetric sparse matrix."""
    if not node_list:
        return [], coo_matrix((0, 0))

    n = len(node_list)
    node_index = {node: i for i, node in enumerate(node_list)}

    rows, cols, data = [], [], []
    seen = set()
    for src, tgt, w in edge_list:
        si, ti = node_index[src], node_index[tgt]
        if si == ti:
            continue  # skip self-loops
        key = (min(si, ti), max(si, ti))
        if key in seen:
            continue
        seen.add(key)
        rows.extend([si, ti])
        cols.extend([ti, si])
        data.extend([w, w])

    G = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return node_list, G


def layout(
    edges: Union[list, dict],
    iterations: int = 100,
    dim: int = 2,
    mode: str = "default",
    seed: Optional[int] = None,
    node_sizes: Optional[dict] = None,
    node_positions: Optional[dict] = None,
) -> dict:
    """Compute a graph layout from an edge list. No numpy required.

    Parameters
    ----------
    edges : list or dict
        Graph edges in any of these formats:

        - Edge tuples: ``[(0, 1), (1, 2)]`` or ``[("A", "B", 5.0), ...]``
        - Edge dicts: ``[{"source": "A", "target": "B", "weight": 5.0}, ...]``
        - Adjacency dict: ``{"A": ["B", "C"], "B": ["A", "D"]}``
    iterations : int
        Number of layout iterations. Default 100.
    dim : int
        Layout dimensions. Default 2.
    mode : str
        Layout preset: ``"default"``, ``"community"`` (LinLog + dissuade hubs),
        ``"hub-dissuade"`` (hubs to borders), ``"compact"`` (higher gravity).
    seed : int, optional
        Random seed for reproducibility.
    node_sizes : dict, optional
        Node radii as ``{node_id: float}``. Enables anti-collision.
    node_positions : dict, optional
        Initial positions as ``{node_id: (x, y, ...)}``.

    Returns
    -------
    dict
        Positions as ``{node_id: (x, y, ...)}``.
    """
    if mode not in _MODE_PRESETS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {list(_MODE_PRESETS.keys())}")

    node_list, G = _parse_edges(edges)

    if not node_list:
        return {}

    # Build FA2 kwargs from mode preset
    fa2_kwargs = {
        "dim": dim,
        "seed": seed,
        "verbose": False,
        **_MODE_PRESETS[mode],
    }

    # Handle node sizes → adjustSizes
    sizes_array = None
    if node_sizes is not None:
        fa2_kwargs["adjustSizes"] = True
        sizes_array = np.array([float(node_sizes.get(n, 1.0)) for n in node_list])

    # Handle initial positions
    pos_array = None
    if node_positions is not None:
        pos_array = np.array([
            node_positions.get(n, tuple(0.0 for _ in range(dim)))
            for n in node_list
        ], dtype=np.float64)

    # Use inferSettings for auto-tuning, then override with mode preset
    fa2 = ForceAtlas2.inferSettings(G, **fa2_kwargs)

    positions = fa2.forceatlas2(G, pos=pos_array, iterations=iterations, sizes=sizes_array)

    return {node: pos for node, pos in zip(node_list, positions)}


def visualize(
    edges: Union[list, dict],
    iterations: int = 100,
    dim: int = 2,
    mode: str = "default",
    output: str = "matplotlib",
    path: Optional[str] = None,
    seed: Optional[int] = None,
    title: Optional[str] = None,
    **kwargs,
):
    """Layout a graph and render it in one call.

    Parameters
    ----------
    edges : list or dict
        Graph edges (same formats as ``layout()``).
    iterations : int
        Number of layout iterations. Default 100.
    dim : int
        Layout dimensions. Default 2.
    mode : str
        Layout preset (see ``layout()``).
    output : str
        Output format: ``"matplotlib"`` (returns Figure), ``"png"``,
        ``"svg"``, ``"json"``.
    path : str, optional
        File path to write output. If None, returns the result.
    seed : int, optional
        Random seed for reproducibility.
    title : str, optional
        Title for the plot (image outputs only).
    **kwargs
        Extra arguments passed to ``plot_layout()`` for image outputs.

    Returns
    -------
    matplotlib.figure.Figure, bytes, or dict
        Depends on ``output`` format.
    """
    positions = layout(edges, iterations=iterations, dim=dim, mode=mode, seed=seed)

    if not positions:
        return {} if output == "json" else None

    # Build a NetworkX graph for viz (it handles node IDs natively)
    try:
        import networkx as nx
        G_viz = nx.Graph()
        node_list, G_sparse = _parse_edges(edges)
        G_viz.add_nodes_from(node_list)
        # Re-parse edges to get original IDs
        parsed = _parse_edge_list(edges)
        for src, tgt, w in parsed:
            G_viz.add_edge(src, tgt, weight=w)
    except ImportError:
        # Fallback: use sparse matrix with integer positions
        node_list, G_sparse = _parse_edges(edges)
        G_viz = G_sparse
        # Remap positions to integer keys
        positions = {i: positions[n] for i, n in enumerate(node_list)}

    if output == "json":
        from .viz import export_layout
        return export_layout(G_viz, positions, fmt="json", path=path)

    if output in ("png", "svg"):
        from .viz import export_layout
        return export_layout(G_viz, positions, fmt=output, path=path, title=title, **kwargs)

    if output == "matplotlib":
        from .viz import plot_layout
        return plot_layout(G_viz, positions, title=title, **kwargs)

    raise ValueError(f"Unknown output format {output!r}. Use 'matplotlib', 'png', 'svg', or 'json'.")
