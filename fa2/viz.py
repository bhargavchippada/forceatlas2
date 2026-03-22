"""Visualization and export for ForceAtlas2 layouts.

Provides one-call rendering from graph + positions to matplotlib figures
or exported files (PNG, SVG, JSON, GEXF, GraphML).

Requires matplotlib: ``pip install fa2[viz]``
"""

import json
from typing import Optional, Union

import numpy as np
from scipy.sparse import issparse

__all__ = ["plot_layout", "export_layout"]


def _resolve_graph_and_positions(G, positions):
    """Normalize graph and positions to common formats.

    Returns (adj_or_nx, pos_array, node_list, edges, node_index).
    node_index is a dict mapping node ID → array index.
    """
    node_list = None
    edges = []

    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            node_list = list(G.nodes())
            edges = list(G.edges(data=True))
            if isinstance(positions, dict):
                pos_array = np.array([positions[n] for n in node_list], dtype=np.float64)
            else:
                pos_array = np.asarray(positions, dtype=np.float64)
            node_index = {node: i for i, node in enumerate(node_list)}
            return G, pos_array, node_list, edges, node_index
    except ImportError:
        pass

    try:
        import igraph
        if isinstance(G, igraph.Graph):
            node_list = list(range(G.vcount()))
            edges = [(e.source, e.target, {"weight": e["weight"] if "weight" in e.attributes() else 1.0})
                     for e in G.es]
            if hasattr(positions, 'coords'):  # igraph.Layout
                pos_array = np.array(positions.coords, dtype=np.float64)
            elif isinstance(positions, list):
                pos_array = np.array(positions, dtype=np.float64)
            else:
                pos_array = np.asarray(positions, dtype=np.float64)
            node_index = {node: i for i, node in enumerate(node_list)}
            return G, pos_array, node_list, edges, node_index
    except ImportError:
        pass

    # Raw matrix — avoid todense for edge extraction
    if issparse(G):
        adj_dense = None
        n = G.shape[0]
        rows, cols = G.nonzero()
    else:
        adj_dense = np.asarray(G)
        n = adj_dense.shape[0]
        rows, cols = np.nonzero(adj_dense)

    node_list = list(range(n))

    if isinstance(positions, dict):
        pos_array = np.array([positions[i] for i in node_list], dtype=np.float64)
    elif isinstance(positions, list):
        pos_array = np.array(positions, dtype=np.float64)
    else:
        pos_array = np.asarray(positions, dtype=np.float64)

    for r, c in zip(rows, cols):
        if c > r:
            w = float(G[r, c]) if adj_dense is None else float(adj_dense[r, c])
            edges.append((r, c, {"weight": w}))

    node_index = {node: i for i, node in enumerate(node_list)}
    return G, pos_array, node_list, edges, node_index


def plot_layout(
    G,
    positions,
    node_color: Optional[Union[str, list, np.ndarray]] = None,
    node_size: Optional[Union[float, list, np.ndarray]] = None,
    edge_alpha: float = 0.15,
    node_alpha: float = 0.8,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    color_by_degree: bool = False,
    cmap: str = "viridis",
    show_labels: bool = False,
    ax=None,
):
    """Render a graph layout as a matplotlib figure.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
        The graph.
    positions : dict, list, numpy.ndarray, or igraph.Layout
        Node positions from ``ForceAtlas2.forceatlas2()`` or wrappers.
    node_color : str, list, or ndarray, optional
        Node colors. If None and ``color_by_degree=True``, colors by degree.
        Can be a single color string, a list of colors, or numeric values
        for colormap mapping.
    node_size : float, list, or ndarray, optional
        Node sizes. If None, sizes are proportional to degree.
    edge_alpha : float
        Edge transparency (0-1). Default 0.15.
    node_alpha : float
        Node transparency (0-1). Default 0.8.
    figsize : tuple
        Figure size in inches. Default (10, 8).
    title : str, optional
        Figure title.
    color_by_degree : bool
        Color nodes by degree using colormap. Default False.
    cmap : str
        Matplotlib colormap name. Default "viridis".
    show_labels : bool
        Show node labels. Default False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_layout. "
            "Install it with: pip install 'fa2[viz]'"
        ) from None

    _, pos_array, node_list, edges, node_index = _resolve_graph_and_positions(G, positions)
    n = len(node_list)
    dim = pos_array.shape[1]

    if dim not in (2, 3):
        raise ValueError(f"plot_layout supports 2D and 3D layouts, got dim={dim}")

    # Compute degrees using lookup dict
    degrees = np.zeros(n)
    for src, tgt, _ in edges:
        degrees[node_index[src]] += 1
        degrees[node_index[tgt]] += 1

    # Default node sizes proportional to degree
    if node_size is None:
        node_size = 20 + degrees * 5
    if isinstance(node_size, (int, float)):
        node_size = [node_size] * n

    # Default node colors
    if node_color is None:
        if color_by_degree:
            node_color = degrees
        else:
            node_color = "steelblue"

    if dim == 3:
        return _plot_3d(pos_array, node_list, edges, node_index, node_color, node_size,
                        edge_alpha, node_alpha, figsize, title, cmap, show_labels, ax)

    # 2D plot
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Draw edges
    for src, tgt, data in edges:
        si, ti = node_index[src], node_index[tgt]
        ax.plot([pos_array[si, 0], pos_array[ti, 0]],
                [pos_array[si, 1], pos_array[ti, 1]],
                color="gray", alpha=edge_alpha, linewidth=0.5, zorder=1)

    # Draw nodes — only pass cmap when color is numeric (avoids matplotlib warning)
    scatter_kw = dict(s=node_size, alpha=node_alpha, zorder=2, edgecolors="white", linewidths=0.3)
    _is_numeric = isinstance(node_color, np.ndarray) or (
        isinstance(node_color, list) and node_color and isinstance(node_color[0], (int, float))
    )
    if _is_numeric:
        scatter_kw.update(c=node_color, cmap=cmap)
    else:
        scatter_kw["color"] = node_color
    ax.scatter(pos_array[:, 0], pos_array[:, 1], **scatter_kw)

    if show_labels:
        for i, label in enumerate(node_list):
            ax.annotate(str(label), (pos_array[i, 0], pos_array[i, 1]),
                        fontsize=7, ha="center", va="center")

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    if created_fig:
        fig.tight_layout()

    return fig


def _plot_3d(pos_array, node_list, edges, node_index, node_color, node_size,
             edge_alpha, node_alpha, figsize, title, cmap, show_labels, ax):
    """Render a 3D layout."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        if getattr(ax, "name", None) != "3d":
            raise ValueError("For 3D layouts, ax must be a 3D Axes (projection='3d')")
        fig = ax.figure

    for src, tgt, _ in edges:
        si, ti = node_index[src], node_index[tgt]
        ax.plot([pos_array[si, 0], pos_array[ti, 0]],
                [pos_array[si, 1], pos_array[ti, 1]],
                [pos_array[si, 2], pos_array[ti, 2]],
                color="gray", alpha=edge_alpha, linewidth=0.5)

    scatter_kw = dict(s=node_size, alpha=node_alpha, edgecolors="white", linewidths=0.3)
    _is_numeric = isinstance(node_color, np.ndarray) or (
        isinstance(node_color, list) and node_color and isinstance(node_color[0], (int, float))
    )
    if _is_numeric:
        scatter_kw.update(c=node_color, cmap=cmap)
    else:
        scatter_kw["color"] = node_color
    ax.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], **scatter_kw)

    if title:
        ax.set_title(title)
    ax.set_axis_off()

    return fig


def export_layout(
    G,
    positions,
    fmt: str = "json",
    path: Optional[str] = None,
    **plot_kwargs,
) -> Optional[bytes]:
    """Export a graph layout to various formats.

    Parameters
    ----------
    G : numpy.ndarray, scipy.sparse matrix, networkx.Graph, or igraph.Graph
        The graph.
    positions : dict, list, numpy.ndarray, or igraph.Layout
        Node positions.
    fmt : str
        Output format: ``"json"``, ``"png"``, ``"svg"``, ``"gexf"``, ``"graphml"``.
    path : str, optional
        File path to write to. If None, returns bytes (for ``"png"``,
        ``"svg"``) or a dict (for ``"json"``).
    **plot_kwargs
        Additional arguments passed to ``plot_layout()`` for image formats.

    Returns
    -------
    bytes, dict, or None
        Content if ``path`` is None, otherwise writes to file and returns None.
    """
    fmt = fmt.lower()

    if fmt == "json":
        return _export_json(G, positions, path)
    elif fmt in ("png", "svg"):
        return _export_image(G, positions, fmt, path, **plot_kwargs)
    elif fmt in ("gexf", "graphml"):
        return _export_graph_format(G, positions, fmt, path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json', 'png', 'svg', 'gexf', or 'graphml'.")


def _export_json(G, positions, path):
    """Export to D3.js/Sigma.js compatible JSON."""
    _, pos_array, node_list, edges, _ = _resolve_graph_and_positions(G, positions)

    nodes_json = []
    for i, node_id in enumerate(node_list):
        # Convert numpy types to native Python for JSON serialization
        nid = int(node_id) if isinstance(node_id, np.integer) else node_id
        entry = {"id": nid}
        for d in range(pos_array.shape[1]):
            key = ["x", "y", "z"][d] if d < 3 else f"d{d}"
            entry[key] = float(pos_array[i, d])
        nodes_json.append(entry)

    edges_json = []
    for src, tgt, data in edges:
        src_id = int(src) if isinstance(src, np.integer) else src
        tgt_id = int(tgt) if isinstance(tgt, np.integer) else tgt
        edge = {"source": src_id, "target": tgt_id}
        w = data.get("weight", 1.0)
        if w != 1.0:
            edge["weight"] = float(w)
        edges_json.append(edge)

    result = {"nodes": nodes_json, "edges": edges_json}

    if path is not None:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        return None
    return result


def _export_image(G, positions, fmt, path, **plot_kwargs):
    """Export to PNG or SVG."""
    import io

    import matplotlib.pyplot as plt

    # Image export always creates its own figure — discard any caller ax
    plot_kwargs.pop("ax", None)
    fig = plot_layout(G, positions, **plot_kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    data = buf.read()

    if path is not None:
        with open(path, "wb") as f:
            f.write(data)
        return None
    return data


def _export_graph_format(G, positions, fmt, path):
    """Export to GEXF or GraphML with positions embedded."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            f"networkx is required for {fmt} export. "
            "Install it with: pip install 'fa2[networkx]'"
        ) from None

    _, pos_array, node_list, edges, _ = _resolve_graph_and_positions(G, positions)

    # Check if G is already a NetworkX graph
    is_networkx = isinstance(G, nx.Graph)
    if is_networkx:
        graph = G.copy()
    else:
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        for src, tgt, data in edges:
            graph.add_edge(src, tgt, **data)

    # Embed positions as node attributes
    for i, node in enumerate(node_list):
        pos = pos_array[i]
        graph.nodes[node]["x"] = float(pos[0])
        graph.nodes[node]["y"] = float(pos[1])
        if pos_array.shape[1] > 2:
            graph.nodes[node]["z"] = float(pos[2])

    import io
    buf = io.BytesIO()
    writer = nx.write_gexf if fmt == "gexf" else nx.write_graphml
    if path is not None:
        writer(graph, path)
        return None
    writer(graph, buf)
    buf.seek(0)
    return buf.read()
