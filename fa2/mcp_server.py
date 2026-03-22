"""MCP (Model Context Protocol) server for ForceAtlas2.

Exposes graph layout as tools for AI agents. Run with::

    python -m fa2.mcp_server

Configure in Claude/MCP settings::

    {
        "mcpServers": {
            "fa2": {
                "command": "python",
                "args": ["-m", "fa2.mcp_server"]
            }
        }
    }
"""

import base64
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "mcp package is required for the fa2 MCP server. "
        "Install it with: pip install 'fa2[mcp]'"
    ) from None

mcp = FastMCP("fa2")


@mcp.tool()
def layout_graph(
    edges: list,
    iterations: int = 100,
    dim: int = 2,
    mode: str = "default",
    seed: Optional[int] = None,
) -> dict:
    """Compute 2D/3D positions for a graph using ForceAtlas2 force-directed layout.

    Good for network visualization, community detection visualization,
    and knowledge graph rendering.

    Args:
        edges: List of [source, target] or [source, target, weight] arrays.
               Example: [["A", "B"], ["B", "C", 5.0]]
        iterations: Number of layout iterations (default 100, more = higher quality).
        dim: Dimensions: 2 for 2D, 3 for 3D (default 2).
        mode: Layout style — "default", "community" (tight clusters),
              "hub-dissuade" (pushes hub nodes to the periphery), "compact".
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping node IDs to coordinate lists.
    """
    from fa2.easy import layout

    edge_tuples = [tuple(e) for e in edges]

    positions = layout(
        edge_tuples,
        iterations=iterations,
        dim=dim,
        mode=mode,
        seed=seed,
    )

    return {str(k): list(v) for k, v in positions.items()}


@mcp.tool()
def layout_and_render(
    edges: list,
    iterations: int = 100,
    mode: str = "default",
    seed: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    """Layout a graph AND render it as a PNG image.

    Returns a base64-encoded PNG image that can be displayed directly.

    Args:
        edges: List of [source, target] or [source, target, weight] arrays.
        iterations: Number of layout iterations.
        mode: Layout style — "default", "community", "hub-dissuade", "compact".
        seed: Random seed for reproducibility.
        title: Optional chart title.

    Returns:
        Base64-encoded PNG image string.
    """
    from fa2.easy import visualize

    edge_tuples = [tuple(e) for e in edges]

    png_bytes = visualize(
        edge_tuples,
        iterations=iterations,
        mode=mode,
        output="png",
        seed=seed,
        title=title,
    )

    if png_bytes is None:
        return ""

    return base64.b64encode(png_bytes).decode("ascii")


@mcp.tool()
def evaluate_layout(
    edges: list,
    positions: dict,
) -> dict:
    """Compute quality metrics for a graph layout.

    Measures how well the layout preserves graph structure.

    Args:
        edges: List of [source, target] or [source, target, weight] arrays.
        positions: Dict mapping node IDs to [x, y] coordinate arrays.

    Returns:
        Dict with stress, neighborhood_preservation (always present), and
        edge_crossings (2D layouts only, omitted for 3D).
        Lower stress is better. Higher neighborhood_preservation (0-1) is better.
    """
    from fa2.easy import _parse_edges
    from fa2.metrics import edge_crossing_count, neighborhood_preservation, stress

    edge_tuples = [tuple(e) for e in edges]
    node_list, G = _parse_edges(edge_tuples)

    # Convert positions to proper format (str keys from layout_graph → original types)
    pos_dict = {}
    for k, v in positions.items():
        if isinstance(k, str):
            try:
                key = int(k)
            except ValueError:
                key = k
        else:
            key = k
        pos_dict[key] = tuple(v)

    if not pos_dict:
        return {"stress": 0.0, "neighborhood_preservation": 1.0}

    n = len(node_list)
    result = {}
    result["stress"] = round(stress(G, pos_dict), 4)

    # Infer dimensionality from positions, not from args
    sample_pos = next(iter(pos_dict.values()))
    if len(sample_pos) == 2:
        result["edge_crossings"] = edge_crossing_count(G, pos_dict)

    k = min(10, n - 1) if n > 1 else 1
    result["neighborhood_preservation"] = round(neighborhood_preservation(G, pos_dict, k=k), 4)

    return result


def main():
    mcp.run()


if __name__ == "__main__":
    main()
