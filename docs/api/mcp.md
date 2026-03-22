# MCP Server (AI Agents)

ForceAtlas2 provides an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server
so AI agents can use graph layout as a tool.

Requires: `pip install fa2[mcp]`

## Setup

Add to your MCP client configuration (Claude, etc.):

```json
{
    "mcpServers": {
        "fa2": {
            "command": "python",
            "args": ["-m", "fa2.mcp_server"]
        }
    }
}
```

Or run standalone:

```bash
python -m fa2.mcp_server
```

## Tools

### `layout_graph`

Compute 2D/3D positions for a graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edges` | list | required | `[["A", "B"], ["B", "C", 5.0]]` |
| `iterations` | int | 100 | Layout iterations |
| `dim` | int | 2 | Dimensions (2 or 3) |
| `mode` | str | "default" | "default", "community", "hub-dissuade", "compact" |
| `seed` | int | None | Random seed |

**Returns**: `{"node_id": [x, y], ...}`

### `layout_and_render`

Layout a graph AND render it as a base64-encoded PNG image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edges` | list | required | Edge list |
| `iterations` | int | 100 | Layout iterations |
| `mode` | str | "default" | Layout preset |
| `seed` | int | None | Random seed |
| `title` | str | None | Chart title |

**Returns**: Base64-encoded PNG string

### `evaluate_layout`

Compute quality metrics for a graph layout.

| Parameter | Type | Description |
|-----------|------|-------------|
| `edges` | list | Edge list |
| `positions` | dict | `{"node_id": [x, y], ...}` |

**Returns**: `{"stress": float, "edge_crossings": int, "neighborhood_preservation": float}`

!!! note
    `edge_crossings` is only included for 2D layouts.
