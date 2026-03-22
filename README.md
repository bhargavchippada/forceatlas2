# ForceAtlas2 for Python

[![CI](https://github.com/bhargavchippada/forceatlas2/actions/workflows/ci.yml/badge.svg)](https://github.com/bhargavchippada/forceatlas2/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/fa2.svg)](https://pypi.org/project/fa2/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://bhargavchippada.github.io/forceatlas2/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The fastest Python implementation of the [ForceAtlas2](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679) graph layout algorithm, with Cython optimization for 10-100x speedup. Supports NetworkX, igraph, and raw adjacency matrices.

ForceAtlas2 is a force-directed layout algorithm designed for network visualization. It spatializes **weighted undirected** graphs in 2D, 3D, or higher dimensions, where edge weights define connection strength. It scales well to large graphs (>10,000 nodes) using Barnes-Hut approximation (O(n log n) complexity).

**[Documentation](https://bhargavchippada.github.io/forceatlas2/)** · **[PyPI](https://pypi.org/project/fa2/)** · **[Paper](http://doi.org/10.1371/journal.pone.0098679)**

<p align="center">
  <img src="https://raw.githubusercontent.com/bhargavchippada/forceatlas2/master/examples/forceatlas2_animation.gif" alt="ForceAtlas2 layout animation — 500 nodes with 7 communities separating over 600 iterations">
</p>
<p align="center"><em>500-node stochastic block model (7 communities) laid out with ForceAtlas2 LinLog mode</em></p>

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/bhargavchippada/forceatlas2/master/examples/geometric_graph.png" alt="Random geometric graph laid out with ForceAtlas2">
</p>
<p align="center"><em>Random geometric graph (400 nodes) laid out with ForceAtlas2</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/bhargavchippada/forceatlas2/master/examples/forceatlas2_3d_animation.gif" alt="ForceAtlas2 3D layout animation — 1000 nodes with 8 communities separating over 600 iterations">
</p>
<p align="center"><em>1000-node stochastic block model (8 communities) laid out in 3D with ForceAtlas2 LinLog mode</em></p>

## Installation

```bash
pip install fa2
```

For maximum performance, install with Cython (recommended):

```bash
pip install cython
pip install fa2 --no-binary fa2
```

To build from source:

```bash
git clone https://github.com/bhargavchippada/forceatlas2.git
cd forceatlas2
pip install cython numpy
pip install -e ".[dev]" --no-build-isolation
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Adjacency matrix handling |
| scipy | Yes | Sparse matrix support |
| tqdm | Yes | Progress bar |
| cython | No (recommended) | 10-100x speedup |
| networkx | No | NetworkX graph wrapper |
| igraph | No | igraph graph wrapper |
| matplotlib | No | Visualization (`pip install fa2[viz]`) |

**Python**: >= 3.9 (tested on 3.9 through 3.14)

## Quick Start

### Simplest — No Numpy Required

```python
from fa2.easy import layout, visualize

# Edge list in → positions out
positions = layout([("A", "B"), ("B", "C"), ("A", "C")], mode="community")

# One call to render
visualize([("A", "B"), ("B", "C"), ("A", "C")], output="png", path="graph.png")
```

### CLI

```bash
# Layout from JSON edge list
python -m fa2 layout edges.json --mode community -o layout.json

# Render to image
python -m fa2 render edges.csv -o graph.png

# Compute quality metrics
echo '[["A","B"],["B","C"]]' | python -m fa2 metrics
```

### With NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2

G = nx.random_geometric_graph(400, 0.2)

forceatlas2 = ForceAtlas2(
    outboundAttractionDistribution=True,  # Dissuade hubs
    edgeWeightInfluence=1.0,
    jitterTolerance=1.0,
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,
    verbose=True,
)

positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

nx.draw_networkx_nodes(G, positions, node_size=20, node_color="blue", alpha=0.4)
nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.05)
plt.axis("off")
plt.show()
```

### With Raw Adjacency Matrix

```python
import numpy as np
from fa2 import ForceAtlas2

# Create a symmetric adjacency matrix
G = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
], dtype=float)

forceatlas2 = ForceAtlas2(verbose=False, seed=42)
positions = forceatlas2.forceatlas2(G, iterations=1000)
# Returns: [(x1, y1), (x2, y2), ...]
```

### With Scipy Sparse Matrix

```python
import scipy.sparse
from fa2 import ForceAtlas2

# For large graphs, sparse matrices are more memory-efficient
G_sparse = scipy.sparse.csr_matrix(adjacency_matrix)
forceatlas2 = ForceAtlas2(verbose=False)
positions = forceatlas2.forceatlas2(G_sparse, iterations=1000)
```

### With igraph

```python
import igraph
from fa2 import ForceAtlas2

G = igraph.Graph.Famous("Petersen")
forceatlas2 = ForceAtlas2(verbose=False)
layout = forceatlas2.forceatlas2_igraph_layout(G, iterations=1000)
igraph.plot(G, layout=layout)
```

## API Reference

### `ForceAtlas2(**kwargs)`

Create a ForceAtlas2 layout engine with the following parameters:

#### Behavior

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outboundAttractionDistribution` | bool | `False` | Dissuade hubs — distributes attraction along outbound edges so hubs are pushed to borders |
| `linLogMode` | bool | `False` | Use Noack's LinLog model: `F = log(1 + distance)` instead of `F = distance`. Produces tighter community clusters |
| `adjustSizes` | bool | `False` | Prevent node overlap using anti-collision forces (Gephi parity). Pass `sizes` or `size_attr` to set node radii |
| `edgeWeightInfluence` | float | `1.0` | How much edge weights matter. `0` = all edges equal, `1` = normal, other values apply `weight^influence` |
| `normalizeEdgeWeights` | bool | `False` | Min-max normalize edge weights to [0, 1]. Applied after inversion |
| `invertedEdgeWeightsMode` | bool | `False` | Invert edge weights (`w = 1/w`). Applied before normalization |

#### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `barnesHutOptimize` | bool | `True` | Use Barnes-Hut tree approximation for repulsion. Reduces O(n^2) to O(n log n) |
| `barnesHutTheta` | float | `1.2` | Barnes-Hut accuracy/speed tradeoff. Lower = more accurate but slower |
| `jitterTolerance` | float | `1.0` | How much oscillation is tolerated during convergence. Higher = faster but less precise |
| `backend` | str | `"auto"` | `"auto"`: Cython if compiled, else vectorized. `"cython"` / `"loop"`: force loop-based (Cython or pure Python). `"vectorized"`: NumPy (no BH, O(n²)) |

#### Tuning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scalingRatio` | float | `2.0` | Repulsion strength. Higher = more spread out graph. Must be > 0 |
| `strongGravityMode` | bool | `False` | Distance-independent gravity: constant pull regardless of distance from center |
| `gravity` | float | `1.0` | Center attraction strength. Prevents disconnected components from drifting. Must be >= 0 |

#### Layout & Other

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | `2` | Number of layout dimensions. Use `3` for 3D layouts, etc. |
| `seed` | int/None | `None` | Random seed for reproducible layouts |
| `verbose` | bool | `True` | Show progress bar (tqdm) and timing breakdown |

### Class Methods

#### `ForceAtlas2.inferSettings(G, **overrides)`

Auto-tune parameters based on graph characteristics. Returns a configured `ForceAtlas2` instance.

- **G**: Any supported graph type (ndarray, sparse, networkx.Graph, igraph.Graph)
- **\*\*overrides**: Override any inferred parameter
- **Returns**: `ForceAtlas2` instance

```python
fa = ForceAtlas2.inferSettings(G, verbose=False, seed=42)
pos = fa.forceatlas2(G, iterations=100)
```

### Methods

#### `forceatlas2(G, pos=None, iterations=100, callbacks=None, sizes=None)`

Compute layout from an adjacency matrix.

- **G**: `numpy.ndarray` or `scipy.sparse` matrix (must be symmetric)
- **pos**: Initial positions as `(N, dim)` array, or `None` for random
- **iterations**: Number of layout iterations (must be >= 1)
- **callbacks**: List of `callback(iteration, nodes)` functions
- **sizes**: Node radii as `(N,)` array (for `adjustSizes=True`)
- **Returns**: List of tuples with `dim` elements per node

#### `forceatlas2_networkx_layout(G, pos=None, iterations=100, weight_attr=None, callbacks=None, size_attr=None, store_pos_as=None)`

Compute layout for a NetworkX graph. Supports NetworkX 2.7+ and 3.x.

- **G**: `networkx.Graph` (undirected)
- **pos**: Initial positions as `{node: tuple}` dict
- **weight_attr**: Edge attribute name for weights
- **callbacks**: List of `callback(iteration, nodes)` functions
- **size_attr**: Node attribute name for sizes (used with `adjustSizes`)
- **store_pos_as**: If set, saves positions as node attributes under this key
- **Returns**: Dict of `{node: tuple}`

#### `forceatlas2_igraph_layout(G, pos=None, iterations=100, weight_attr=None, callbacks=None, size_attr=None, store_pos_as=None)`

Compute layout for an igraph graph.

- **G**: `igraph.Graph` (must be undirected)
- **pos**: Initial positions as list or `(N, dim)` numpy array
- **weight_attr**: Edge attribute name for weights
- **callbacks**: List of `callback(iteration, nodes)` functions
- **size_attr**: Vertex attribute name for sizes
- **store_pos_as**: If set, saves positions as vertex attributes
- **Returns**: `igraph.Layout`

## Advanced Usage

### Reproducible Layouts

Use the `seed` parameter for deterministic results:

```python
fa = ForceAtlas2(seed=42, verbose=False)
pos1 = fa.forceatlas2_networkx_layout(G, iterations=1000)

fa2 = ForceAtlas2(seed=42, verbose=False)
pos2 = fa2.forceatlas2_networkx_layout(G, iterations=1000)
# pos1 == pos2 guaranteed
```

### LinLog Mode (Community Detection)

LinLog mode replaces the linear attraction force `F = distance` with a logarithmic one `F = log(1 + distance)` ([Noack's LinLog energy model](http://doi.org/10.1371/journal.pone.0098679)). This produces layouts where communities form tighter, more clearly separated clusters:

```python
fa = ForceAtlas2(linLogMode=True, verbose=False)
positions = fa.forceatlas2_networkx_layout(G, iterations=2000)
```

The attraction grows only logarithmically with distance, so distant connected nodes are pulled less strongly relative to repulsion, naturally emphasizing community structure.

### Dissuade Hubs

Push high-degree nodes to the periphery by distributing attraction force across outbound edges:

```python
fa = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
positions = fa.forceatlas2_networkx_layout(G, iterations=2000)
```

Each edge's attraction is divided by the source node's mass (degree + 1), so hub nodes with many connections experience less total attraction pull. An `outboundAttCompensation` factor (mean node mass) is applied to maintain overall force balance.

### Iteration Callbacks (Animation / History)

Track positions over time for animation or convergence analysis:

```python
history = []

def record_positions(iteration, nodes):
    if iteration % 100 == 0:
        history.append([(n.x, n.y) for n in nodes])

fa = ForceAtlas2(verbose=False, seed=42)
final_pos = fa.forceatlas2(G, iterations=1000, callbacks=[record_positions])
# history contains snapshots every 100 iterations
```

### Custom Edge Weights

```python
import networkx as nx

G = nx.Graph()
G.add_edge("A", "B", strength=5.0)
G.add_edge("B", "C", strength=1.0)
G.add_edge("A", "C", strength=0.5)

fa = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
pos = fa.forceatlas2_networkx_layout(G, weight_attr="strength", iterations=1000)
```

### 3D Layout

```python
fa = ForceAtlas2(dim=3, verbose=False, seed=42)
pos_3d = fa.forceatlas2_networkx_layout(G, iterations=1000)
# pos_3d = {node: (x, y, z), ...}
```

### Prevent Node Overlap (adjustSizes)

```python
import networkx as nx

G = nx.karate_club_graph()
for n in G.nodes():
    G.nodes[n]["size"] = G.degree(n) * 0.5  # Size proportional to degree

fa = ForceAtlas2(adjustSizes=True, verbose=False, seed=42)
pos = fa.forceatlas2_networkx_layout(G, iterations=1000, size_attr="size")
```

### Auto-Tuning (inferSettings)

```python
fa = ForceAtlas2.inferSettings(G, verbose=False, seed=42)
pos = fa.forceatlas2_networkx_layout(G, iterations=1000)
```

### Edge Weight Processing

```python
# Invert weights (strong connections → weak attraction)
fa = ForceAtlas2(invertedEdgeWeightsMode=True, verbose=False)

# Normalize weights to [0, 1]
fa = ForceAtlas2(normalizeEdgeWeights=True, verbose=False)

# Both combined
fa = ForceAtlas2(invertedEdgeWeightsMode=True, normalizeEdgeWeights=True, verbose=False)
```

### Store Positions as Node Attributes

```python
fa = ForceAtlas2(verbose=False, seed=42)
pos = fa.forceatlas2_networkx_layout(G, iterations=1000, store_pos_as="fa2_pos")
# Now G.nodes[n]["fa2_pos"] == pos[n] for all nodes
```

### Tuning Tips

| Goal | Settings |
|------|----------|
| **Spread out** | Increase `scalingRatio` (e.g., 10.0) |
| **Compact** | Decrease `scalingRatio` (e.g., 0.5), increase `gravity` |
| **Community clusters** | Enable `linLogMode=True` |
| **Prevent hub dominance** | Enable `outboundAttractionDistribution=True` |
| **Faster convergence** | Increase `jitterTolerance` (e.g., 5.0) |
| **Higher quality** | More `iterations`, lower `jitterTolerance` |
| **Large graphs (>5000)** | Keep `barnesHutOptimize=True` (default) |
| **Strong gravity** | Set `strongGravityMode=True` for constant-magnitude pull |
| **Prevent overlap** | `adjustSizes=True` with node sizes via `size_attr` |
| **3D layout** | `dim=3` (or any integer >= 2) |
| **Auto-tune** | `ForceAtlas2.inferSettings(G)` |
| **No Cython available** | `backend="vectorized"` (auto-detected by default) |

## Performance

The Cython-compiled version provides 10-100x speedup over pure Python:

#### Backend Comparison (small graphs)

| Graph Size | Edges | Iterations | Pure Python | Vectorized | Cython | Speedup |
|-----------|-------|-----------|------------|-----------|--------|---------|
| 50 nodes | ~225 | 100 | ~178ms | ~11ms | ~3ms | ~60x |
| 200 nodes | ~377 | 50 | ~982ms | ~61ms | ~12ms | ~82x |
| 500 nodes | ~415 | 20 | ~1,045ms | ~157ms | ~16ms | ~65x |

#### Large Graph Scaling (Cython, 2D)

| Nodes | Edges | Iterations | Time |
|-------|-------|-----------|------|
| 1,000 | ~10,000 | 50 | 0.08s |
| 5,000 | ~52,000 | 10 | 0.19s |
| 10,000 | ~105,000 | 5 | 0.28s |
| 50,000 | ~525,000 | 1 | 0.87s |
| 100,000 | ~1,050,000 | 1 | 1.84s |
| 500,000 | ~5,250,000 | 1 | 10.9s |

#### Dimensional Scaling (Cython, 10k nodes, 5 iterations)

| Dim | Time | Overhead vs 2D |
|-----|------|----------------|
| 2D | 0.28s | — |
| 3D | 1.06s | ~3.8x |
| 5D | 4.88s | ~17x |

*Higher dimensions use list-based NodeND (slower than scalar Node2D). The 2D path uses direct C struct fields for maximum performance.*

Three backends are available via `backend=`:
- **`"auto"`** (default): Uses Cython if compiled, otherwise NumPy vectorized
- **`"vectorized"`**: NumPy-vectorized (no Barnes-Hut, O(n²) — best for small-medium graphs without Cython)
- **`"loop"`**: Pure Python loops (slowest, always available)

*Benchmarks on Ubuntu Linux, Python 3.13, Cython 3.2. Barnes-Hut enabled for Cython/loop backends. Sparse random graphs with ~20 edges/node.*

To verify Cython is active:

```python
import fa2.fa2util
print(fa2.fa2util.__file__)  # Should end in .so (Linux/Mac) or .pyd (Windows), not .py
```

## Algorithm

Based on the paper:

> Jacomy M, Venturini T, Heymann S, Bastian M (2014) *ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software.* PLoS ONE 9(6): e98679. https://doi.org/10.1371/journal.pone.0098679

The implementation follows the [Gephi Java source](https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java) and has been verified against both the paper and the reference code.

### Force Model

ForceAtlas2 uses a "(1, 1)" energy model — inverse-distance repulsion and linear attraction:

| Force | Formula | Description |
|-------|---------|-------------|
| **Repulsion** | `F = k_r * m1 * m2 / d` | All node pairs repel. Mass = degree + 1. Barnes-Hut quadtree approximation reduces O(n^2) to O(n log n) |
| **Linear Attraction** | `F = -c * w * d` | Connected nodes attract proportionally to distance and edge weight |
| **Log Attraction** | `F = -c * w * log(1 + d)` | LinLog mode: sub-linear attraction for community emphasis |
| **Gravity** | `F = m * g / d` | Pull toward center, weakens with distance (standard mode) |
| **Strong Gravity** | `F = c * m * g` | Distance-independent pull toward center (constant magnitude) |

### Adaptive Speed

Each iteration measures **swinging** (erratic oscillation) and **traction** (useful movement) across all nodes. Global speed is set proportional to `traction / swinging`, with per-node damping for oscillating nodes. This allows fast convergence while preventing instability.

### Barnes-Hut Approximation

A 2^dim spatial tree recursively partitions the space. For distant node groups, repulsion is computed against the group's center of mass instead of individual nodes. The `barnesHutTheta` parameter (default 1.2) controls the distance/size threshold — higher values are faster but less accurate.

## Visualization & Export

Requires: `pip install fa2[viz]`

```python
from fa2.viz import plot_layout, export_layout

# Render to matplotlib figure
fig = plot_layout(G, positions, color_by_degree=True, title="My Graph")

# Export to various formats
export_layout(G, positions, fmt="json", path="graph.json")   # D3.js/Sigma.js compatible
export_layout(G, positions, fmt="png", path="graph.png")     # PNG image
export_layout(G, positions, fmt="gexf", path="graph.gexf")   # Gephi format
```

## Layout Quality Metrics

```python
from fa2.metrics import stress, edge_crossing_count, neighborhood_preservation

s = stress(G, positions)                           # Lower is better
crossings = edge_crossing_count(G, positions)      # 2D only
np_score = neighborhood_preservation(G, positions)  # 0-1, higher is better
```

## MCP Server (AI Agents)

ForceAtlas2 can be used as an MCP tool by AI agents:

```json
{
    "mcpServers": {
        "fa2": {"command": "python", "args": ["-m", "fa2.mcp_server"]}
    }
}
```

Requires: `pip install fa2[mcp]`

Tools: `layout_graph`, `layout_and_render`, `evaluate_layout`

## Migration Guide

All versions are backwards compatible — existing code continues to work unchanged.

### From v1.0.x to v1.1.0

v1.1.0 adds new modules without changing any existing API. No code changes required.

| What's new | Module | Install |
|------------|--------|---------|
| Simple API — `layout()`, `visualize()` from edge lists | `fa2.easy` | included |
| CLI — `python -m fa2 layout/render/metrics` | `fa2.__main__` | included |
| Visualization — `plot_layout()`, `export_layout()` | `fa2.viz` | `pip install fa2[viz]` |
| Quality metrics — `stress()`, `edge_crossing_count()`, `neighborhood_preservation()` | `fa2.metrics` | included |
| MCP server — AI agent tools | `fa2.mcp_server` | `pip install fa2[mcp]` |
| Mode presets — `"community"`, `"hub-dissuade"`, `"compact"` | `fa2.easy` | included |

**Before (v1.0.x):**

```python
import numpy as np
from fa2 import ForceAtlas2

G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
fa = ForceAtlas2(linLogMode=True, verbose=False, seed=42)
pos = fa.forceatlas2(G, iterations=100)
```

**After (v1.1.0) — same code works, plus simpler alternative:**

```python
from fa2.easy import layout

pos = layout([("A", "B"), ("B", "C"), ("A", "C")], mode="community")
```

### From v0.3.x to v1.x

| Change | v0.3.x | v1.0.0+ |
|--------|--------|---------|
| Python support | 2.7, 3.x | 3.9+ only |
| NetworkX | 2.x only | 2.7+ and 3.x |
| Cython | 0.29.x | 3.x |
| `linLogMode` | Not implemented | Implemented (correct `log(1+d)` formula) |
| `seed` parameter | Not available | New — for reproducibility |
| `callbacks` | Not available | New — for animation/monitoring |
| `dim` parameter | N/A | New — 3D+ layouts |
| `adjustSizes` | Silent no-op | Implemented (Gephi anti-collision parity) |
| `inferSettings()` | N/A | New — auto-tuning from graph characteristics |
| `normalizeEdgeWeights` | N/A | New — min-max normalize to [0,1] |
| `invertedEdgeWeightsMode` | N/A | New — w = 1/w inversion |
| `backend` parameter | N/A | New — `"auto"`, `"cython"`, `"vectorized"`, `"loop"` |
| igraph support | Fragile | Robust (handles weighted, edgeless, directed-rejection) |
| Error handling | `assert` statements | Proper `ValueError`/`TypeError` with messages |
| Input validation | Minimal | Symmetry, pos/sizes shape, param ranges, self-loop warning |
| Barnes-Hut | Double-counting leaf repulsion | Correct one-sided repulsion (matches Gephi) |
| `multiThreaded` | Silent no-op | Raises `NotImplementedError` |

### Breaking changes (v0.3.x → v1.x)

- **Python 2 dropped**: Python 2.x is no longer supported.
- **`multiThreaded=True`** now raises `NotImplementedError` instead of being silently ignored.
- **Invalid parameter values** (negative `scalingRatio`, etc.) now raise `ValueError`.

## Development

```bash
# Clone and install
git clone https://github.com/bhargavchippada/forceatlas2.git
cd forceatlas2
pip install cython numpy
pip install -e ".[dev]" --no-build-isolation

# Run tests (372 total)
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=fa2 --cov-report=term-missing

# Run benchmarks only
pytest tests/test_benchmark.py --benchmark-only -s

# Lint
ruff check fa2/ tests/

# Regenerate C file after modifying fa2util.pyx
cython fa2/fa2util.pyx -3 -o fa2/fa2util.c
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass (100% coverage on `forceatlas2.py`)
5. Submit a pull request

### Areas needing help

- **`multiThreaded`**: Parallel force computation. Gephi parallelizes repulsion + gravity (not attraction) with thread pooling. Python's GIL limits benefit, but Cython `nogil` or multiprocessing could help.

## License

```
Copyright (C) 2017 Bhargav Chippada bhargavchippada19@gmail.com
Licensed under the GNU GPLv3.
```

Based on the Gephi ForceAtlas2 plugin:

```
Copyright 2008-2011 Gephi
Authors: Mathieu Jacomy <mathieu.jacomy@gmail.com>
Licensed under GPL v3 / CDDL
```

And Max Shinn's Python port:

```
Copyright 2016 Max Shinn <mws41@cam.ac.uk>
Available under the GPLv3
```

Also thanks to Eugene Bosiakov (https://github.com/bosiakov/fa2l).

<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/bhargavchippada/forceatlas2/master/examples/grid_graph.png" alt="2D Grid graph">
</p>
<p align="center"><em>2D Grid graph (25x25) laid out with ForceAtlas2</em></p>
