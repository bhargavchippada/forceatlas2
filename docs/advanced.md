# Advanced Usage

## Backends

ForceAtlas2 supports three computation backends:

| Backend | Speed | Barnes-Hut | When Used |
|---------|-------|------------|-----------|
| **Cython** | Fastest (10-100x vs pure Python) | Yes | `backend="auto"` when `.so` is compiled |
| **Vectorized** (NumPy) | 10-16x vs pure Python loops | No (O(n²)) | `backend="auto"` when no Cython, or `backend="vectorized"` |
| **Loop** (pure Python) | Baseline | Yes | `backend="loop"` |

```python
# Force a specific backend
fa2 = ForceAtlas2(backend="vectorized", verbose=False)

# Auto-select (default): Cython if available, else vectorized
fa2 = ForceAtlas2(backend="auto", verbose=False)
```

!!! tip "Choosing a backend"
    - **Cython + Barnes-Hut** is best for large graphs (>1000 nodes). Install with `pip install cython && pip install fa2 --no-binary fa2`.
    - **Vectorized** is best when Cython isn't available. It uses NumPy broadcasting but computes all-pairs repulsion (no Barnes-Hut), so it becomes slow above ~5000 nodes.
    - **Loop** (pure Python) is the slowest but supports Barnes-Hut. Rarely needed — the vectorized backend is almost always faster.

To check which backend is active:

```python
import fa2.fa2util
print(fa2.fa2util.__file__)
# Ends with .so/.pyd → Cython compiled
# Ends with .py → pure Python (auto will use vectorized)
```

## Anti-Collision (Prevent Overlap)

Use `adjustSizes` to prevent node overlap. Nodes are treated as circles with
a radius — overlapping nodes get a strong constant repulsive force (100x normal),
and attraction is zeroed when nodes touch.

```python
import networkx as nx
from fa2 import ForceAtlas2

G = nx.karate_club_graph()

# Set node sizes as attributes
for node in G.nodes():
    G.nodes[node]["size"] = G.degree(node) * 2.0

fa2 = ForceAtlas2(adjustSizes=True, verbose=False)
positions = fa2.forceatlas2_networkx_layout(
    G, iterations=100, size_attr="size"
)
```

You can also pass sizes directly to the core method:

```python
import numpy as np

sizes = np.array([5.0, 3.0, 8.0, 2.0])  # per-node radii
positions = fa2.forceatlas2(G_matrix, iterations=100, sizes=sizes)
```

!!! note
    If `adjustSizes=True` but no sizes are provided, all nodes default to
    radius `1.0`.

## Edge Weight Transformations

### Inversion

Invert weights so that stronger connections have lower values ($w = 1/w$):

```python
fa2 = ForceAtlas2(invertedEdgeWeightsMode=True, verbose=False)
```

### Normalization

Normalize weights to [0, 1] using min-max scaling:

```python
fa2 = ForceAtlas2(normalizeEdgeWeights=True, verbose=False)
```

Both can be combined — inversion is applied first, then normalization.

### Edge Weight Influence

The `edgeWeightInfluence` parameter controls how much edge weights affect attraction:

| Value | Behavior |
|-------|----------|
| `0` | All edges equal (weights ignored) |
| `1` (default) | Normal weight influence |
| `> 1` | Amplifies weight differences (heavy edges attract much more) |

## Callbacks for Animation

Use callbacks to capture positions at each iteration:

```python
import numpy as np
from fa2 import ForceAtlas2

history = []

def capture(iteration, nodes):
    # iteration is 0-indexed
    positions = [(n.x, n.y) for n in nodes]
    history.append(positions)

fa2 = ForceAtlas2(verbose=False)
fa2.forceatlas2(G, iterations=100, callbacks=[capture])

# history[i] contains positions at iteration i
```

For N-dimensional layouts, use `n.pos` instead of `n.x, n.y`:

```python
def capture_nd(iteration, nodes):
    positions = [tuple(n.pos) for n in nodes]
    history.append(positions)

fa2 = ForceAtlas2(dim=3, verbose=False)
fa2.forceatlas2(G, iterations=100, callbacks=[capture_nd])
```

!!! note
    Callback `nodes` are the live Node objects — not copies. Reading
    positions is safe; modifying forces or positions mid-iteration is not
    recommended. Multiple callbacks are called in order.

## Storing Positions as Node Attributes

Save layout results directly on the graph. Positions are both returned
and stored as attributes.

=== "NetworkX"

    ```python
    positions = fa2.forceatlas2_networkx_layout(
        G, iterations=100, store_pos_as="fa2_pos"
    )
    # Access: G.nodes[0]["fa2_pos"]  →  (x, y)
    # positions dict is also returned
    ```

=== "igraph"

    ```python
    layout = fa2.forceatlas2_igraph_layout(
        G, iterations=100, store_pos_as="fa2_pos"
    )
    # Access: G.vs[0]["fa2_pos"]  →  (x, y)
    # igraph.Layout is also returned
    ```

## Parameter Tuning

| Parameter | Effect | Increase | Decrease |
|-----------|--------|----------|----------|
| `scalingRatio` | Repulsion strength | Spread out | Compact |
| `gravity` | Center pull | Tighter | More drift |
| `barnesHutTheta` | BH accuracy | Faster, less accurate | Slower, more accurate |
| `jitterTolerance` | Speed tolerance | Faster convergence | Smoother, slower |
| `edgeWeightInfluence` | Weight effect | Stronger clustering | Uniform spacing |
| `iterations` | Layout quality | Better convergence | Faster execution |

### LinLog Mode

Enable `linLogMode=True` for community detection layouts. The logarithmic attraction formula groups communities more tightly:

```python
fa2 = ForceAtlas2(linLogMode=True, verbose=False)
```

### Dissuade Hubs

Enable `outboundAttractionDistribution=True` to push high-degree hubs toward the borders:

```python
fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
```

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Adjacency matrix is not symmetric` | Directed graph passed as matrix | Use `G.to_undirected()` (NetworkX) or `G.as_undirected()` (igraph) |
| `ValueError: Only undirected ... graphs are supported` | Directed NetworkX/igraph graph | Convert to undirected first |
| `ValueError: scalingRatio must be positive` | Invalid parameter value | Check parameter ranges (see [API docs](api/forceatlas2.md)) |
| `ValueError: pos must have shape (N, dim)` | Wrong initial positions shape | Ensure pos array matches `(num_nodes, dim)` |
| `NotImplementedError: multiThreaded` | `multiThreaded=True` | Not yet implemented — omit or set to `False` |
| `UserWarning: self-loops` | Diagonal entries in adjacency matrix | Self-loops inflate node mass; remove if unintended |
| `UserWarning: Running pure Python fa2util` | No compiled Cython extension | Rebuild: `pip install cython && pip install fa2 --no-binary fa2` |

### Common Layout Issues

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| All nodes clumped together | `scalingRatio` too low | Increase `scalingRatio` (try 10, 50, 100) |
| Nodes flying apart | `scalingRatio` too high or `gravity` too low | Decrease `scalingRatio` or increase `gravity` |
| Layout not converging | Too few iterations | Increase `iterations` (try 500-2000) |
| Very slow for large graph | Pure Python without Barnes-Hut | Install Cython; ensure `barnesHutOptimize=True` |
| Memory error | Dense matrix for large graph | Use scipy sparse matrices |
| Communities not visible | Linear attraction mode | Try `linLogMode=True` |

## Migration Guide

### From v0.3.x / v0.4.x to v0.9+

| Old API | New API |
|---------|---------|
| `forceatlas2.forceatlas2_networkx_layout(G, ...)` | Same (unchanged) |
| `forceatlas2.forceatlas2_igraph_layout(G, ...)` | Same (unchanged) |
| No `seed` parameter | `ForceAtlas2(seed=42)` |
| No callbacks | `forceatlas2(G, callbacks=[fn])` |

### From v0.9.x to v1.0.0

| v0.9.x | v1.0.0 |
|--------|--------|
| 2D only | `ForceAtlas2(dim=3)` for 3D+ |
| No overlap prevention | `ForceAtlas2(adjustSizes=True)` |
| Manual parameter tuning | `ForceAtlas2.inferSettings(G)` |
| No weight transforms | `normalizeEdgeWeights`, `invertedEdgeWeightsMode` |
| Cython or pure Python | `backend="vectorized"` option |
| Positions returned only | `store_pos_as="attr_name"` to save on graph |
