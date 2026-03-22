# Layout Quality Metrics (fa2.metrics)

Quantitative measures to compare parameter settings, iteration counts,
or different layouts of the same graph.

## Quick Examples

```python
from fa2.metrics import stress, edge_crossing_count, neighborhood_preservation

# How well does the layout preserve graph distances? (lower = better)
s = stress(G, positions)

# How many edge crossings? (2D only, lower = better)
crossings = edge_crossing_count(G, positions)

# Do spatial neighbors match graph neighbors? (0-1, higher = better)
np_score = neighborhood_preservation(G, positions, k=10)
```

All functions accept any supported graph type (NetworkX, igraph, numpy, scipy sparse)
and any position format (dict, list, ndarray).

## Functions

::: fa2.metrics.stress

::: fa2.metrics.edge_crossing_count

::: fa2.metrics.neighborhood_preservation
