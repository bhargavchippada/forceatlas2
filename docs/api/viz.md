# Visualization (fa2.viz)

Render graph layouts to matplotlib figures or export to various formats.

Requires: `pip install fa2[viz]`

## Quick Examples

```python
from fa2.viz import plot_layout, export_layout

# Render to matplotlib
fig = plot_layout(G, positions, color_by_degree=True, title="My Graph")

# Export to D3.js-compatible JSON
export_layout(G, positions, fmt="json", path="graph.json")

# Export to PNG/SVG
export_layout(G, positions, fmt="png", path="graph.png")

# Export to Gephi format
export_layout(G, positions, fmt="gexf", path="graph.gexf")
```

## Functions

::: fa2.viz.plot_layout

::: fa2.viz.export_layout
