# Simple API (fa2.easy)

No numpy knowledge required. One-call functions that accept edge lists and return plain dicts.

## Quick Examples

```python
from fa2.easy import layout, visualize

# Edge list → positions
positions = layout([("A", "B"), ("B", "C"), ("A", "C")], mode="community")

# Edge list → PNG image
visualize([("A", "B"), ("B", "C")], output="png", path="graph.png")
```

### Input Formats

All functions accept these edge formats:

=== "Tuples"

    ```python
    edges = [("A", "B"), ("B", "C", 5.0)]  # optional weight
    ```

=== "Dicts"

    ```python
    edges = [{"source": "A", "target": "B", "weight": 5.0}]
    ```

=== "Adjacency Dict"

    ```python
    edges = {"A": ["B", "C"], "B": ["A", "D"]}
    ```

### Mode Presets

| Mode | Parameters Set | Use Case |
|------|---------------|----------|
| `"default"` | Standard FA2 defaults | General layout |
| `"community"` | LinLog + dissuade hubs | Community detection |
| `"hub-dissuade"` | Dissuade hubs + strong gravity | Push hubs to periphery |
| `"compact"` | High gravity, low scaling | Tight layouts |

## Functions

::: fa2.easy.layout

::: fa2.easy.visualize
