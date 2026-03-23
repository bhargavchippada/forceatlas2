# CLAUDE.md — forceatlas2

## Project Overview

ForceAtlas2 — the fastest Python implementation of the ForceAtlas2 graph layout algorithm (from Gephi). Published on PyPI as `fa2` (version 1.1.2). Uses Cython for 10-100x performance over pure Python.

- **Repository**: https://github.com/bhargavchippada/forceatlas2
- **License**: GPLv3
- **Package name (PyPI)**: `fa2`
- **Python**: >=3.9
- **Reference paper**: Jacomy et al. 2014, PLoS ONE 9(6): e98679
- **Reference implementation**: [Gephi ForceAtlas2.java](https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java)

## Architecture

```
fa2/
├── __init__.py              # Re-exports ForceAtlas2, Timer; defines __version__
├── forceatlas2.py           # Main ForceAtlas2 class (public API) — 100% test coverage
├── fa2util.py               # Force computations (pure Python fallback, N-dim)
├── fa2util.pyx              # Cython source: Node2D (scalar) + NodeND (list) dual-class design
├── fa2util.c                # Pre-generated C code (fallback when Cython not installed)
├── fa2util_vectorized.py    # NumPy-vectorized backend (no Barnes-Hut, O(n²))
├── easy.py                  # Simple API: layout() and visualize() — no numpy required
├── viz.py                   # Visualization: plot_layout(), export_layout() (PNG/SVG/JSON/GEXF)
├── metrics.py               # Layout quality: stress(), edge_crossing_count(), neighborhood_preservation()
├── mcp_server.py            # MCP server: 3 tools for AI agents
└── __main__.py              # CLI: python -m fa2 layout/render/metrics
tests/
├── conftest.py              # Shared fixtures
├── test_fa2util.py          # Unit tests for force functions (48 tests)
├── test_forceatlas2.py      # Integration tests (118 tests)
├── test_vectorized.py       # Vectorized backend tests (24 tests)
├── test_metrics.py          # Layout quality metrics tests (41 tests)
├── test_viz.py              # Visualization tests (49 tests)
├── test_easy.py             # Simple API tests (32 tests)
├── test_cli.py              # CLI tests (40 tests)
├── test_mcp.py              # MCP server tests (12 tests)
└── test_benchmark.py        # Performance benchmarks (8 tests)
docs/                        # MkDocs documentation source
mkdocs.yml                   # MkDocs config (GitHub Pages deployment)
```

### Key Classes & Modules
- `ForceAtlas2` (forceatlas2.py) — main entry point with `forceatlas2()`, `forceatlas2_networkx_layout()`, `forceatlas2_igraph_layout()`, `inferSettings()`
- `easy.layout()` / `easy.visualize()` — simplified API, no numpy required
- `viz.plot_layout()` / `viz.export_layout()` — visualization and export
- `metrics.stress()` / `metrics.edge_crossing_count()` / `metrics.neighborhood_preservation()` — quality metrics
- `Node2D` / `NodeND` (fa2util.pyx) — Cython node classes (2D scalar / N-dim list)
- `Region` (fa2util) — Barnes-Hut spatial tree (2^dim partitioning)
- `mcp_server` — MCP tools for AI agents (layout_graph, layout_and_render, evaluate_layout)

### Force Functions (fa2util)
Force functions exist in three backends:
1. **Cython** (.pyx): `*_2d()` functions with scalar fields + `*_nd()` with list fields
2. **Pure Python** (.py): single set of N-dimensional functions using lists
3. **Vectorized** (.py): NumPy broadcasting, no Barnes-Hut

| Function | Formula |
|----------|---------|
| `linRepulsion` | F = k_r * m1 * m2 / d |
| `linRepulsion_region` | Same, one-sided for Barnes-Hut |
| `linRepulsion_antiCollision` | d_adj = d - s1 - s2; overlap → 100×force; else F/d_adj² |
| `linGravity` | F = m * g / d (toward origin) |
| `strongGravity` | F = c * m * g (distance-independent) |
| `linAttraction` | F = -c * w * d (linear) |
| `logAttraction` | F = -c * w * log(1+d) (LinLog mode) |
| `linAttraction_antiCollision` | Zero force when overlapping |
| `logAttraction_antiCollision` | Zero force when overlapping |
| `adjustSpeedAndApplyForces` | Adaptive speed (swing/traction) + position update |

### Backend Selection
- `backend="auto"` (default): Cython if compiled (.so), else vectorized (NumPy)
- `backend="cython"` / `backend="loop"`: force loop-based path (Cython or pure Python)
- `backend="vectorized"`: NumPy broadcasting (O(n²), no Barnes-Hut, 10-16x faster than pure Python loops)

### Build System
- `pyproject.toml` — project metadata, dependencies, tool config (setuptools backend)
- `setup.py` — Cython extension build with graceful fallback to pure Python
- Build chain: Cython .pyx → compiled .so → pre-generated .c → pure Python fallback

## Dependencies

- **Required**: numpy, scipy, tqdm
- **Optional**: networkx, igraph, matplotlib (`fa2[viz]`), mcp (`fa2[mcp]`)
- **Dev**: pytest, pytest-cov, pytest-benchmark, ruff, cython, networkx, igraph

## Common Commands

```bash
# Install in development mode
pip install cython numpy && pip install -e ".[dev]" --no-build-isolation

# Full install (compiles Cython — needed for .so)
pip install . --no-build-isolation

# Run tests (372 total: 364 unit/integration + 8 benchmarks)
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=fa2 --cov-report=term-missing

# Run benchmarks only
pytest tests/test_benchmark.py --benchmark-only -s

# Lint
ruff check fa2/ tests/

# CLI usage
python -m fa2 layout edges.json --mode community -o layout.json
python -m fa2 render edges.csv -o graph.png
python -m fa2 metrics edges.json

# Build docs locally
pip install -e ".[docs]" && mkdocs serve

# Regenerate C file after .pyx changes
cython fa2/fa2util.pyx -3 -o fa2/fa2util.c
```

## Features (v1.1.0)

### Core Algorithm (v1.0.0)
- **N-dimensional layout**: `dim` parameter (default 2, supports 3D+)
- **adjustSizes**: anti-collision repulsion/attraction (Gephi ForceFactory.java parity)
- **inferSettings()**: auto-tuning classmethod based on graph characteristics
- **normalizeEdgeWeights**: min-max normalization to [0,1]
- **invertedEdgeWeightsMode**: w = 1/w inversion
- **NumPy vectorized backend**: `backend="vectorized"` for 10-16x speedup without Cython
- **store_pos_as**: save positions as node attributes in NX/igraph wrappers
- **size_attr**: read node sizes from graph attributes for adjustSizes
- linLogMode, seed, callbacks, NetworkX/igraph support, Cython 3.x, Python 3.9-3.14

### Visualization & Tools (v1.1.0)
- **Simple API** (`fa2.easy`): `layout()` and `visualize()` — no numpy knowledge required
- **Visualization** (`fa2.viz`): `plot_layout()` for 2D/3D matplotlib, `export_layout()` for JSON/PNG/SVG/GEXF/GraphML
- **Quality metrics** (`fa2.metrics`): `stress()`, `edge_crossing_count()`, `neighborhood_preservation()`
- **CLI** (`python -m fa2`): layout, render, metrics commands with JSON/CSV input
- **MCP server** (`fa2.mcp_server`): 3 tools for AI agents (layout_graph, layout_and_render, evaluate_layout)
- **Mode presets**: "default", "community", "hub-dissuade", "compact"
- 372 tests total, 98% coverage

## Not Yet Implemented

- **multiThreaded**: Parallel force computation.

See TODO comments in `forceatlas2.py` (class docstring) for details.
