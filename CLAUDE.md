# CLAUDE.md — forceatlas2

## Project Overview

ForceAtlas2 — the fastest Python implementation of the ForceAtlas2 graph layout algorithm (from Gephi). Published on PyPI as `fa2` (version 1.0.0). Uses Cython for 10-100x performance over pure Python.

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
└── fa2util_vectorized.py    # NumPy-vectorized backend (no Barnes-Hut, O(n²))
tests/
├── conftest.py              # Shared fixtures (graph matrices)
├── test_fa2util.py          # Unit tests for force functions (48 tests)
├── test_forceatlas2.py      # Integration tests (118 tests)
├── test_vectorized.py       # Vectorized backend tests (24 tests)
└── test_benchmark.py        # Performance benchmarks (8 tests)
docs/                            # MkDocs documentation source
mkdocs.yml                      # MkDocs config (GitHub Pages deployment)
```

### Key Classes
- `ForceAtlas2` (forceatlas2.py) — main entry point with `forceatlas2()`, `forceatlas2_networkx_layout()`, `forceatlas2_igraph_layout()`, `inferSettings()`
- `Node2D` (fa2util.pyx) — 2D node with scalar C fields (x, y, dx, dy) for maximum Cython performance
- `NodeND` (fa2util.pyx) — N-dimensional node with list fields (pos, force, old_force)
- `Node(dim)` (fa2util.pyx) — factory function returning Node2D for dim=2, NodeND for dim>2
- `Node` (fa2util.py) — single Python class handling all dimensions (aliases: `Node2D = Node`, `NodeND = Node`)
- `Edge` (fa2util) — internal edge representation
- `Region` (fa2util) — Barnes-Hut spatial tree (2^dim partitioning, dispatches to 2D/ND force functions)
- `_igraph_to_sparse` (forceatlas2.py) — module-level igraph-to-sparse converter

### Dual Node Design (Cython)
For 2D (dim=2), Cython uses `Node2D` with `cdef public double x, y, dx, dy, old_dx, old_dy` — these are direct C struct fields, giving identical performance to v0.9.0. For dim>2, `NodeND` uses Python lists. The batch functions (`apply_repulsion`, etc.) check `isinstance(nodes[0], Node2D)` once, then run typed inner loops. Node2D exposes `.pos`, `.force`, `.old_force` as properties returning fresh lists for API compatibility.

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
- `setup.py` — Cython extension build
- Build chain: Cython .pyx → compiled .so → pre-generated .c → pure Python fallback

## Dependencies

- **Required**: numpy, scipy, tqdm
- **Optional**: networkx (graph wrapper), igraph (graph wrapper)
- **Dev**: pytest, pytest-cov, pytest-benchmark, ruff, cython, networkx, igraph

## Common Commands

```bash
# Install in development mode
pip install cython numpy && pip install -e ".[dev]" --no-build-isolation

# Full install (compiles Cython — needed for .so)
pip install . --no-build-isolation

# Copy compiled .so into working directory (for editable installs)
cp $(python -c "import fa2.fa2util; print(fa2.fa2util.__file__)") fa2/

# Run tests (198 total: 190 unit/integration + 8 benchmarks)
pytest tests/ -v

# Run tests with coverage (100% on all fa2 modules)
pytest tests/test_fa2util.py tests/test_forceatlas2.py tests/test_vectorized.py --cov=fa2 --cov-report=term-missing

# Run benchmarks only
pytest tests/test_benchmark.py --benchmark-only -s

# Lint
ruff check fa2/ tests/

# Regenerate C file after .pyx changes
cython fa2/fa2util.pyx -3 -o fa2/fa2util.c
```

## Algorithm Verification

The implementation has been verified against both the Gephi Java source (ForceFactory.java, ForceAtlas2.java) and the original paper (Jacomy et al. 2014). All force formulas match:

- Repulsion: `F = k_r * m1 * m2 / d` (the `distance^2` in code is the factor/distance optimization pattern)
- LinAttraction: `F = -c * w * d`
- LogAttraction: `F = -c * w * log(1 + d)` (Noack's LinLog model, paper Formula 3)
- Gravity: `F = m * g` (strong, distance-independent) or `F = m * g / d` (standard)
- Anti-collision: subtract node sizes from distance; overlap/touching → 100× constant push; attraction → zero when overlapping (Gephi ForceFactory.java parity)
- Barnes-Hut: one-sided `linRepulsion_region` for leaf nodes (matches Gephi reference)
- Speed adjustment: swing/traction adaptive system (paper Formulas 8-14)

## Features (v1.0.0)

- **N-dimensional layout**: `dim` parameter (default 2, supports 3D+)
- **adjustSizes**: anti-collision repulsion/attraction (Gephi ForceFactory.java parity)
- **inferSettings()**: auto-tuning classmethod based on graph characteristics
- **normalizeEdgeWeights**: min-max normalization to [0,1]
- **invertedEdgeWeightsMode**: w = 1/w inversion
- **NumPy vectorized backend**: `backend="vectorized"` for 10-16x speedup without Cython
- **store_pos_as**: save positions as node attributes in NX/igraph wrappers
- **size_attr**: read node sizes from graph attributes for adjustSizes
- **Dual Node2D/NodeND**: full 2D Cython performance parity with v0.9.0
- linLogMode, seed, callbacks, NetworkX/igraph support, Cython 3.x, Python 3.9-3.14
- Parameter validation, input validation, 190 tests, 100% coverage

## Not Yet Implemented

- **multiThreaded**: Parallel force computation.

See TODO comments in `forceatlas2.py` (class docstring) for details.
