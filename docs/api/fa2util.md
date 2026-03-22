# Force Functions (fa2util)

Internal force computation module. These functions exist in three backends:

1. **Cython** (`.pyx`): Compiled C extension for maximum performance
2. **Pure Python** (`.py`): Fallback when Cython is not compiled
3. **Vectorized** (`fa2util_vectorized.py`): NumPy broadcasting backend

!!! note
    Some low-level force functions (`linRepulsion`, `linGravity`, `strongGravity`, etc.)
    are internal (`cdef` in Cython) and not part of the public API. They are called
    by the batch functions below.

## Data Classes

::: fa2.fa2util.Node
    options:
      show_source: false

::: fa2.fa2util.Edge
    options:
      show_source: false

## Batch Operations

These are the main entry points used by the `ForceAtlas2` class:

::: fa2.fa2util.apply_repulsion

::: fa2.fa2util.apply_gravity

::: fa2.fa2util.apply_attraction

## Attraction Functions

::: fa2.fa2util.linAttraction

::: fa2.fa2util.logAttraction

::: fa2.fa2util.linAttraction_antiCollision

::: fa2.fa2util.logAttraction_antiCollision

## Speed Adjustment

::: fa2.fa2util.adjustSpeedAndApplyForces

## Barnes-Hut Spatial Tree

::: fa2.fa2util.Region
    options:
      show_source: false
