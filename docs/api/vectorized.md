# Vectorized Backend

NumPy-vectorized force computation for 10-16x speedup without Cython. Uses broadcasting for pairwise operations on `(N, dim)` arrays.

This module is used automatically when `backend="vectorized"` or when `backend="auto"` and no Cython extension is compiled.

!!! note
    The vectorized backend does not support Barnes-Hut approximation — all repulsion is O(n²) pairwise. For graphs over ~5000 nodes, the Cython backend with Barnes-Hut is significantly faster.

## Functions

::: fa2.fa2util_vectorized.apply_repulsion

::: fa2.fa2util_vectorized.apply_repulsion_adjustSizes

::: fa2.fa2util_vectorized.apply_gravity

::: fa2.fa2util_vectorized.apply_attraction

::: fa2.fa2util_vectorized.adjustSpeedAndApplyForces
