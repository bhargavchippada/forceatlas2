"""ForceAtlas2 - The fastest ForceAtlas2 algorithm for Python.

A port of Gephi's ForceAtlas2 layout algorithm with support for
NetworkX, igraph, and raw adjacency matrices. Optimized with Cython
for 10-100x speedup over pure Python.
"""

from .forceatlas2 import ForceAtlas2, Timer

__all__ = ["ForceAtlas2", "Timer"]
__version__ = "1.1.1"
