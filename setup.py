"""Setup script for building Cython extensions.

This file is used by setuptools (via pyproject.toml build-system) to compile
the Cython extension module fa2util.pyx for 10-100x speedup.
"""

import os
import sys

from setuptools import Extension, setup


def _log(msg):
    sys.stderr.write(msg + "\n")


# Try Cython first, fall back to pre-generated C, then pure Python
try:
    import numpy
    from Cython.Build import cythonize

    ext_modules = cythonize(
        "fa2/fa2util.pyx",
        compiler_directives={"language_level": "3"},
    )
    for ext in ext_modules:
        ext.include_dirs = [numpy.get_include()]
    _log("Building with Cython extensions (10-100x speedup).")

except ImportError:
    if os.path.isfile("fa2/fa2util.c"):
        import numpy

        ext_modules = [Extension("fa2.fa2util", ["fa2/fa2util.c"], include_dirs=[numpy.get_include()])]
        _log("Cython not available. Using pre-generated C file.")
    else:
        ext_modules = []
        _log("No Cython or C file available. Installing pure Python version (slower).")

setup(ext_modules=ext_modules)
