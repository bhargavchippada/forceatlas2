"""Setup script for building Cython extensions.

This file is used by setuptools (via pyproject.toml build-system) to compile
the Cython extension module fa2util.pyx for 10-100x speedup.

Build chain: Cython .pyx → compiled .so → pre-generated .c → pure Python fallback.
If C compilation fails (e.g., no C compiler installed), the package installs
in pure Python mode (slower but functional).
"""

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _log(msg):
    sys.stderr.write(msg + "\n")


class BuildExtFallback(build_ext):
    """Build C extensions with graceful fallback to pure Python.

    If the C compiler is missing or compilation fails, the package
    installs without the extension — users get the pure Python fallback
    (fa2util.py) which is slower but fully functional.
    """

    def run(self):
        try:
            super().run()
        except Exception as e:
            _log(f"C extension build failed: {e}")
            _log("Installing pure Python version (slower). "
                 "For 10-100x speedup, install a C compiler and reinstall.")
            self.extensions = []

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            _log(f"Failed to compile {ext.name}: {e}")
            _log("Installing pure Python version (slower). "
                 "For 10-100x speedup, install a C compiler and reinstall.")


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

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtFallback})
