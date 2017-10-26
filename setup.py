from distutils.core import setup

try: # If we have Cython installed
    from distutils.extension import Extension
    import Cython.Distutils
    import os
    if os.name == "nt": raise OSError
    ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.py', 'fa2/fa2util.pxd'])]
    cmdclass = {'build_ext' : Cython.Distutils.build_ext}
    cythonopts = {"ext_modules" : ext_modules,
                  "cmdclass" : cmdclass}
except ImportError:
    print("WARNING: Cython is not installed.  If you want this to be fast, install Cython and reinstall forceatlas2.")
    cythonopts = {"py_modules" : ["fa2.fa2util"]}
except OSError:
    print("WARNING: Windows and Cython don't always get along, so forceatlas2 is installing without optimizations.  Feel free to fix for your computer if you know what you're doing.")
    cythonopts = {"py_modules" : ["fa2.fa2util"]}


setup(
    name = 'fa2',
    version = '0.1',
    description = 'The ForceAtlas2 algorithm for Python (and NetworkX)',
    author = 'Bhargav Chippada',
    author_email = 'bhargavchippada19@gmail.com',
    url = 'https://github.com/bhargavchippada/forceatlas2',
    packages = ['fa2'],
    requires = ['numpy', 'scipy', 'cython'],
    **cythonopts
)
