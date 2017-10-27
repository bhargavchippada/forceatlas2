import sys
from distutils.core import setup

print("Installing fa2 package (fastest forceatlas2 python implementation)")

try:  # If we have Cython installed
    from distutils.extension import Extension
    import Cython.Distutils

    cythonopts = None

    import os

    if os.name == "nt":
        ans = input(
            "WARNING: Windows and Cython don't always get along so do you want to install without optimizations? (y/n)")
        if ans == 'y':
            cythonopts = {"py_modules": ["fa2.fa2util"]}

    if cythonopts is None:
        ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.py', 'fa2/fa2util.pxd'])]
        cmdclass = {'build_ext': Cython.Distutils.build_ext}
        cythonopts = {"ext_modules": ext_modules,
                      "cmdclass": cmdclass}
except ImportError:
    ans = input(
        "WARNING: Cython is not installed.  If you want this to be fast (10-100x) then install Cython first. Do you want to install Cython and comeback? (y/n)")
    if ans != 'n':
        print("run: pip install Cython && pip install fa2")
        sys.exit(0)
    print("WARNING: Cython is not installed.  If you want this to be fast (10-100x), install Cython and reinstall fa2.")
    cythonopts = {"py_modules": ["fa2.fa2util"]}
except Exception as e:
    print(e)
    sys.exit(0)

setup(
    name='fa2',
    version='0.1',
    description='The fastest ForceAtlas2 algorithm for Python (and NetworkX)',
    author='Bhargav Chippada',
    author_email='bhargavchippada19@gmail.com',
    url='https://github.com/bhargavchippada/forceatlas2',
    download_url='https://github.com/bhargavchippada/forceatlas2/archive/v0.1.tar.gz',
    keywords=['forceatlas2', 'networkx', 'force-directed-graph', 'force-layout', 'graph'],
    packages=['fa2'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    requires=['numpy', 'scipy', 'tqdm'],
    **cythonopts
)
