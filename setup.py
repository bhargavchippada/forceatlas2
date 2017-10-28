from codecs import open
from os import path

from setuptools import setup
from setuptools.extension import Extension

print("Installing fa2 package (fastest forceatlas2 python implementation)")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

if path.isfile(path.join(here, 'fa2/fa2util.c')):
    # cython build locally and add fa2/fa2util.c to MANIFEST or fa2.egg-info/SOURCES.txt
    # run: python setup.py build_ext --inplace
    ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.c'])]
    cmdclass = {}
    cythonopts = {"ext_modules": ext_modules,
                  "cmdclass": cmdclass}
else:
    cythonopts = None

    # Uncomment the following line if you want to install without optimizations
    # cythonopts = {"py_modules": ["fa2.fa2util"]}

    if cythonopts is None:
        from Cython.Build import build_ext

        ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.py', 'fa2/fa2util.pxd'])]
        cmdclass = {'build_ext': build_ext}
        cythonopts = {"ext_modules": ext_modules,
                      "cmdclass": cmdclass}

setup(
    name='fa2',
    version='0.2',
    description='The fastest ForceAtlas2 algorithm for Python (and NetworkX)',
    long_description=long_description,
    author='Bhargav Chippada',
    author_email='bhargavchippada19@gmail.com',
    url='https://github.com/bhargavchippada/forceatlas2',
    download_url='https://github.com/bhargavchippada/forceatlas2/archive/v0.2.tar.gz',
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
    install_requires=['numpy', 'scipy', 'tqdm'],
    extras_require={
        'networkx': ['networkx']
    },
    include_package_data=True,
    **cythonopts
)
