from codecs import open
from os import path

from setuptools import setup

print("Installing fa2 package (fastest forceatlas2 python implementation)\n")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

print(">>>> Cython is installed?")
try:
    from Cython.Distutils import Extension
    from Cython.Build import build_ext
    USE_CYTHON = True
    print('Yes\n')
except ImportError:
    from setuptools.extension import Extension
    USE_CYTHON = False
    print('Cython is not installed; using pre-generated C files if available')
    print('Please install Cython first and try again if you face any installation problems\n')
    print(">>>> Are pre-generated C files available?")

if USE_CYTHON:
    ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.py', 'fa2/fa2util.pxd'], cython_directives={'language_level' : 3})]
    cmdclass = {'build_ext': build_ext}
    opts = {"ext_modules": ext_modules, "cmdclass": cmdclass}
elif path.isfile(path.join(here, 'fa2/fa2util.c')):
    print("Yes\n")
    ext_modules = [Extension('fa2.fa2util', ['fa2/fa2util.c'])]
    cmdclass = {}
    opts = {"ext_modules": ext_modules, "cmdclass": cmdclass}
else:
    print("Pre-generated C files are not available. This library will be slow without Cython optimizations.\n")
    opts = {"py_modules": ["fa2.fa2util"]}

# Uncomment the following line if you want to install without optimizations
# opts = {"py_modules": ["fa2.fa2util"]}

print(">>>> Starting to install!\n")

setup(
    name='fa2',
    version='0.3.5',
    description='The fastest ForceAtlas2 algorithm for Python (and NetworkX)',
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Bhargav Chippada',
    author_email='bhargavchippada19@gmail.com',
    url='https://github.com/bhargavchippada/forceatlas2',
    download_url='https://github.com/bhargavchippada/forceatlas2/archive/v0.3.5.tar.gz',
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
        'networkx': ['networkx'],
        'igraph': ['python-igraph']
    },
    include_package_data=True,
    **opts
)