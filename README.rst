ForceAtlas2 for Python and NetworkX
===================================

A port of Gephi’s Force Atlas 2 layout algorithm to Python 2 and Python
3 (with a wrapper for NetworkX). This is the fastest python
implementation available with most of the features complete. It also
supports Barnes Hut approximation for maximum speedup.

ForceAtlas2 is a very fast layout algorithm for force directed graphs.
The implementation is based on this `paper`_ and the corresponding
`gephi-java-code`_. Its really quick compared to the
fruchterman reingold algorithm (spring layout) of networkx and scales
well to high number of nodes (>10000).

Installation
------------

Install from pip:

::

    pip install fa2

To build and install run from source:

::

    python setup.py install

**Cython is highly recommended if you are buidling from source as it will speed up by a factor of 10-100x depending on the graph**

Dependencies
~~~~~~~~~~~~

-  numpy (adjacency matrix as complete matrix)
-  scipy (adjacency matrix as sparse matrix)
-  tqdm (progressbar)
-  Cython (10-100x speedup)
-  networkx (To use the NetworkX wrapper function, you obviously need
   NetworkX)

Usage
-----

from fa2 import ForceAtlas2

Create a ForceAtlas2 object with the appropriate settings. ForceAtlas2
class contains two important methods:

.. code:: python

    forceatlas2 (G, pos, iteraions)
    # G is a graph in 2D numpy ndarray format (or) scipy sparse matrix format
    # pos is a numpy array (Nx2) of initial positions of nodes
    # iterations is num of iterations to run the algorithm

.. code:: python

    forceatlas2_networkx_layout(G, pos, iterations)
    # G is networkx graph
    # pos is a dictionary, as in networkx
    # iterations is num of iterations to run the algorithm

Below is an example usage. You can also see the feature settings of
ForceAtlas2 class.

.. code-block:: python
  
    import networkx as nx
    from fa2 import ForceAtlas2
    import matplotlib.pyplot as plt

    G = nx.karate_club_graph()

    forceatlas2 = ForceAtlas2(
                              # Behavior alternatives
                              outboundAttractionDistribution=False,  # Dissuade hubs
                              linLogMode=False,  # NOT IMPLEMENTED
                              adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                              edgeWeightInfluence=1.0,

                              # Performance
                              jitterTolerance=1.0,  # Tolerance
                              barnesHutOptimize=True,
                              barnesHutTheta=1.2,
                              multiThreaded=False,  # NOT IMPLEMENTED

                              # Tuning
                              scalingRatio=2.0,
                              strongGravityMode=False,
                              gravity=1.0,

                              # Log
                              verbose=True)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    nx.draw_networkx(G, positions, cmap=plt.get_cmap('jet'), node_size=50, with_labels=False)
    plt.show()
    
You can also take a look at forceatlas2.py file for understanding the
ForceAtlas2 class and its functions better.

Features Completed
------------------

-  **barnesHutOptimize**: Barnes Hut optimization, n² complexity to
   n.ln(n)
-  **gravity**: Attracts nodes to the center. Prevents islands from
   drifting away
-  **Dissuade Hubs**: Distributes attraction along outbound edges. Hubs
   attract less and thus are pushed to the borders
-  **scalingRatio**: How much repulsion you want. More makes a more
   sparse graph
-  **strongGravityMode**: A stronger gravity view
-  **jitterTolerance**: How much swinging you allow. Above 1
   discouraged. Lower gives less speed and more precision
-  **verbose**: Shows a progressbar of iterations completed. Also, shows
   time taken for different force computations
-  **edgeWeightInfluence**: How much influence you give to the edges
   weight. 0 is “no influence” and 1 is “normal”

Documentation
-------------

You will find all the documentation in the source code

Contributors
------------

Contributions are highly welcome. Please submit your pull requests and
become a collaborator.

Copyright
---------

::

    Copyright (C) 2017 Bhargav Chippada bhargavchippada19@gmail.com.
    Licensed under the GNU GPLv3.

The files are heavily based on the java files included in Gephi, git
revision 2b9a7c8 and Max Shinn’s port to python of the algorithm. Here I
include the copyright information from those files:

::

    Copyright 2008-2011 Gephi
    Authors : Mathieu Jacomy <mathieu.jacomy@gmail.com>
    Website : http://www.gephi.org
    Copyright 2011 Gephi Consortium. All rights reserved.
    Portions Copyrighted 2011 Gephi Consortium.
    The contents of this file are subject to the terms of either the
    GNU General Public License Version 3 only ("GPL") or the Common
    Development and Distribution License("CDDL") (collectively, the
    "License"). You may not use this file except in compliance with
    the License.

    <https://github.com/mwshinn/forceatlas2-python>
    Copyright 2016 Max Shinn <mws41@cam.ac.uk>
    Available under the GPLv3

    Also, thanks to Eugene Bosiakov <https://github.com/bosiakov/fa2l>

.. _paper: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679
.. _gephi-java-code: https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java
