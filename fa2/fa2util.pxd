# Cython optimizations.  Cython allows huge speed boosts by giving
# each variable a type.  This file is called a "pxd extension file"
# (see the "Pure Python" section of the Cython manual).  In essence,
# it provides types for function definitions and then, if cython is
# available, it uses these types to optimize normal python code.  It
# is associated with the fa2util.py file.  
#
# IF ANY CHANGES ARE MADE TO fa2util.py, THE CHANGES MUST BE REFLECTED
# HERE!!
#
# Copyright 2016 Max Shinn <mws41@cam.ac.uk>
#
# Available under the GPLv3

import cython

# This will substitute for the nLayout object
cdef class Node:
    cdef public double mass
    cdef public double old_dx, old_dy
    cdef public double dx, dy
    cdef public double x, y

# This is not in the original java function, but it makes it easier to
# deal with edges.
cdef class Edge:
    cdef public int node1, node2
    cdef public double weight



# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` (and optionally `n2`).  It does
# not return anything.

@cython.locals(xDist = cython.double, 
               yDist = cython.double, 
               distance2 = cython.double, 
               factor = cython.double)
cdef void linRepulsion(Node n1, Node n2, double coefficient=*)


@cython.locals(xDist = cython.double, 
               yDist = cython.double, 
               distance = cython.double, 
               factor = cython.double)
cdef void linGravity(Node n, double g, double coefficient=*)


@cython.locals(xDist = cython.double, 
               yDist = cython.double, 
               factor = cython.double)
cdef void strongGravity(Node n, double g, double coefficient=*)

@cython.locals(xDist = cython.double, 
               yDist = cython.double, 
               factor = cython.double)
cpdef void linAttraction(Node n1, Node n2, double e, double coefficient=*)

cpdef void apply_repulsion(list nodes, double coefficient)

cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, useStrongGravity=*)

cpdef void apply_attraction(list nodes, list edges, double coefficient, double edgeWeightInfluence)
