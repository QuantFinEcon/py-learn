""" 
http://cython.readthedocs.io/en/latest/src/tutorial/pure.html 
works in sync with intergrate_py.py to override declaration
cpdef: provide the Python wrappers to the definitions in the .pxd

from math import sin
import cython

@cython.locals(x=cython.double)
def f(x):
    return sin(x**2)

@cython.locals(a=cython.double, b=cython.double,
               N=cython.Py_ssize_t, dx=cython.double,
               s=cython.double, i=cython.Py_ssize_t)
def integrate_f(a, b, N):
    dx = (b-a)/N
    s = 0
    for i in range(N):
        s += f(a+i*dx)
    return s * dx
"""

cimport cython

cpdef double f(double x)

@cython.locals(dx=double, s=double, i=int)
cpdef integrate_f(double a, double b, int N)

