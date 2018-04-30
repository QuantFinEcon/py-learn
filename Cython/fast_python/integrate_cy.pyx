cdef extern from "math.h":
    double sin(double x)

cdef double f(double x):
    return sin(x**2)

cpdef double integrate_f(double a, double b, int N):
    cdef double dx, s
    cdef int i

    dx = (b-a)/N
    s = 0
    for i in range(N):
        s += f(a+i*dx)
    return s * dx

