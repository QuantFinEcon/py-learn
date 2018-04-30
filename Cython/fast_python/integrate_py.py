from math import sin

def f(x):
    return sin(x**2)

def integrate_f(a, b, N):
    dx = (b-a)/N
    s = 0
    for i in range(N):
        s += f(a+i*dx)
    return s * dx

