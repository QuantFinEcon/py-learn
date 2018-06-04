# -*- coding: utf-8 -*-
"""
learn-callables
"""

import timeit

def fib1(n):
    if n<=0: return 0
    elif n==1: return 1
    else:
        return fib1(n-1) + fib1(n-2)
    
timeit.timeit("fib1(10)",
              """ 
def fib1(n):
    if n<=0: return 0
    elif n==1: return 1
    else:
        return fib1(n-1) + fib1(n-2)
              """,
              number=1000)

%timeit -r 10 fib1(10)

#==============================================================================
# callable object, stateful function
#==============================================================================

import collections

class fibb(collections.abc.Callable):
    """ function-like object """
    def __init__(self):
        self.cache= {"0":0,"1":1}
        
    def __call__(self, n):
        if str(n) not in self.cache:
            if n <= 0:
                return self.cache["0"]
            elif n == 1:
                return self.cache["1"]
            else:
                self.cache[str(n)] = self.__call__(n-2) + self.__call__(n-1)
        
        return self.cache[str(n)]

fib2 = fibb()

fib2(10)
fib2.cache

%timeit -r 10 fib2(10)

#==============================================================================
# memonisation deocrator
#==============================================================================

class Memoize(collections.abc.Callable):
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        ret = self.func(*args)
        self.cache[args] = ret
        return ret

@Memoize
def fib3(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib3(n-2) + fib3(n-1)

fib3(10)
fib3.cache

%timeit -r 10 fib3(10)


#==============================================================================
# 
#==============================================================================

def fib(a, cache={0:1,1:1}):
    if a in cache: return cache[a]                                                                                 
    res = fib(a-1, cache) + fib(a-2, cache)                                                                        
    cache[a] = res                                                                                                 
    return res  

#==============================================================================
# 
#==============================================================================

from functools import lru_cache
lru_cache?

@lru_cache(maxsize=1000)
def fib1(n):
    if n<=0: return 0
    elif n==1: return 1
    else:
        return fib1(n-1) + fib1(n-2)
    
%timeit -r 10 fib1(10)










    
