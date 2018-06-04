# -*- coding: utf-8 -*-
"""
learn-list
"""


import math

class StatsList(list):
    """ lazy eval """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    @property
    def mean(self):
        return sum(self) / len(self)
    @property
    def stdev(self):
        n = len(self)
        return math.sqrt( n*sum(x**2 for x in self)-sum(self)**2)/n

x=StatsList([1,2,3])
x.mean
x.stdev

#==============================================================================
# 
#==============================================================================

class StatsList2(list):
    """Eager Stats."""
    def __init__( self, *args, **kw ):
        self.sum0 = 0 # len(self)
        self.sum1 = 0 # sum(self)
        self.sum2 = 0 # sum(x**2 for x in self)
        super().__init__( *args, **kw )
        for x in self:
            self._new(x)

    def _new( self, value ):
        self.sum0 += 1
        self.sum1 += value
        self.sum2 += value*value

    def _rmv( self, value ):
        self.sum0 -= 1
        self.sum1 -= value
        self.sum2 -= value*value

    def insert( self, index, value ):
        super().insert( index, value )
        self._new(value)

    def pop( self, index=0 ):
        value= super().pop( index )
        self._rmv(value)
        return value

    @property
    def mean(self):
        return self.sum1/self.sum0
    @property
    def stdev(self):
        return math.sqrt( self.sum0*self.sum2-self.sum1*self.sum1
                         )/self.sum0

    def __setitem__( self, index, value ):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
        olds = [ self[i] for i in range(start,stop,step) ]
        super().__setitem__( index, value )
        for x in olds:
            self._rmv(x)
        for x in value:
            self._new(x)
        else: # int 
            old= self[index]
            super().__setitem__( index, value )
            self._rmv(old)
            self._new(value)

    def __delitem__( self, index ):
    # Index may be a single integer, or a slice
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            olds = [ self[i] for i in range(start,stop,step) ]
            super().__delitem__( index )
            for x in olds:
                self._rmv(x)
        else:
            old = self[index]
            super().__delitem__( index )
            self._rmv(old)


x=StatsList2([1,2,3])
x.stdev

#==============================================================================
# 
#==============================================================================

class Explore(list):
    def __getitem__( self, index ):
        print(type(index))
        print(dir(index))
        print( index, index.indices(len(self)) )
        return super().__getitem__( index )

x=Explore("1bcde")
x[0:3:2]
x[:]
x[:-1]

slice

#==============================================================================
# 
#==============================================================================












