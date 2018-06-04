# -*- coding: utf-8 -*-
"""
learn-descriptors
"""

#==============================================================================
# methods as dynamic attributes 
#==============================================================================

from random import randint

class list_wrapper(object):

    def __init__(self, iterator=None):
        if iterator:
            self.length = len(iterator)
            self._x = list(iterator)
        else:
            self.length = 0
            self._x = list()

    @classmethod
    def random(cls, n):
        return cls([randint(-1000,1000) for _ in range(n)])

    @classmethod
    def random2(cls, n):
        obj = cls.__new__(cls) # wont call __init__
        obj.length = n
        obj._x = [randint(-1000,1000) for _ in range(n)]
        return obj
    
    @staticmethod
    def var():
        print("no instance")
    
    @property #getter
    def x(self):
        return self._x
    @x.setter
    def x(self, element):
        self.length += 1 # eager ompute, bind to attributes
        self._x.append(element)
    @x.deleter
    def x(self):
        self.length -= 1
        self._x.pop(-1)

list_wrapper()
a=list_wrapper.random(10)
a.x
a.x = 9999
a.length
del a.x

list_wrapper.random2(10).x

#==============================================================================
# create descriptors - __get__ (non data don't have) and __set__ and __del__
#==============================================================================

class UnitValue_1:
    """Measure and Unit combined."""
    def __init__( self, unit ):
        self.value= None # mutable
        self.unit= unit # immutable
        self.default_format= "5.2f"
    def __set__( self, instance, value ): #only so NON DATA 
        self.value= value
    def __str__( self ):
        return "{value:{spec}} {unit}".format( spec=self.default_format, **self.__dict__)
    def __format__( self, spec="5.2f" ):
        #print( "formatting", spec )
        if spec == "": 
            spec= self.default_format
        return "{value:{spec}} {unit}".format( spec=spec,
                **self.__dict__)

class RTD_1:
    """ descriptors as cls attributes """
    rate= UnitValue_1( "kt" )
    time= UnitValue_1( "hr" )
    distance= UnitValue_1( "nm" )

    def __init__( self, rate=None, time=None, distance=None ):
        if rate is None:
            self.time = time
            self.distance = distance
            self.rate = distance / time
        if time is None:
            self.rate = rate
            self.distance = distance
            self.time = distance / rate
        if distance is None:
            self.rate = rate
            self.time = time
            self.distance = rate * time
    def __str__( self ):
        # access cls attrs
        return "rate: {0.rate} time: {0.time} distance: {0.distance}".format(self)

m1 = RTD_1( rate=5.8, distance=12 )
str(m1)

#==============================================================================
# data descriptor
#==============================================================================

class Unit:
    """ superclass of data descriptor """
    conversion= 1.0
    def __get__( self, instance, owner ): #auto conversion
        return instance.kph * self.conversion
    def __set__( self, instance, value ):
        instance.kph = value / self.conversion

class KPH( Unit ):
    def __get__( self, instance, owner ): #auto conversion
        return instance._kph # return _kph instead of kph, avoid INF recursion
    def __set__( self, instance, value ):
        instance._kph = value
        
class Knots( Unit ):
    conversion= 0.5399568

class MPH( Unit ):
    conversion= 0.62137119

class Measurement:
    """ Owner owning data descriptors objs
        Share the same superclass Unit """
    kph= KPH()
    knots= Knots()
    mph= MPH()

    def __init__( self, kph=None, mph=None, knots=None ):
        if kph: self.kph= kph
        elif mph: self.mph= mph
        elif knots: self.knots= knots
        else:
            raise TypeError
    def __str__( self ):
        return "rate: {0.kph} kph = {0.mph} mph = {0.knots} knots".format(self)

m2 = Measurement(knots=5.9)
str(m2)