# -*- coding: utf-8 -*-
"""
learn-mutability
"""

class ImmutableObj(object):
    __slots__ = ('x','y') # turn off __dict__ 
    def __init__(self,x,y):
        super().__setattr__('x',x) #use object's 
        super().__setattr__('y',y)
    def __str__(self):
        return "str: x={0.x} y={0.y}".format(self)
    def __repr__(self):
        return "repr: x={0.x} y={0.y}".format(self)
    def __setattr__(self,name,value):
        print("name={name} value={value}".format(name=name,value=value))
        print("No setting allowed after init")
        super.__setattr__(name,value) # self[name] = value
    def __getattr__( self, name ):
        print("name={name}".format(name=name))
        return self.get(name, None) # None force to look for solution
        
a=ImmutableObj(1,2)
a
print(a)        
a.x=1
object.__setattr__(a,'x',100) # multi inheritance MRO bypass
print(a)

#==============================================================================
# extend immutable obj
#==============================================================================

class NewTuple(tuple):
    def __new__(cls,*args):
        return super().__new__(cls,(args))
    def __getattr__(self,name):
        print("getattr")
        try:
            _fixedDict = {"x":1,"y":2}
            return self[_fixedDict[name]]
        except:
            print("{name} not found".format(name=name))
            raise AttributeError
    def __setattr__(self,index,value):
        print("cannot set attr. overriden!")
        
a=NewTuple(1,2,3)
a.a
a.x
a.x=1


a=lambda x: x
a(1)

#==============================================================================
# 
#==============================================================================


class NewList(list):
    def __init__(self,x,y):
        self.x = x
        self.y = y        
    def __setattr__(self,name,value):
        if name in self.__dict__:
            raise AttributeError( "Cannot set {name}".
                                 format(name=name) )
        else:
            super().__setattr__(name,value)
    def __getattribute__( self, name ):
        if name.startswith('_'): 
            raise AttributeError
        return object.__getattribute__( self, name )

a=NewList(1,2)
a.x=1

class test:
    a=1
    def __getattribute__( self, name ):
        if name.startswith('_'): 
            raise AttributeError
        return object.__getattribute__( self, name )

dir(test())