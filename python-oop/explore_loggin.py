
import logging # interface to manage global state
logging.getLogger(__name__) # for entire module


import sys
logging.basicConfig( stream=sys.stderr, level=logging.DEBUG )
'''
default level is WARN
configure a StreamHandler instance that will write to sys.stderr

multiple handlers ,multiple dest for errors
• We might want duplicate logs to improve the reliability of operations.
• We might be using the sophisticated Filter objects to create distinct
subsets of messages.
• We might have different levels for each destination. We can use this to
separate debugging messages from informational messages.
• We might have different handlers based on the Logger names to
represent different foci.

configure once, decentralised at top of all apps
only configure logging inside the if __name__ == "__main__": portion of an application
'''
logging.basicConfig?


'''
config files for multiple loggers, ea with their handler
'''
import os 
os.getcwd()
os.chdir("C:\\Users\\1580873\\Desktop\\Completed_Developments\\Log Debug Deploy")

import logging.config
import yaml
config = "config.yml"
config_dict= yaml.load(open(config))
logging.config.dictConfig(config_dict)

verbose= logging.getLogger( "verbose.example.SomeClass" )
audit= logging.getLogger( "audit.example.SomeClass" )

verbose.info( "Verbose information" )
audit.info( "Audit record with before and after" )




#Having multiple logger objects available in a class allows us to finely control the
#kinds of output. We can configure each Logger to have different handlers
def test():
    log=logging.getLogger("test")
    print(1)
    print(a)
test()


# add custom level to catch
logging.addLevelName(15, "VERBOSE")
logging.VERBOSE= 15



# class shared decorator for all methods
def logged(cls):
    cls.logger= logging.getLogger( cls.__qualname__ )
    return cls

def desciptor(cls):
    print("wraping in layers")
    return cls

@desciptor
@logged
class Player:
    def __init__( self, bet, strategy, stake ):
        # share by all instances of the class
        self.logger= logging.getLogger( self.__class__.__qualname__ )
        self.logger.debug( "init bet {0}, strategy {1}, stake {2}".
                          format(bet, strategy, stake) )



import sys

if __name__ == "__main__":
    logging.config.dictConfig(yaml.load("log_config.yaml"))
try:
    application = Main()
    status = application.run()
except Exception as e:
    logging.exception(e)
    status = 2
finally:
    logging.shutdown()
sys.exit(status)


#==============================================================================
# 
#==============================================================================


pg 459




#==============================================================================
# 
#==============================================================================
def f(cls):
    cls.abc = 1
    return cls

@f
class g(object):
    def __init__(self, a:int):
        self.a = a
    def test1():
        pass
    def test2():
        pass


b = g(a=2)
b.a
b.abc
dir(b)

b.__class__
b.__class__.__qualname__
b.__dict__

b1=g(a=6)
b1.a

b.abc=100
b1.abc


########################
# package
###################


cls -> module -> package


import logging

import pandas as pd

pd.DataFrame??
pd.api??



import sys

sys.platform
sys.api_version
sys.version
sys.path
sys.path??

sys.path.append("C:\\Users\\1580873\\Desktop\\RiskView")

import outlook_helpers

import argparse as argp
argp.__file__
argp.


def f(a,b,*args,**kwargs):
    print(args)
    print(kwargs)

f(1,2,3,4,5,k="hello")


# GC


import gc

# We are using ctypes to access our unreachable objects by memory address.
class PyObject(ctypes.Structure):
    _fields_ = [("refcnt", ctypes.c_long)]


gc.disable()  # Disable generational gc

lst = []
lst.append(lst)

# Store address of the list
lst_address = id(lst)

# Destroy the lst reference
del lst

object_1 = {}
object_2 = {}
object_1['obj2'] = object_2
object_2['obj1'] = object_1

obj_address = id(object_1)

# Destroy references
del object_1, object_2

# Uncomment if you want to manually run garbage collection process 
# gc.collect()

# Check the reference count
print(PyObject.from_address(obj_address).refcnt)
print(PyObject.from_address(lst_address).refcnt)


#==============================================================================
# https://docs.python.org/3/library/gc.html
#==============================================================================

import gc

gc.enable()
gc.set_debug(gc.DEBUG_LEAK)
print(gc.get_count())

#lst = []
#lst.append(lst)
#list_id = id(lst)
#del lst

gc.collect()
for item in gc.garbage:
    print(item)
    assert list_id == id(item)

def deep_purge_list( garbage ):
  for item in garbage:
    if isinstance( item, dict ):
      item.clear()
    if isinstance( item, list ):
      del item[:]
    try:
      item.__dict__.clear()
    except:
      pass
  del garbage[:]

deep_purge_list(gc.garbage)


gc.get_objects()
gc.is_tracked(df)

#==============================================================================
# 
#==============================================================================
The following constants are provided for use with set_debug():

gc.DEBUG_STATS
Print statistics during collection. This information can be useful when tuning the collection frequency.

gc.DEBUG_COLLECTABLE
Print information on collectable objects found.

gc.DEBUG_UNCOLLECTABLE
Print information of uncollectable objects found (objects which are not reachable but cannot be freed by the collector). These objects will be added to the garbage list.

gc.DEBUG_INSTANCES
When DEBUG_COLLECTABLE or DEBUG_UNCOLLECTABLE is set, print information about instance objects found.

gc.DEBUG_OBJECTS
When DEBUG_COLLECTABLE or DEBUG_UNCOLLECTABLE is set, print information about objects other than instance objects found.

gc.DEBUG_SAVEALL
When set, all unreachable objects found will be appended to garbage rather than being freed. This can be useful for debugging a leaking program.

gc.DEBUG_LEAK
The debugging flags necessary for the collector to print information about a leaking program (equal to DEBUG_COLLECTABLE | DEBUG_UNCOLLECTABLE | DEBUG_INSTANCES | DEBUG_OBJECTS | DEBUG_SAVEALL).

#==============================================================================
# 
#==============================================================================

import objgraph

objgraph.show_refs(df, filename='objgraph.png')
objgraph.show_backrefs(df, filename='sample-backref-graph.png')

objgraph.show_most_common_types()

#snapshot
objgraph.show_growth(limit=3) 

#after computation...
objgraph.show_growth()

#==============================================================================
# 
#==============================================================================

import guppy




import resource
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#==============================================================================
# 
#==============================================================================

import psutil, os
def usage():
    process = psutil.Process(os.getpid())
    return process.memory_full_info().rss / float(2 ** 20)

usage() # initial memory usage




arr = np.arange(10 ** 8) # create a large array without boxing
usage()
del arr
usage()

arr = np.arange(10 ** 8, dtype='O') # create lots of objects
usage()
del arr
usage()  # numpy frees the array, but python keeps the heap big



import sys
sys.getrefcount(df)
sys.getrefcount(chunks)





