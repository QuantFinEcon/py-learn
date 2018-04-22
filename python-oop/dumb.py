@init.py

from logger.exception_decorator import *
#import different types of logger using logging module
from logger.exception_logger import *


instructions="""
#########
# Setup
#########
Set LOGTO "constant" to path of exception catching output e.g. C:/folder/log.txt

############
# Functions
############
1. catch_exception - decorate f with try catch
2. logfunction - decorator for functions
3. logmethod - decorator for functions. Use this for class or instance methods, it formats with the object out front
4. logclass - Since python has no suport for class decorator, use it as higher order function. Setup with cls = logclass(cls)
5. LogMetaClass - 
	class Test():
		__metadata__ = LogMetaClass
6. logmodule(pandas)	
7. logging_metaclass - use as metaclass to use 1.catch_exception for all methods in class
8. class_decorator(..logger..)
"""

print(instructions)

@exception_logger.py

import logging
import os as __os
 
def create_logger(LOGTO=__os.getcwd()+"\\logger.txt" , level = logging.INFO):
    """
    Creates a logging object and returns it
    how to use?
    catch_exception( create_logger(destination_path, level = logging.INFO) )
    """
    print("Saving log to " + str(LOGTO))
    logger = logging.getLogger("example_logger")
    logger.setLevel(level)
 
    # create the logging file handler
    fh = logging.FileHandler(LOGTO)
 
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger

#logger = create_logger()

@exception_decorator.py


from functools import wraps


def catch_exception(logger):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    @param logger: The logging object

    ##############
    Example
    ###############
    class test():
        def __init__(self):
            pass
        @log.catch_exception(logger)
        def method(self,a=1):
            an_undeclared_var_to_cause_error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # log the exception
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)
 
            # re-raise the exception to interpreter
            raise
        return wrapper
    return decorator

def class_decorator(decorator):
    """
    Decorate class so that all its methods is decorated with the decorator.
    There's propably a better way to do this since will also decorate nested classes.
    Special methods like __init__ are not excluded
    ##############
    Example
    ###############
    @class_decorator(log.catch_exception(logger))
    class test():    
        def __init__(self):
            pass    
        def method(self):
            an_undeclared_var_to_cause_error
    """
    @wraps(decorator)
    def decorate(cls):
        for attr in cls.__dict__: 
            print("methodname: " + attr)
            if callable(getattr(cls, attr)) \
            and attr not in ['__init__','__module__', '__qualname__']:
                print(cls, attr)
                # cls.attr = decorator(getattr(cls, attr))
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

class logging_metaclass(type):
    """
    Decorate class so that all its methods is decorated with the decorator.
    There's propably a better way to do this since will also decorate nested classes.
    Special methods like __init__ are not excluded
    This is a new type so __module__ and __qualname__ not excluded.
    
    ##############
    Example
    ###############
    class test(metaclass = logging_metaclass, logger=log.catch_exception(logger)):
        # logger as kwargs ---> passed into Metaclass **kwargs namespace
        def __init__(self):
            pass    
        def method(self):
            an_undeclared_var_to_cause_error
    """
    def __new__(self, class_name, bases, local_namespace, **kwargs):
        #kwargs dict is passed from class kwargs
        for methodname in local_namespace:
            print("methodname: " + methodname)
            if callable(local_namespace[methodname]) \
            and methodname not in ['__init__','__module__', '__qualname__']:
                print("method " + str(local_namespace[methodname]) + " is callable")
                local_namespace[methodname] = \
                kwargs["logger"](local_namespace[methodname])
        return type.__new__(self, class_name, bases, local_namespace)

@test_logger.py


import logger as log
import unittest

#dir(unittest.TestCase)
#unittest.TestCase.assertRaises?

class TestLogger(unittest.TestCase):
    
    def test_raise_exception(self):

        logger=log.create_logger()    
        @log.catch_exception(logger)
        def zero_divide():
            1 / 0
        self.assertRaises(RuntimeError, zero_divide())

    def test_class_decorator(self):

        logger=log.create_logger()    
        @log.class_decorator(log.catch_exception(logger))
        class test():    
            def __init__(self):
                pass    
            def method(self):
                an_undeclared_var_to_cause_error

        self.assertRaises(RuntimeError, test().method())


    def test_logging_metaclass(self):
    
        logger=log.create_logger()    
        class test2(metaclass = log.logging_metaclass, logger=log.catch_exception(logger)):
            # logger as kwargs ---> passed into Metaclass **kwargs namespace
            def __init__(self):
                pass    
            def method(self):
                an_undeclared_var_to_cause_error 
        
        self.assertRaises(RuntimeError, test2().method())


if __name__ == '__main__':
    unittest.main()


