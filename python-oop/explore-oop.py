# =============================================================================
# overload constructor
# =============================================================================

class MyData:
    def __init__(self, data):
        "Initialize MyData from a sequence"
        self.data = data
    
    @classmethod
    def fromfilename(cls, filename):
        "Initialize MyData from a file"
        data = open(filename).readlines()
        return cls(data)
    
    @classmethod
    def fromdict(cls, datadict):
        "Initialize MyData from a dict's items"
        return cls(datadict.items())
 
MyData([1, 2, 3]).data
MyData.fromfilename("/n tmp/nfoobar").data
MyData.fromdict({"spam": "ham"}).data
MyData.fromdict({"spam": "ham"})
d=MyData.fromdict({"spam": "ham"})


import time

class Date(metaclass=MultipleMeta):
    def __init__(self, year:int, month:int, day:int):
        self.year = year
        self.month = month
        self.day = day

    def __init__(self):
        t = time.localtime()
        self.__init__(t.tm_year, t.tm_mon, t.tm_mday)




# =============================================================================
# 
# =============================================================================
class A(object):
    def foo(self,x):
        print("executing foo(%s,%s)"%(self,x))

    @classmethod
    def class_foo(cls,x):
        print("executing class_foo(%s,%s)"%(cls,x))

    @staticmethod
    def static_foo(x):
        print("executing static_foo(%s)"%x)

a=A()

a.foo(1)
a.class_foo(1)
a.static_foo(1)



