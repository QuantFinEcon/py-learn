
import this


class A:
    pass

dir(A)
['__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__']

A.__base__
A.__class__
A.__class__.__base__
type.__base__











class A:
    pass

class B:
    def test():
        print('justin')

a = {1:A, 2:B}

a.get(1,'J')
a.get(3,'J')
a.get(2,'J').test()



import collections
b = collections.defaultdict(lambda: 'J', a)
b.get(1)

import functools

functools.partial?


testing123






