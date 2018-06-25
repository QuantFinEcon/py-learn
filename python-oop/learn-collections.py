# -*- coding: utf-8 -*-
"""
learn-collections
https://docs.python.org/3/library/collections.html#collections.UserList
"""

import collections
import builtins
vars(builtins)

[x for x in dir(collections) if not x.startswith("_")]

collections.ChainMap?
x = collections.ChainMap(locals(), globals(), vars(builtins))
for k,v in x.items():
    print(k, v)







