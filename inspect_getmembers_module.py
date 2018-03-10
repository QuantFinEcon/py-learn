import inspect

import os
os.getcwd()
os.chdir('C://Users//Justin//Dropbox//SCB Credit Risk//PY-HOME-TRAINING')

import example



for name, data in inspect.getmembers(example):
    # members can be classes, functions, methods or attributes of classes
    # or objects of classes
    if name.startswith('__'):
#         Modules have several private attributes that are used as part of the 
#         import implementation as well as a set of __builtins__. 
#         All of these are ignored in the output for this example because 
#         they are not actually part of the module and the list is long.
        continue
    print('{} : {!r}'.format(name, data))



