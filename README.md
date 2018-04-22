# py-learn

Personal notes of a python developer

## About this GitRepo

A place where I save all my nuggets of learnings on Pytthon as well as small projects
Why do this? To keep my codes all organised w backups!

## Notes

1. deploying package 
    - sys.path
    - .pth in site-packages to add paths to sys.path (like configuring PYTHONPATH) on initialisation of interpreter --> via site.py
    - \__init__.py in parent_folder(import parent_folder) and subfolders(import parent_folder.subfolder) is to create a namespace and            only import modules.py from its folder, outside the folder, or files inside other subfolders outside folder. PEP8 says keep it          flat, not nested so import pandas as __pd to avoid calling import mypackage.pd 

2. logging 
    - class decorator for all methods of class

3. design patterns
    - abc raise ImplementationError

4. asynchronous multithreading

5. memory management
    - gc
    - psutil / resource
    - objgraph

6. front end interactive web programming
    - django
    - reactJS

7. parallel processing
    - dask

8. automation
    - win32com.client
    - pythoncom

9. Configuration
    - YAML
    - JSON

## References

Books I read to gain mastery of Python

- [*Design Patterns*](https://sourcemaking.com/design_patterns/)
- [*Mastering Object-Oriented Python*](https://www.bookdepository.com/Mastering-Object-Oriented-Python-Steven-Lott/9781783280971?ref=grid-view&qid=1520657285926&sr=1-1)
- [*Python Patterns/Idioms* by @faif](https://github.com/faif/python-patterns)
- [*Python Designs Receipes Idioms](http://python-3-patterns-idioms-test.readthedocs.io/)

## Contributing

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/BigPyQuant/silver-adventure/issues).

## License

This repository contains a variety of content; some developed by Yeo Shui Ming, and some from third-parties.  The third-party content is distributed under the license provided by those parties.

The content developed by Yeo Shui Ming is distributed under the following license:

    Copyright 2018 Yeo Shui Ming

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

