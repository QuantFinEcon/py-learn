# py-learn

Personal notes of a python developer

## About this GitRepo

A place where I save all my nuggets of self learning on Python as well as small projects

## Notes

- deploying package 
    - sys.path
    - .pth in site-packages to add paths to sys.path (like configuring PYTHONPATH) on initialisation of interpreter --> via site.py
    - \__init__.py in parent_folder(import parent_folder) and subfolders(import parent_folder.subfolder) is to create a namespace and            only import modules.py from its folder, outside the folder, or files inside other subfolders outside folder. PEP8 says keep it          flat, not nested so import pandas as __pd to avoid calling import mypackage.pd 

- Logging
    - metaclass

- Design Patterns
    - abc / interfaces
    - collections

- memory management
    - gc
    - weakref
    - psutil / resource
    - objgraph

- Front End Client-Server RESTful object tranmission
    - django (Full Stack)
    - flask (Non Full Stack)
    - http.client
    - wsgiref
    - json

- Parallel Processing
    - dask
    - multiprocessing
    - threading
    - asyncio
    - concurrent.futures

- Automation
    - win32com.client
    - pythoncom

- Configuration
    - pickle
    - YAML
    - JSON

- Security
    - hashlib
    - base64


## References

References I read to gain mastery of Python

- [*Design Patterns*](https://sourcemaking.com/design_patterns/)
- [*Mastering Object-Oriented Python*](https://www.bookdepository.com/Mastering-Object-Oriented-Python-Steven-Lott/9781783280971?ref=grid-view&qid=1520657285926&sr=1-1)
- [*Python Patterns/Idioms* by @faif](https://github.com/faif/python-patterns)
- [*Python Designs Receipes Idioms](http://python-3-patterns-idioms-test.readthedocs.io/)

## Design Patterns

A collection of design patterns and idioms in Python.

When an implementation is added or modified, be sure to update this file and
rerun `append_output.sh` (eg. ./append_output.sh borg.py) to keep the output
comments at the bottom up to date.

Current Patterns:

__Creational Patterns__:

| Pattern | Description |
|:-------:| ----------- |
| [abstract_factory](design-patterns/creational/abstract_factory.py) | use a generic function with specific factories |
| [borg](design-patterns/creational/borg.py) | a singleton with shared-state among instances |
| [builder](design-patterns/creational/builder.py) | instead of using multiple constructors, builder object receives parameters and returns constructed objects |
| [factory_method](design-patterns/creational/factory_method.py) | delegate a specialized function/method to create instances |
| [lazy_evaluation](design-patterns/creational/lazy_evaluation.py) | lazily-evaluated property pattern in Python |
| [pool](design-patterns/creational/pool.py) | preinstantiate and maintain a group of instances of the same type |
| [prototype](design-patterns/creational/prototype.py) | use a factory and clones of a prototype for new instances (if instantiation is expensive) |

__Structural Patterns__:

| Pattern | Description |
|:-------:| ----------- |
| [3-tier](design-patterns/structural/3-tier.py) | data<->business logic<->presentation separation (strict relationships) |
| [adapter](design-patterns/structural/adapter.py) | adapt one interface to another using a white-list |
| [bridge](design-patterns/structural/bridge.py) | a client-provider middleman to soften interface changes |
| [composite](design-patterns/structural/composite.py) | lets clients treat individual objects and compositions uniformly |
| [decorator](design-patterns/structural/decorator.py) | wrap functionality with other functionality in order to affect outputs |
| [facade](design-patterns/structural/facade.py) | use one class as an API to a number of others |
| [flyweight](design-patterns/structural/flyweight.py) | transparently reuse existing instances of objects with similar/identical state |
| [front_controller](design-patterns/structural/front_controller.py) | single handler requests coming to the application |
| [mvc](design-patterns/structural/mvc.py) | model<->view<->controller (non-strict relationships) |
| [proxy](design-patterns/structural/proxy.py) | an object funnels operations to something else |

__Behavioral Patterns__:

| Pattern | Description |
|:-------:| ----------- |
| [chain](design-patterns/behavioral/chain.py) | apply a chain of successive handlers to try and process the data |
| [catalog](design-patterns/behavioral/catalog.py) | general methods will call different specialized methods based on construction parameter |
| [chaining_method](design-patterns/behavioral/chaining_method.py) | continue callback next object method |
| [command](design-patterns/behavioral/command.py) | bundle a command and arguments to call later |
| [iterator](design-patterns/behavioral/iterator.py) | traverse a container and access the container's elements |
| [mediator](design-patterns/behavioral/mediator.py) | an object that knows how to connect other objects and act as a proxy |
| [memento](design-patterns/behavioral/memento.py) | generate an opaque token that can be used to go back to a previous state |
| [observer](design-patterns/behavioral/observer.py) | provide a callback for notification of events/changes to data |
| [publish_subscribe](design-patterns/behavioral/publish_subscribe.py) | a source syndicates events/data to 0+ registered listeners |
| [registry](design-patterns/behavioral/registry.py) | keep track of all subclasses of a given class |
| [specification](design-patterns/behavioral/specification.py) |  business rules can be recombined by chaining the business rules together using boolean logic |
| [state](design-patterns/behavioral/state.py) | logic is organized into a discrete number of potential states and the next state that can be transitioned to |
| [strategy](design-patterns/behavioral/strategy.py) | selectable operations over the same data |
| [template](design-patterns/behavioral/template.py) | an object imposes a structure but takes pluggable components |
| [visitor](design-patterns/behavioral/visitor.py) | invoke a callback for all items of a collection |

__Design for Testability Patterns__:

| Pattern | Description |
|:-------:| ----------- |
| [setter_injection](design-patterns/dft/setter_injection.py) | the client provides the depended-on object to the SUT via the setter injection (implementation variant of dependency injection) |

__Fundamental Patterns__:

| Pattern | Description |
|:-------:| ----------- |
| [delegation_pattern](design-patterns/fundamental/delegation_pattern.py) | an object handles a request by delegating to a second object (the delegate) |

__Others__:

| Pattern | Description |
|:-------:| ----------- |
| [blackboard](design-patterns/other/blackboard.py) | architectural model, assemble different sub-system knowledge to build a solution, AI approach - non gang of four pattern |
| [graph_search](design-patterns/other/graph_search.py) | graphing algorithms - non gang of four pattern |
| [hsm](design-patterns/other/hsm/hsm.py) | hierarchical state machine - non gang of four pattern |


## Contributing

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/J-YSM/py-learn/issues).

## License

This repository contains a variety of content; some developed by J., and some from third-parties.  The third-party content is distributed under the license provided by those parties.

The content developed by J. is distributed under the following license:

    Copyright 2018 J.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.