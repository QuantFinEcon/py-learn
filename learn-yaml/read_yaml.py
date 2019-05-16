import os


os.getcwd()

os.chdir(r'C:\Users\yeoshuiming\Dropbox\GitHub\py-learn\learn-yaml')

import yaml

with open("anchor.yml", 'r') as stream:
    try:
        x=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
x

"""
# Anchors can be used to duplicate/inherit properties
base: &base
  name: Everyone has same name

# The regexp << is called Merge Key Language-Independent Type. It is used to
# indicate that all the keys of one or more specified maps should be inserted
# into the current map.

foo: &foo
  <<: *base
  age: 10

bar: &bar
  <<: *foo
  twice_age: 20
"""

with open("collections.yml", 'r') as stream:
    try:
        x=yaml.load(stream)
#        x=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

x

yaml.SafeLoader.yaml_constructors
yaml.Loader.yaml_constructors


with open("scalar_types.yml", 'r') as stream:
    try:
        x=yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

x

