from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("worker", ["worker.py"])]

setup(
  name = 'stupid little app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

""" python setup.py build_ext --inplace """