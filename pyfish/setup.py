
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

FLUIDS_INC = "/home/who/Documents/CALResearch/BinaryTurbulance/fluids/include"
PYFLUIDS_INC = "/home/who/Documents/CALResearch/BinaryTurbulance/fluids/pyfluids"

fluids = Extension("fish",
                   sources = ["fish.pyx"],
                   library_dirs = ['../lib'],
                   libraries = ['fish'],
                   include_dirs=["../include",
                                 FLUIDS_INC,
                                 PYFLUIDS_INC,
                                 np.get_include()])

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [fluids])
