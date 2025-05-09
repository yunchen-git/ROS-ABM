try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    # from distutils.core import setup
    # from distutils.extension import Extension
    print("error, please install setuptools")
    
import numpy as np

from Cython.Build import cythonize
"""
This script is meant to compile the intercellular diffusion module and extracellular diffusion module to C-code. 

To compile go to a command console and:
1. Change to this directory (type cd, press space then drag/drop this folder to the console. Press enter)
2. Type out: "python setup.py build_ext --inplace" without the quotation marks. Press enter.

"""

setup(
    ext_modules=cythonize(["IntC_diff.pyx", "ExtC_gradient.pyx"], compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)
