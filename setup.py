################################################################################
# File: setup.py
# Purpose:
#   Build script for compiling the Cython diffusion modules:
#   - IntC_diff.pyx (intercellular transport)
#   - ExtC_gradient.pyx (extracellular diffusion/fields)
#
# Key public API:
#   - Standard setuptools setup(...) entry for `python setup.py build_ext --inplace`
#
# How it fits in the codebase:
#   Produces compiled extension modules used by Simulation.py at runtime.
#
# Inputs/Outputs:
#   - Outputs platform-specific compiled artifacts in-place (e.g., *.so / *.pyd).
#
# Notes / gotchas:
#   - Uses tracked generated C sources by default for stable local rebuilds.
#   - Set ROS_ABM_USE_CYTHON=1 to regenerate C from .pyx sources.
#   - Requires a working compiler toolchain.
################################################################################
import os
from pathlib import Path

try:
    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    # from distutils.core import setup
    # from distutils.extension import Extension
    print("error, please install setuptools")
    
import numpy as np

"""
This script is meant to compile the intercellular diffusion module and extracellular diffusion module to C-code. 

To compile go to a command console and:
1. Change to this directory (type cd, press space then drag/drop this folder to the console. Press enter)
2. Type out: "python setup.py build_ext --inplace" without the quotation marks. Press enter.

"""

CYTHON_SOURCES = ["IntC_diff.pyx", "ExtC_gradient.pyx"]
GENERATED_C_SOURCES = ["IntC_diff.c", "ExtC_gradient.c"]

use_cython = os.environ.get("ROS_ABM_USE_CYTHON") == "1"
generated_c_available = all(Path(source).exists() for source in GENERATED_C_SOURCES)
cythonize = None

if use_cython or not generated_c_available:
    try:
        from Cython.Build import cythonize
    except ImportError as exc:
        raise RuntimeError(
            "Cython is required when generated C sources are unavailable "
            "or ROS_ABM_USE_CYTHON=1 is set."
        ) from exc
    sources = CYTHON_SOURCES
else:
    sources = GENERATED_C_SOURCES

extensions = [
    Extension(
        source.rsplit(".", 1)[0],
        [source],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for source in sources
]

if cythonize is not None:
    extensions = cythonize(extensions, compiler_directives={"language_level": "3"})


class ForceBuildExt(_build_ext):
    """Rebuild in-place extensions so stale tracked binaries do not mask ABI drift."""

    def finalize_options(self):
        super().finalize_options()
        self.force = True


setup(
    ext_modules=extensions,
    cmdclass={"build_ext": ForceBuildExt},
)
