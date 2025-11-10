import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


CYTHON_COMPILER_DIRECTIVES = {
    "language_level": 3,
}
CYTHON_DEFINE_MACROS = []
if os.getenv('CYTHON_COVERAGE', False):
    CYTHON_COMPILER_DIRECTIVES["linetrace"] = True
    CYTHON_DEFINE_MACROS.append(('CYTHON_TRACE_NOGIL', '1'))

# C++ infrastructure configuration
CPP_INCLUDE_DIRS = [
    'pycalphad/cpp/include',
    'pycalphad/cpp/utils',
]

# C++ utility sources (to be compiled with Cython extensions as needed)
CPP_UTIL_SOURCES = [
    'pycalphad/cpp/utils/numpy_wrapper.cpp',
    'pycalphad/cpp/utils/lapack_wrapper.cpp',
    'pycalphad/cpp/utils/memory_utils.cpp',
]

# Find LAPACK/BLAS libraries
def find_lapack_libs():
    """Try to find LAPACK and BLAS libraries on the system."""
    # For macOS, use Accelerate framework
    if sys.platform == 'darwin':
        return [], ['-framework', 'Accelerate']

    # For other platforms, try to find libraries
    # scipy includes lapack/blas bindings, so they should be available
    return ['lapack', 'blas'], []

LAPACK_LIBS, EXTRA_LINK_ARGS = find_lapack_libs()

CYTHON_EXTENSION_INCLUDES = ['.', np.get_include()] + CPP_INCLUDE_DIRS
CYTHON_EXTENSION_MODULES = [
    Extension('pycalphad.core.hyperplane',
              sources=['pycalphad/core/hyperplane.pyx'],
              include_dirs=CYTHON_EXTENSION_INCLUDES,
              define_macros=CYTHON_DEFINE_MACROS,
              libraries=LAPACK_LIBS,
              extra_link_args=EXTRA_LINK_ARGS),
    Extension('pycalphad.core.eqsolver',
              sources=['pycalphad/core/eqsolver.pyx'],
              include_dirs=CYTHON_EXTENSION_INCLUDES,
              define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.phase_rec',
              sources=['pycalphad/core/phase_rec.pyx'],
              include_dirs=CYTHON_EXTENSION_INCLUDES,
              define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.composition_set',
              sources=['pycalphad/core/composition_set.pyx'],
              include_dirs=CYTHON_EXTENSION_INCLUDES,
              define_macros=CYTHON_DEFINE_MACROS),
    Extension('pycalphad.core.minimizer',
              sources=['pycalphad/core/minimizer.pyx'],
              include_dirs=CYTHON_EXTENSION_INCLUDES,
              define_macros=CYTHON_DEFINE_MACROS,
              libraries=LAPACK_LIBS,
              extra_link_args=EXTRA_LINK_ARGS),
]

# https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
mingw32_link_args = [
    "-static-libgcc",
    "-static-libstdc++",
    "-Wl,-Bstatic,--whole-archive",
    "-lwinpthread",
    "-Wl,--no-whole-archive",
]

class Build(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "mingw32":
            for ext in self.extensions:
                ext.extra_link_args = mingw32_link_args
        return super().build_extensions()

setup(
    ext_modules=cythonize(
        CYTHON_EXTENSION_MODULES,
        include_path=CYTHON_EXTENSION_INCLUDES,
        compiler_directives=CYTHON_COMPILER_DIRECTIVES,
    ),
    cmdclass={"build_ext": Build},
    package_data={
        'pycalphad.core': ['*.pxd'] + (['*.pyx', '*.c', '*.h', '*.cpp', '*.hpp'] if os.getenv('CYTHON_COVERAGE', False) else []),
        'pycalphad.tests.databases': ['*'],
        'pycalphad.cpp': ['**/*.hpp', '**/*.h', '**/*.cpp'],
    },
    # This include is for the compiler to find the *.h files during the build_ext phase
    # the include must contain a symengine directory with header files
    include_dirs=[np.get_include()],
)
