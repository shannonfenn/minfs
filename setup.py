try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Distutils import Extension, build_ext
import numpy as np
import sys
import glob
import os

include_dirs = [np.get_include()]

args = sys.argv[1:]

# get rid of intermediate and library files
if "clean" in args:
    print("Deleting cython files...")
    to_remove = []
    to_remove += glob.glob('minfs/*.c')
    to_remove += glob.glob('minfs/*.cpp')
    to_remove += glob.glob('minfs/*.so')
    for f in to_remove:
        os.remove(f)


# We want to always use build_ext --inplace
if args.count('build_ext') > 0 and args.count('--inplace') == 0:
    sys.argv.insert(sys.argv.index('build_ext')+1, '--inplace')

extensions = [
    Extension('minfs.utils',
              ['minfs/utils.pyx'],
              language='c++',
              include_dirs=include_dirs),
    Extension('minfs.reduction_rules',
              ['minfs/reduction_rules.pyx'],
              language='c++',
              include_dirs=include_dirs),
    # Extension('minfs.fs_solver_raps',
    #           ['minfs/fs_solver_raps.pyx'],
    #           language='c++',
    #           include_dirs=include_dirs),
    ]

setup(
    name='minfs',
    include_dirs=[np.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions
    )
