"""
:Authors: - Wilker Aziz
"""
from setuptools import setup, find_packages

try:
    import numpy as np
except ImportError:
    raise ImportError("First you need to run: pip install numpy")


"""
try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError('First you need to run: pip install cython')


ext_modules = cythonize('**/*.pyx',
                        language='c++',
                        language_level=3)
"""

ext_modules = []

setup(
    name='dgm4nlp',
    license='Apache 2.0',
    author='Wilker Aziz',
    description='VAEs for NLP',
    packages=find_packages(),
    install_requirements=['tabulate'],
    include_dirs=[np.get_include()],
    ext_modules=ext_modules,
)
