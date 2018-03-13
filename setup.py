#!/usr/bin/env python

from distutils.core import setup
import sys

import _sample

VERSION = _sample.__version__
DESCRIPTION = _sample.__doc__

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist', 'bdist_egg', 'bdist_dumb',
            'bdist_rpm', 'bdist_wheel', 'bdist_wininst', 'install_egg_info',
            'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}


if __name__ == '__main__':
    setup(name='dirty_cat',
          version=VERSION,
          author='Patricio Cerda',
          author_email='patricio.cerda@inria.fr',
          url='https://github.com/dirty-cat/dirty_cat',
          description=("Machine learning with dirty categories."),
          long_description=DESCRIPTION,
          license='BSD',
          classifiers=[
              'Development Status :: 3 - Production/Stable',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'License :: OSI Approved :: BSD License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          platforms='any',
          packages=['dirty_cat'],
          install_requires=['jellyfish', 'python-Levenshtein',
                            'distance', 'sklearn', 'numpy', 'scipy'],
          **extra_setuptools_args)
