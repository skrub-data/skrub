#!/usr/bin/env python

from distutils.core import setup
import sys

import dirty_cat

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist', 'bdist_egg', 'bdist_dumb',
            'bdist_rpm', 'bdist_wheel', 'bdist_wininst', 'install_egg_info',
            'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}


if __name__ == '__main__':
    setup(name='dirty_cat',
          version=dirty_cat.__version__,
          author='Patricio Cerda',
          author_email='patricio.cerda@inria.fr',
          url='https://github.com/dirty-cat/dirty_cat',
          description=("Machine learning with dirty categories."),
          long_description=dirty_cat.__doc__,
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
          packages=['dirty_cat', ],
          **extra_setuptools_args)
