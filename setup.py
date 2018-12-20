#!/usr/bin/env python

import os
from setuptools import setup, find_packages

version_file = os.path.join(
    os.path.dirname(__file__), 'dirty_cat', 'VERSION.txt')
with open(version_file) as fh:
    VERSION = fh.read().strip()

description_file = os.path.join(os.path.dirname(__file__), 'README.rst')
with open(description_file) as fh:
    DESCRIPTION = fh.read()


if __name__ == '__main__':
    setup(name='dirty_cat',
          version=VERSION,
          author='Patricio Cerda',
          author_email='patricio.cerda@inria.fr',
          url='http://dirty-cat.github.io/',
          description=("Machine learning with dirty categories."),
          long_description=DESCRIPTION,
          license='BSD',
          classifiers=[
              'Development Status :: 2 - Pre-Alpha',
              'Environment :: Console',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Topic :: Scientific/Engineering',
              'Topic :: Software Development :: Libraries',
          ],
          platforms='any',
          packages=find_packages(),
          package_data={'dirty_cat': ['VERSION.txt', 'data/midwest_survey/*.csv']},
          install_requires=['scikit-learn>=0.20', 'numpy', 'scipy', 'requests',
                            'joblib'],
          )
