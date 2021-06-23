#!/bin/bash -e

python -m pip install --progress-bar off --upgrade pip setuptools wheel flake8
if [ ! -z "$MIN_REQUIREMENTS" ]; then
    pip install --progress-bar off --upgrade -r requirements-min.txt
else
    pip install --progress-bar off --upgrade -r requirements.txt
fi
