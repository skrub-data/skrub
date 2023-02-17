#!/bin/bash -e

pip install --progress-bar off --only-binary :all: --upgrade ".[$DEPS_VERSION]"
