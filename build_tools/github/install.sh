#!/bin/bash -e

pip install --progress-bar off --only-binary :all: --no-binary liac-arff --upgrade ".[$DEPS_VERSION]"
