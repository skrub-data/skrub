#!/bin/bash -e

if [[ "$DEPS_VERSION" == "nightly" ]]; then
    echo "Installing development dependency wheels"
    dev_anaconda_url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple
    pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy pandas scikit-learn scipy
    
    # pyarrow nightly builds are not hosted on anaconda.org
    pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ --prefer-binary --pre pyarrow

    pip install --progress-bar off --only-binary :all: --no-binary liac-arff
else
    pip install --progress-bar off --only-binary :all: --no-binary liac-arff --upgrade ".[$DEPS_VERSION]"
fi
