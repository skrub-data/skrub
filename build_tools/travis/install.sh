
if [[ "$SCIKIT_LEARN_VERSION" == "stable" ]]; then
  pip install scikit-learn

elif [[ "$SCIKIT_LEARN_VERSION" == "nightly" ]]; then
  # https://scikit-learn.org/stable/developers/advanced_installation.html#installing-nightly-builds
  pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn
fi
