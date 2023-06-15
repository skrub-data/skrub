#!/usr/bin/env bash
set -x
set -e

# Decide what kind of documentation build to run, and run it.
#
# If the last commit message has a "[doc skip]" marker, do not build
# the doc. On the contrary if a "[doc build]" marker is found, build the doc
# instead of relying on the subsequent rules.
#
# We always build the documentation for jobs that are not related to a specific
# PR (e.g. a merge to master or a maintenance branch).
#
# If this is a PR, do a full build if there are some files in this PR that are
# under the "doc/" or "examples/" folders, otherwise perform a quick build.
#
# If the inspection of the current commit fails for any reason, the default
# behavior is to quick build the documentation.

get_build_type() {
    if [ -z "$CIRCLE_SHA1" ]
    then
        echo SKIP: undefined CIRCLE_SHA1
        return
    fi
    commit_msg=$(git log --format=%B -n 1 $CIRCLE_SHA1)
    if [ -z "$commit_msg" ]
    then
        echo QUICK BUILD: failed to inspect commit $CIRCLE_SHA1
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ skip\] ]]
    then
        echo SKIP: [doc skip] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ quick\] ]]
    then
        echo QUICK: [doc quick] marker found
        return
    fi
    if [[ "$commit_msg" =~ \[doc\ build\] ]]
    then
        echo BUILD: [doc build] marker found
        return
    fi
    if [ -z "$CI_PULL_REQUEST" ]
    then
        echo BUILD: not a pull request
        return
    fi
    git_range="origin/main...$CIRCLE_SHA1"
    git fetch origin main >&2 || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
    filenames=$(git diff --name-only $git_range)
    if [ -z "$filenames" ]
    then
        echo QUICK BUILD: no changed filenames for $git_range
        return
    fi
    changed_examples=$(echo "$filenames" | grep -e ^examples/)
    if [[ -n "$changed_examples" ]]
    then
        echo BUILD: detected examples/ filename modified in $git_range: $changed_examples
        pattern=$(echo "$changed_examples" | paste -sd '|')
        # pattern for examples to run is the last line of output
        echo "$pattern"
        return
    fi
    echo QUICK BUILD: no examples/ filename modified in $git_range:
    echo "$filenames"
}

build_type=$(get_build_type)
if [[ "$build_type" =~ ^SKIP ]]
then
    exit 0
fi

if [[ "$CIRCLE_BRANCH" =~ ^main$|^[0-9]+\.[0-9]+\.X$ && -z "$CI_PULL_REQUEST" ]]
then
    make_args="html"
elif [[ "$build_type" =~ ^QUICK ]]
then
    make_args=html-noplot
elif [[ "$build_type" =~ ^'BUILD: detected examples' ]]
then
    # pattern for examples to run is the last line of output
    pattern=$(echo "$build_type" | tail -n 1)
    make_args="html EXAMPLES_PATTERN=$pattern"
else
    make_args=html
fi

# Installing required system packages to support the rendering of math
# notation in the HTML documentation
sudo -E apt-get -yq update
sudo -E apt-get -yq remove texlive-binaries --purge
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes \
    install dvipng texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended\
    latexmk

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && bash ./miniconda.sh -b -p "miniconda"
export PATH="miniconda/bin:$PATH"

# Configure the conda environment and put it in the path using the
# provided versions
mamba create -n $CONDA_ENV_NAME --yes python="${PYTHON_VERSION:-*}" \
  numpy="${NUMPY_VERSION:-*}" scipy="${SCIPY_VERSION:-*}" \
  pytest coverage matplotlib="${MATPLOTLIB_VERSION:-*}" sphinx \
  seaborn statsmodels pillow cython joblib pandas="${PANDAS_VERSION:-*}"
#removed scikit learn from conda since it is installed from main after

source activate testenv

pip install .[doc]

pip install scikit-learn

# Build and install the project in dev mode
python setup.py develop

#if [[ "$CIRCLE_BRANCH" =~ ^main$ && -z "$CI_PULL_REQUEST" ]]
#then
#    # List available documentation versions if on default branch
#    python build_tools/circle/list_versions.py > doc/versions.rst
#fi

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $make_args 2>&1 | tee ~/log.txt

cd -
set +o pipefail

affected_doc_paths() {
    files=$(git diff --name-only origin/main...$CIRCLE_SHA1)
    echo "$files" | grep ^doc/.*\.rst | sed 's/^doc\/\(.*\)\.rst$/\1.html/'
    echo "$files" | grep ^examples/.*.py | sed 's/^\(.*\)\.py$/auto_\1.html/'
    project_files=$(echo "$files" | grep '^skrub/')
    if [ -n "$project_files" ]
    then
        grep -hlR -f<(echo "$project_files" | sed 's/^/skrub\/blob\/[a-z0-9]*\//') doc/_build/html/modules/generated | cut -d/ -f5-
    fi
}

if [ -n "$CI_PULL_REQUEST" ]
then
    echo "The following documentation files may have been changed by PR #$CI_PULL_REQUEST:"
    affected=$(affected_doc_paths)
    echo "$affected"
    (
    echo '<html><body><ul>'
    echo "$affected" | sed 's|.*|<li><a href="&">&</a></li>|'
    echo '</ul><p>General: <a href="index.html">Home</a> | <a href="modules/classes.html">API Reference</a> | <a href="auto_examples/index.html">Examples</a></p></body></html>'
    ) > 'doc/_build/html/_changed.html'
fi
