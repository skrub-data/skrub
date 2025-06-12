Release process
===============

Target audience
---------------

This document is aimed at established contributors to the project.

Process
-------

Going further, we assume you have write-access to both the repository, PyPI and
conda-forge project page.

.. note:: We follow scikit-learn versioning conventions:

   - Major/Minor releases are numbered X.Y.0.
   - Bug-fix releases are done as needed between major/minor releases and only apply to
     the last stable version. These releases are numbered X.Y.Z.

To release a new minor version of skrub (e.g., from 0.1.0 to 0.2.0), here are the main
steps and appropriate resources: the main steps and appropriate resources:

Preparing the release branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create the ``0.2.X`` branch, branching from upstream/main, and push it upstream
  (it may already exist). You can also use the GitHub UI to create the branch if you
  disabled ``git push upstream`` in your local git config.
- Edit CHANGES.rst: replace "ongoing development" with ``0.2.0``
- Edit VERSION.txt: replace ``0.2.dev0`` with ``0.2.0``
- Build the wheel and test it:

  - ``rm -r dist skrub.egg-info``
  - ``python -m build`` (may need ``pip install build``)
  - ``twine check dist/*`` (may need ``pip install twine``)
  - In a directory outside of the skrub repo:

    - Install the wheel in a fresh virtualenv
    - Run all tests with ``pytest --pyargs skrub``

- git commit the changes done to CHANGES.rst and VERSION.txt
- If we are doing a bugfix release (``0.2.X`` already existed before) we need to rebase
  on the existing ``0.2.X``.

  - Run ``git rebase -i upstream/0.2.X``
  - All commits that have been made on main that we want to keep will be replayed on
    top of the last release's tag in ``0.2.X``.

- Open a PR targeting ``0.2.X``. This will update the doc for the stable release. While
  the update runs, we can prepare a PR on the main branch to be merged after the
  release, see the next section.

Meanwhile, preparing the post-released PR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- For a major/minor (not a patch) release:
    - VERSION.txt: update to ``0.3.dev0`` (the next minor).
    - CHANGES.rst: create a header for the new entries ("ongoing development").
    - doc/version.json: update the version numbers of the stable release and dev branch.


The doc update has succeeded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Merge the PR targeting ``0.2.X``, **without squashing the commits**.

.. warning::

    This PR should be merged with the rebase mode instead of the usual squash mode
    because we want to keep the history in the ``0.2.X`` branch close to the history of
    the main branch, which will help for future bug fix releases.

    By default, only the squash & merge option is available to merge PRs on the main
    branch. So, when releasing, we need to temporarily enable the rebase option.
    To do so, head to Settings -> General -> Pull request, enable rebasing, merge the
    PR targeting ``0.2.X`` with the rebase option, then disable the setting again.

- Check the rendering of the doc for the built ``0.2.X`` branch, the examples and the
  changelog. Ideally, we should go over all features and double check that the docs are
  being rendered correctly, because issues there often go unnoticed.


Next, we'll build the wheel and push it to Pypi!


Pushing the wheel to Pypi
^^^^^^^^^^^^^^^^^^^^^^^^^

- Checkout to the release candidate branch:

  .. code:: shell

     git fetch upstream
     git checkout upstream/0.2.X

- Build the wheel and test it:

  - ``rm -r dist skrub.egg-info``
  - ``python -m build`` (may need ``pip install build``)
  - ``twine check dist/*`` (may need ``pip install twine``)
  - In a directory outside of the skrub repo:

    - Install the wheel in a fresh virtualenv
    - Run all tests with ``pytest --pyargs skrub``

- If test passed successfully, upload to Pypi: ``twine upload dist/*``.
- Tag the release commit and push the tag:

  - ``git tag -s '0.2.0'``, ``-s`` is for signing and is optional.
  - ``git push upstream tag 0.2.0``

- Check that your version is now on Pypi.
- Merge the post-release PR
- For major/minor releases only, in the documentation branches repository
  https://github.com/skrub-data/skrub-data.github.io, update the documentation symlink
  to stable version, here from 0.1 to 0.2:

  .. code:: shell

     rm stable
     ln -s 0.2 stable

  ``stable`` should point on the latest number release.


Update the conda-forge recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create the branch ``release-0.2.0`` in
  `skrub-feedstock <https://github.com/conda-forge/skrub-feedstock>`_
- Edit ``recipe/meta.yml``, which is the only file we edit manually in that repo:
    - Update the version number.
    - Update the sha256 using Pypi hash.
    - If needed, reset the build number to 0.
    - If needed, update the requirements.

      - Check the new requirements with:

        .. code:: shell

           git checkout 0.2.0
           git diff 0.1.0 -- pyproject.toml

- Open a PR targeting ``upstream/skrub-feedstock`` main branch.
- Use the the checklist posted in the PR template. In particular, it asks to post a
  comment asking a bot to re-render the recipe. Make sure to wait until it has finished.
- Merge the PR. It takes up to an hour for the package to be available from the
  conda-forge channel.
- When it becomes available, install it in a fresh environment and run tests.

.. note::

   You can add new maintainers to that repo by listing them at the end of meta.yml.

- If the new recipe works fine, announce the release on social network channels 🎉!
