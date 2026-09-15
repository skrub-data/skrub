Release process
===============

Target audience
---------------

This document is aimed at established contributors to the project.

Process
-------

Going further, we assume you have write-access to both the repository, PyPI and
conda-forge project page. For PyPI, this requires having an API key to identify
the build.

.. note:: We follow scikit-learn versioning conventions:

   - Major/Minor releases are numbered X.Y.0.
   - Bug-fix releases are done as needed between major/minor releases and only apply to
     the last stable version. These releases are numbered X.Y.Z.
   - Bug-fix releases should never include breaking changes (changes in behavior,
     deprecations, changes in the minimum requirements ...).

Before starting the process
---------------------------

Make sure that your working tree is clean
and that you are up to date with ``upstream/main`` by doing ``git fetch upstream``.
Additionally, a python environment that includes the ``twine`` and ``build`` packages
is needed for some of the steps in the process, so it should be prepared beforehand.

Install and test the wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is not necessary, but it's good to do to make sure that everything is
working and that no stray tests are there.

The important thing is that we don't install the package from the same folder
skrub is in, to make sure that pytest does not pick up any of the tests from
the source folder and only looks at what is inside the wheel.

.. note::

   pytest here is not using the ``addopts`` specified in ``pyproject.toml``, so there
   may be some tests that fail because of that.

.. code-block:: shell

   # From the skrub dir, create the wheel
   python -m build

   # move to a different dir
   cd /tmp

   # create a new env with pip/uv
   uv venv release-X.Y.Z

   # activate the env
   source release-X.Y.Z/bin/activate

   # install the wheel in the env
   uv pip install ~/work/skrub/dist/PATH_TO_WHEEL

   # install pytest and numpydoc and run tests
   uv pip install pytest numpydoc
   pytest --pyargs skrub


Steps for a minor release
--------------------------

The example below goes from 0.1.0 to 0.2.0.

Preparing the release branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create the ``0.2.X`` branch from ``upstream/main`` and push it upstream:

  .. code:: shell

     git fetch --all
     git checkout upstream/main
     git checkout -b 0.2.X
     git push upstream 0.2.X

  .. note::

     If push to upstream has been disabled in your local git config (e.g. with
     ``pushurl = push-to-upstream-is-disabled``), you need to re-enable it temporarily
     for this step. You can also use the GitHub UI to create the branch instead.

     Note that we push directly to ``upstream`` here (not to ``origin``), which is why
     the next step is to open a PR.

- Edit ``CHANGES.rst``: replace "Ongoing development" with ``0.2.0``.
- Edit ``VERSION.txt``: replace ``0.2.dev0`` with ``0.2.0``.

  .. note::

     ``pyproject.toml`` infers the package version from this file:

     .. code-block:: toml

        [tool.setuptools.dynamic]
        version = { file = "skrub/VERSION.txt" }

- Commit the changes to ``CHANGES.rst`` and ``VERSION.txt`` and push to
  ``upstream/0.2.X``:

  .. code:: shell

     git push --set-upstream upstream 0.2.X

- Open a PR targeting ``0.2.X``. This will update the doc for the stable release. While
  the update runs, we can prepare a PR on the main branch to be merged after the
  release, see the next section.

Meanwhile, preparing the post-release PR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new branch from ``upstream/main`` (e.g. ``post-0.2.0``) and open a PR
targeting ``main``.

- For a major/minor (not a patch) release:

  - ``VERSION.txt``: update to ``0.3.dev0`` (the next minor).
  - ``CHANGES.rst``: create a header for the new entries ("Ongoing development").
  - ``doc/version.json``: update the version numbers of the stable release and dev
    branch, and add an entry for the previously stable version. For example, going from
    0.1.0 to 0.2.0 it should go from:

    .. code-block:: json

       [
           {
               "name": "0.2.dev0 (dev)",
               "version": "0.2.dev0",
               "url": "https://skrub-data.org/dev/"
           },
           {
               "name": "0.1.0 (stable)",
               "version": "0.1.0",
               "url": "https://skrub-data.org/stable/",
               "preferred": true
           }
       ]

    to:

    .. code-block:: json

       [
           {
               "name": "0.3.dev0 (dev)",
               "version": "0.3.dev0",
               "url": "https://skrub-data.org/dev/"
           },
           {
               "name": "0.2.0 (stable)",
               "version": "0.2.0",
               "url": "https://skrub-data.org/stable/",
               "preferred": true
           },
           {
               "name": "0.1.0",
               "version": "0.1.0",
               "url": "https://skrub-data.org/0.1/"
           }
       ]

The doc update has succeeded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Wait for the CI to finish building the documentation for the ``0.2.X`` branch. Go
  over the examples, the changelog, and other pages to double check that everything is
  being rendered correctly, because issues there often go unnoticed. On GitHub, you can
  select the ``0.2.X`` branch to see the CI status for that branch.

- Merge the PR targeting ``0.2.X``, **without squashing the commits**.

.. warning::

   This PR should be merged with the rebase mode instead of the usual squash mode
   because we want to keep the history in the ``0.2.X`` branch close to the history of
   the main branch, which will help for future bug fix releases.

   By default, only the squash & merge option is available to merge PRs on the main
   branch. So, when releasing, we need to temporarily enable the rebase option.
   To do so, head to Settings -> General -> Pull request, enable rebasing, merge the
   PR targeting ``0.2.X`` with the rebase option, then disable the setting again.


Pushing the wheel to PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Checkout to the release candidate branch after merging the PR:

  .. code:: shell

     git fetch upstream
     git checkout upstream/0.2.X

- Build the wheel and test it:

  - ``rm -rf dist skrub.egg-info``
  - ``python -m build`` (may need ``pip install build``)
  - ``twine check dist/*`` (may need ``pip install twine``)

  .. tip::

     The build version should match the release number (e.g. ``0.2.0``). If it still
     shows ``dev0`` at the end, something is wrong with ``VERSION.txt``.

  - See `Install and test the wheel`_ above.

- If the tests passed successfully, upload to PyPI: ``twine upload dist/*``.

  .. note::

     This step will ask for a PyPI API key. You need to have been added to the list of
     maintainers on PyPI to do this.

- Tag the release commit and push the tag:

  .. code:: shell

     # -s is for signing and is optional
     git tag -s '0.2.0'
     git push upstream tag 0.2.0

- Check that your version is now on PyPI.
- Merge the post-release PR.


Updating the website repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For major/minor releases only, update the documentation symlink in the website
repository https://github.com/skrub-data/skrub-data.github.io:

.. code:: shell

   git clone git@github.com:skrub-data/skrub-data.github.io.git
   cd skrub-data.github.io
   git fetch

   # update the symlink to point to the new stable version
   rm stable
   ln -s 0.2 stable

   git add stable
   git commit -m "setting 0.2 as stable"
   git push

``stable`` should point to the latest numbered release.


Update the conda-forge recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These steps are done from the
`skrub-feedstock <https://github.com/conda-forge/skrub-feedstock>`_ repository.

- Fork the feedstock repository if you haven't already, then clone your fork and set
  up the upstream remote:

  .. code:: shell

     git remote add upstream git@github.com:conda-forge/skrub-feedstock.git
     git fetch upstream
     git checkout upstream/main
     git checkout -b release-0.2.0

- Edit ``recipe/meta.yml``, which is the only file we edit manually in that repo:

  - Update the version number.
  - Update the sha256 using the PyPI hash. The hash can be found on the PyPI page for
    skrub under "Download files" → "view details" for the source distribution.
  - If needed, reset the build number to 0.
  - If needed, update the requirements:

    .. code:: shell

       git checkout 0.2.0
       git diff 0.1.0 -- pyproject.toml

  .. note::

     If the build fails, the build number needs to be incremented by 1 for each
     failure until it passes.

- Open a PR targeting ``upstream/main`` in skrub-feedstock **from your fork**.
- Use the checklist posted in the PR template. In particular, post a comment
  ``@conda-forge-admin, please rerender`` to trigger the bot to re-render the recipe.
  Make sure to wait until it has finished.
- Merge the PR. It takes up to an hour for the package to be available from the
  conda-forge channel.
- When it becomes available, install it in a fresh environment and run tests.

.. note::

   You can add new maintainers to that repo by listing them at the end of
   ``meta.yml``.


Announcing the release
^^^^^^^^^^^^^^^^^^^^^^^

- Prepare the release discussion on GitHub.
- Write a LinkedIn post.
- Write a Bluesky post.
- Write an announcement on Discord.


Steps for a bugfix release
---------------------------

The example below goes from 0.7.0 to 0.7.1. The ``0.7.X`` branch already exists from
the previous minor release.

Preparing the release branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create a branch from ``upstream/main`` targeting ``upstream/0.7.X`` (e.g.
  ``release-0.7.1``) and rebase it onto the existing release branch:

  .. code:: shell

     git fetch --all
     git checkout upstream/main
     git checkout -b release-0.7.1
     git rebase -i upstream/0.7.X

  ``git rebase -i upstream/0.7.X`` replays the commits from ``main`` on top of
  ``upstream/0.7.X`` and opens an interactive editor so you can select which commits to
  keep.

- Edit ``CHANGES.rst``: add the release header (``0.7.1``), remove the "Ongoing
  development" header, and clean up the changelog if needed.
- Edit ``VERSION.txt``: replace ``0.7.0`` with ``0.7.1``.
- Commit the changes and push the branch:

  .. code:: shell

     git push --set-upstream origin release-0.7.1

- Open a PR targeting ``upstream/0.7.X`` (not ``main``). Updating ``0.7.X`` is what
  triggers the stable version of the skrub website to be rebuilt.

.. warning::

   This PR should be merged with the rebase mode instead of the usual squash mode.
   See the warning in `The doc update has succeeded`_ above for instructions on
   temporarily enabling the rebase option on GitHub.

- Wait for the CI to finish building the documentation. Double check the rendering of
  the changelog and the examples.
- Merge the PR.

Meanwhile, open a post-release PR on ``main`` (e.g. ``post-release-0.7.1``):

- No need to update ``VERSION.txt`` for a bugfix release.
- ``CHANGES.rst``: add a new "Ongoing development" header for new entries. Make sure it
  includes the ``0.7.1`` header from the release branch.
- ``doc/version.json``: update the stable version to ``0.7.1``.
- Merge the post-release PR.


Pushing the wheel to PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Checkout the release candidate branch after merging the PR:

  .. code:: shell

     git fetch upstream
     git checkout upstream/0.7.X

- Build the wheel and test it:

  - ``rm -rf dist skrub.egg-info``
  - ``python -m build`` (may need ``pip install build``)
  - ``twine check dist/*`` (may need ``pip install twine``)
  - See `Install and test the wheel`_ above.

- If the tests passed successfully, upload to PyPI: ``twine upload dist/*``.
- Tag the release commit and push the tag:

  .. code:: shell

     git tag -s '0.7.1'
     git push upstream tag 0.7.1

- Check that your version is now on PyPI.


Update the conda-forge recipe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the same steps as for a minor release (see `Update the conda-forge recipe`_
above), using ``release-0.7.1`` as the branch name and the ``0.7.1`` version number.


Announcing the release
^^^^^^^^^^^^^^^^^^^^^^^

- Prepare the release discussion on GitHub.
- Write a LinkedIn post.
- Write a Bluesky post.
- Write an announcement on Discord.
