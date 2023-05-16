Release process
===============

Target audience
---------------

This document is aimed toward established contributors the project.


Process
-------

Going further, we assume you have write-access to both the repository
and the PyPI project page.

.. note::

   It is useful to publish a beta version of the package before the
   actual one.

To release a new minor version of ``skrub`` (e.g. 0.1.0 -> 0.2.0), here are
the main steps and appropriate resources:

1.  Update ``skrub/CHANGES.rst``. It should be updated at each PR,
    but double-checking before the release is good practice.
2.  Create a branch by running ``git checkout -b 0.<version>.X``
    (e.g. ``0.2.X``).
3.  Update ``skrub/skrub/VERSION.txt`` with the new version
    number (e.g. ``0.2.0``).
4.  Update ``skrub/setup.cfg`` (e.g. Python version supported and dependencies).
5.  Commit the changes with a new tag: the version you're going to push,
    with the commands ``git commit -m "Bump to version 0.2.0"`` and
    ``git push origin 0.2.0``.
    Push the branch to the ``skrub`` repository.
    The CI will automatically create an associated folder in the documentation repo.
6.  In the documentation repository (e.g. ``https://github.com/skrub-data/skrub-data.github.io``),
    update the ``stable`` symlink to the latest stable version: first, unlink ``stable``
    (i.e. ``unlink stable``); then, create a new symlink (i.e. ``ln -s 0.2 stable``);
    finally, commit and push the changes into the repository.
7.  Create a new release via GitHub: ``https://github.com/skrub-data/skrub/releases/new``.
    Provide the tag using the current version, e.g. ``0.2.0`` and make sure to select
    the target branch created earlier (e.g. ``0.2.X``).
8.  Next, you will need to install the ``twine`` and ``setuptools`` packages with
    ``pip install --upgrade twine setuptools``.
9.  Build the source with ``python setup.py bdist_wheel sdist``
10. `Check if today is a good day for releasing <https://shouldideploy.today/>`__
11. It is advised to first push the version on the test package index
    ``test.pypi.org`` before the official package index ``pypi.org``.
    You can do this with the command
    ``twine upload dist/* --repository-url https://test.pypi.org/legacy/``
    This is useful to test if the build done by twine and the push to
    the package indexer is working.
12. Install the new release from the test package index on a dedicated
    environment with the command
    ``pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/skrub``.
    If there are other package install errors, it probably means those packages are not on test.pypi.org.
    Download them from pypi.org directly with just ``pip install``.
13. Finally, if that works fine, you can push to the official package
    index with ``twine upload dist/*``
14. It is also good practice at this point to create a new environment
    and try installing and using the library (for example by launching examples).
    Be sure to install it with the command ``pip install skrub==<version>``
    (e.g. ``pip install skrub==0.2.0``), otherwise some package/env managers
    such as conda might use a cached version.
15. Set package version to ``<next_version>.dev0``, commit and push.

For the bug fix release (e.g. 0.2.0 -> 0.2.1), the process is similar. You don't need
to create the branch ``0.2.X`` because it exists already. You need to cherry-pick the
commits from ``main`` into this branch and then follow the same steps as above:
bumping the version, update the setup, commit and push the changes and finally create
and update the wheel.

Resources
---------

-  `Packaging and distributing software - Python Packaging User
   Guide <https://packaging.python.org/guides/distributing-packages-using-setuptools/>`__
-  `Publishing (Perfect) Python Packages on
   PyPi <https://youtu.be/GIF3LaRqgXo>`__
-  `Managing releases in a
   repository <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>`__
-  `Should I deploy today? <https://shouldideploy.today/>`__
