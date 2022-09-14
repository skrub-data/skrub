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

To release a new minor version of ``dirty_cat`` (e.g. 0.1.0 -> 0.2.0), here are
the main steps and appropriate resources:

1.  Update ``dirty_cat/CHANGES.rst``. It should be updated at each PR,
    but double-checking before the release is good practice.
2.  Create a branch by running ``git checkout -b 0.<version>.X``
    (e.g. ``0.1.X``).
3.  Update ``dirty_cat/dirty_cat/VERSION.txt`` with the new version
    number (e.g. ``1.0``).
4.  Update ``dirty_cat/setup.py`` (e.g. Python version supported and dependencies).
5.  You have to commit the changes with a new tag: the version you're
    going to push with the commands
    ``git commit -m "Bump version 0.1.0"``. Push the branch to the ``dirty_cat``
    repository. This will push and create an associated folder in the documentation
    repository.
6.  In the documentation repository (e.g. ``https://github.com/dirty-cat/dirty-cat.github.io``),
    update the ``stable`` symlink to the latest stable version: first, unlink ``stable``
    (i.e. ``unlink stable``); then, create a new symlink (i.e. ``ln -s 0.1 stable``);
    finally, commit and push the changes into the repository.
7.  Create a new release and tag via GitHub: ``https://github.com/dirty-cat/dirty_cat/releases/new``.
    Provide the tag using the current version, e.g. ``0.1.0`` and make sure to select
    the target branch created earlier (e.g. ``0.1.X``).
8.  Next, you will need to install the ``twine`` package with
    ``pip install --upgrade twine``
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
    ``pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ dirty_cat``
13. Finally, if that works fine, you can push to the official package
    index with ``twine upload dist/*``
14. It is also good practice at this point to create a new environment
    and try installing and using the library (for example by launching examples).
    Be sure to install it with the command ``pip install dirty_cat==<version>``
    (e.g. ``pip install dirty_cat==1.0``), otherwise some package/env managers
    such as conda might use a cached version.
15. To finish the procedure, create a new release on the GitHub repository.

For the bug fix release (e.g. 0.1.0 -> 0.1.1), the process is similar. You don't need
to create the branch ``0.1.X`` because it exists already. You need to cherry-pick the
commits from ``master`` into this branch and then follow the same steps as above:
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
