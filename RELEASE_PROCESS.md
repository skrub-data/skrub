## Release process

### Target audience

This document is aimed toward established contributors the project.

If you're a user looking to install dirty_cat, please refer to
the instructions in the [README](README.rst)!

### Process

> Going further, we assume you have write-access to both the repository and
> the PyPI project page.

> It is useful to publish a beta version of the package before the actual one.

To release a new version of dirty_cat, here are the main steps to follow:

1. Update `dirty_cat/CHANGES.rst`. It should be updated at each PR,
   but double-checking before the release is good practice
2. Update `dirty_cat/dirty_cat/VERSION.txt` with the new version number
3. Update `dirty_cat/setup.py`
4. Make a new branch in order to make a PR
5. You have to commit the changes with a new tag: the version you're
   going to push (e.g. `1.0`) with the commands
   `git commit -m "Preparing for release 1.0"`, `git tag 1.0`, `git push --tags`
6. Next, you will need to install the `twine` package with `pip install twine`
7. Build the source with `python setup.py bdist_wheel sdist`
8. [Check if today is a good day for releasing](https://shouldideploy.today/)
9. It is advised to first push the version on the test package index
   `test.pypi.org` before the official package index `pypi.org`.
   You can do this with the command
   `twine upload dist/* --repository-url https://test.pypi.org/legacy/`
   This is useful to test if the build done by twine and the push to the
   package indexer is working
10. Create a new Python environment and install the package from the test PyPi
11. Install the new release from the test package index on a dedicated environment
    with the command `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ dirty_cat`
12. Finally, if that works fine, you can push to the official package index with
    `twine upload dist/*`
13. It is also good practice at this point to create a new environment
    and try installing and using the library (for example by launching examples).
    Be sure to install it with the command `pip install dirty_cat==<version>`
    (e.g. `pip install dirty_cat==1.0`), otherwise some package/env managers such as conda
    will use the cached version.
14. To finish the procedure, create a new release on the GitHub repository

### Resources

- [Packaging and distributing software - Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
- [Publishing (Perfect) Python Packages on PyPi](https://youtu.be/GIF3LaRqgXo)
- [Managing releases in a repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Should I deploy today?](https://shouldideploy.today/)
