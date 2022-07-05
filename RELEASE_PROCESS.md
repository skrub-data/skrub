## Release process

### Target audience

This document is aimed toward established contributors the project.

If you're a user looking to install dirty_cat, please refer to
the instructions in the [README](README.rst) !

### Process

> Going further, we assume you have write-access to both the repository and 
> the PyPI project page.

To release a new version of dirty_cat,
here are the main steps and appropriate resources:

1. Update `dirty_cat/CHANGES.rst`. It should be updated at each PR,
   but double-checking before the release is good practice.
2. Update `dirty_cat/dirty_cat/VERSION.txt` with the new version number
3. Update `dirty_cat/setup.py`
4. You have to commit the changes with a new tag: 
   the version you're going to push (e.g. `0.1`)
5. Next, you will need to install the `twine` package with `pip install twine`
6. Build the source with `python setup.py bdist_wheel sdist`
7. [Check if today is a good day for releasing](https://shouldideploy.today/)
8. It is advised to first push the version on the test package index 
   `test.pypi.org` before the official package index `pypi.org`.
   You can do this with the command
   `twine upload dist/* --repository-url https://test.pypi.org/legacy/`
9. Install the new release from the test package index on a dedicated environment
10. Finally, if that works fine, you can push to the official package index with
    `twine upload dist/*`
11. It is also good practice at this point to create a new environment
    and try installing and using the library.
12. To finish the procedure, create a new release on the GitHub repository.

### Resources

- [Packaging and distributing software - Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
- [Publishing (Perfect) Python Packages on PyPi](https://youtu.be/GIF3LaRqgXo)
- [Should I deploy today ?](https://shouldideploy.today/)
