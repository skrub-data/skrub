.. _tutorial_write_example:

.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

How to write an example for the gallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial explains to new contributors how to format their examples so that
they are properly rendered in the skrub documentation gallery.

While examples are written in plain Python code, there are some quirks to be aware of
when writing them, due to the way Sphinx and the sphinx-gallery extension work.
This tutorial explains these quirks and how to work around them.

Location of the examples
-----------------------

Once you decide on the subject of your example, start writing the code as a Python
script. Place the script in the ``examples/`` folder of the repository. The example
should be self-contained and runnable as a standalone script. The documentation is
built by executing the code and generating additional content from it.

The name of the file should start with a number, followed by an underscore,
and then a short description of the example. The number is used to order the examples
in the documentation. For instance, if your example is about using the
|TableVectorizer| class, you might want to name the file ``01_table_vectorizer.py``.

Note that the ``examples/`` folder is covered by ``pre-commit`` hooks, which run
various checks on your code when you try to commit. These checks may block you from
pushing.

Dealing with typos in the example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your code includes any kind of intentional typo, for example if you are trying
to correct names by replacing a string with a typo with the new one, the
``codespell`` hook will block your commit. To bypass this, update ``pyproject.toml``
by adding the typo to the ``ignore-word-list`` entry in the ``tool.codespell``
section. After this, commit the updated ``pyproject.toml`` file using
``git commit --no-verify`` to bypass local checks so that following commits will
ignore the typos.
Note that without updating ``pyproject.toml``, the CI will still reject commits
with typos, as it runs the same hooks that are run locally.

Writing the example
-----------------------
Your python script should start with a docstring that briefly explains what the example
is about. This docstring can contain multiple paragraphs and will be rendered
as a RST file in the documentation, so you can use RST syntax
in it.

Importantly, the first line of the docstring should be the title of the example,
not an RST directive (such as ``.. replace::`` or ``.. note::``). Sphinx
adds a reference to the example at the top of the page using the file name as the
title. Adding a directive at the top of the docstring would prevent proper HTML
rendering.

This is an example of what the beginning of your example may look like:

.. code-block:: python

    """
    Title of the example
    ====================

    This is a brief description of the example. It can contain multiple paragraphs,
    and it can use RST syntax.

    .. note::

        You can use RST directives in the docstring, such as ``.. note::``,
        ``.. warning::``, ``.. seealso::``, etc.

    After the definition of the title, you may also add directives such as
    ``.. replace::``, and they will be rendered properly. For example, you can add:

    .. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`


    """

Then, you can start writing the code for the example. The content of your Python script
should be a sequence of code cells, each delimited by a line starting with ``# %%``.
These code cells may contain comments, which will be rendered as rst in the final
documentation.

After the docstring, write the code for your example as a sequence of code cells,
each delimited by a line starting with ``# %%``. Comments in these cells will be
rendered as RST in the final documentation.

.. code-block:: python

    # %%
    # This is a comment that will be rendered as markdown in the final documentation.
    # You can use multiple lines for comments, and you can use RST syntax in them.

    import pandas as pd
    from skrub import TableVectorizer

    # %%
    # This is another code cell. You can write any python code here.
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["a", "b", "c"]
    })
    tv = TableVectorizer()
    X = tv.fit_transform(df)
    print(X)

Running the example
-------------------

Once you have written the code for the example (or while writing it), you can run
it to see how it looks in the final documentation. Depending on your setup, you
may need to install some dependencies. Refer to your IDE's documentation for more
information on running interactive Python scripts. For example, VSCode documentation
is available `here <https://code.visualstudio.com/docs/python/jupyter-support-py>`_.

Once you are happy with your example, you can submit a pull request to the repository,
following the instructions in the :ref:`contributing guide <contributing>`.

Adding cross-references
-----------------------

Adding cross-references to the documentation helps users find more information
about the concepts and functions used in your example. This step is optional, and
you may ask the maintainers for help on which cross-references to add. Good
cross-references include relevant user guide sections, the documentation of the
objects used in the example (like the |TableVectorizer|), or other examples.

You can add cross-references in the docstring and comments of your example in several ways:

- You can add references to the objects in the skrub API using the ``:class:`~skrub.ClassName```
  or ``:func:`~skrub.function_name``` directives.
- If your example uses the same objects multiple times, you can define a replacement at the top
  of the docstring using the ``.. replace::`` directive, and then use the replacement
  instead of the full directive.
- You can also add references to other sections of the documentation using the
  ``:ref:`label``` directive, where ``label`` is the label of the section you want to reference.


For example, if your example uses the |TableVectorizer| class multiple times, define
a replacement at the top of the docstring. You may also want to add a reference
to the user guide section about the |TableVectorizer| class. This can be done as follows:

.. code-block:: python

    """
    Title of the example
    ====================

    .. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

    This example demonstrates how to use the |TableVectorizer| class to vectorize a dataframe.

    See the :ref:`user_guide_building_pipeline_index` guide for more information about the |TableVectorizer| class.
    """

    # %%
    import pandas as pd
    from skrub import TableVectorizer

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["a", "b", "c"]
    })
    tv = TableVectorizer()
    X = tv.fit_transform(df)
    print(X)

You may find more information on the cross-references in the
`official Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/referencing.html>`_.


Generating the new documentation
-------------------------------
Once you have written your example and added any necessary cross-references, you can
generate the new documentation to see how it looks. This can be done in two ways:

- You can run the commands ``make html`` or ``make html-noplot`` in the ``doc/``
  folder of the repository to generate the HTML documentation for the entire project.
- Alternatively, you can use ``pixi run -e doc build-doc`` or ``pixi run -e doc build-doc-quick``
  from the root folder to generate the documentation. The advantage of using ``pixi`` is that
  it automatically sets up a virtual environment with the necessary dependencies, so you
  don't need to worry about installing them manually.

The ``make html`` and ``pixi run -e doc build-doc`` commands generate complete
documentation by executing all example code. The ``-noplot`` (or ``-quick``)
versions skip code execution, making documentation generation much faster. Use
these faster versions to check formatting when you've already tested your example
code locally.

The CI pipeline will always run the full documentation build, so you can safely
use ``make html-noplot`` or ``pixi run -e doc build-doc-quick`` for local testing.


After generating the documentation, open the ``index.html`` file in the ``doc/_build/html/``
folder with a web browser to review the results. Check that:

- Section titles are properly formatted.
- Any formatting in docstrings or comments is rendered as intended. For example,
  Sphinx uses spaces to delimit lists and code blocks, so if you have them in the
  example, make sure that they render correctly.
- Cross-references are working. You can check the logs of the Sphinx
  generation to see if there are any broken references.


Linking your work to examples already in the documentation
----------------------------------------------------------
After generating the documentation, you may want to add references to your example
in other relevant parts of the documentation. This helps users find your example
when reading about related topics.


This step is done after generating the documentation because you need the final
reference name, which is created dynamically from your file name. For example,
if your file is named ``99_my_example.py``:

1. The generated files will be in ``doc/auto_examples``
2. A reference file will be created at ``doc/auto_examples/99_my_example.rst``
3. The reference label will be ``.. _sphx_glr_auto_examples_99_my_example.py``

To link to your example from other documentation pages, use:

.. code-block:: rst

    :ref:`sphx_glr_auto_examples_99_my_example.py`



Merging your example
-----------------------
Finally, if everything looks good, commit your changes and submit a pull request
to the repository. For more information, see the :ref:`contributing guide <contributing>`.


Your PR will be reviewed by the maintainers, who may suggest changes or improvements.
Once approved, it will be merged into the main branch, and your example will
become part of the official documentation. Thank you!
