.. _tutorial_write_example:

How to write an example for the skrub gallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This page explains how to write examples for the skrub gallery. The main intention
behind this tutorial is to explain to new contributors how to format their examples
so that they are properly rendered in the documentation.

While examples are written in plain python code, there are some quirks to be aware of
when writing them that are caused by the way Sphinx and the sphinx-gallery extension
work. This tutorial explains these quirks and how to work around them.

Location of the examples
-----------------------

Once you decide the subject of your example, you can start writing the code for
it as a python script. The code for the example should be placed in a single
file in the ``examples/`` folder of the repository: the example should be
self-contained, and it should be possible to run it as a standalone script.
Indeed, the final documentation is built by executing the code and generating
additional content from it.

The name of the file should start with a number, followed by an underscore,
and then a short description of the example. The number is used to order the examples
in the documentation. For instance, if your example is about using the
``TableVectorizer`` class, you might want to name the file ``01_table_vectorizer.py``.

Note that the ``examples/`` folder is covered by the ``pre-commit`` hooks, which
means that various checks will be performed on the code when you try to commit it:
this might block you from pushing. In particular, if your code includes any kind
of typo, the ``codespell`` hook will block you from committing. You can run
``git commit --no-verify`` to bypass the checks, but it is recommended to fix
the issues instead.

Writing the example
-----------------------
Your python script should start with a docstring that briefly explains what the example
is about. This docstring can contain multiple paragraphs and will be rendered
as a ``.rst`` file in the documentation, which means that you can use ``.rst`` syntax
in it.

Importantly, the first line of the docstring should be the title of the example,
rather than any ``.rst`` directive, such as ``.. replace::`` or ``.. note::``:
this is because Sphinx will add a reference to the example at the top of the page,
using the name of the file as the title, and adding a directive at the top of
the docstring would prevent the HTML from rendering properly.

This is an example of what the beginning of your example may look like:

.. code-block:: python

    """
    Title of the example
    ====================

    This is a brief description of the example. It can contain multiple paragraphs,
    and it can use ``.rst`` syntax.

    .. note::

        You can use ``.rst`` directives in the docstring, such as ``.. note::``,
        ``.. warning::``, ``.. seealso::``, etc.

    After the definition of the title, you may also add directives such as
    ``.. replace::``, and they will be rendered properly. For example, you can add:

    .. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`


    """

Then, you can start writing the code for the example. The content of your python script
should be a sequence of code cells, each delimited by a line starting with ``# %%``.
Code cells may contain comments, which will be rendered as markdown in the final
documentation.

.. code-block:: python

    # %%
    # This is a comment that will be rendered as markdown in the final documentation.
    # You can use multiple lines for comments, and you can use ``.rst`` syntax in them.
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
-----------------------
Once you have written the code for the example (or while you are writing it), you can
run it to see how it looks in the final documentation. Depending on your setup,
you may need to install some dependencies to be able to run the example. Please
refer to the documentation of your own IDE for more information on how to run
interactive python scripts. For example, you can find the documentation for
VSCode `here <https://code.visualstudio.com/docs/python/jupyter-support-py>`_.

Once you are happy with your example, you can submit a pull request to the repository,
following the instructions in the :ref:`contributing guide <contributing>`.

Adding cross-references
-----------------------
An important aspect of writing examples is to add cross-references to the documentation
where relevant. This helps users to find more information about the concepts and
functions used in the example.

There are various ways to add cross-references in the docstring and comments of your example:

- You can add references to the objects in the skrub API using the ``:class:`~skrub.ClassName```
  or ``:func:`~skrub.function_name``` directives.
- If your example uses the same objects multiple times, you can define a replacement at the top
  of the docstring using the ``.. replace::`` directive, and then use the replacement
  instead of the full directive.
- You can also add references to other sections of the documentation using the
  ``:ref:`label``` directive, where ``label`` is the label of the section you want to reference.

For example, if your example uses the ``TableVectorizer`` class multiple times, you can
define a replacement at the top of the docstring. Then, you might want to add a
reference to the user guide section about the ``TableVectorizer`` class. This can be
done as follows:

.. code-block:: python

    """
    Title of the example
    ====================

    .. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

    This example demonstrates how to use the |TableVectorizer| class to vectorize a dataframe.

    See the :ref:`userguide_tablevectorizer` guide for more information about the |TableVectorizer| class.
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

Generating the new documentation
-------------------------------
Once you have written your example and added the necessary cross-references, you can
generate the new documentation to see how it looks. This can be done in two ways:

- You can run the commands ``make html`` or ``make html-noplot`` in the ``doc/``
  folder of the repository to generate the HTML documentation for the entire project.
- Alternatively, you can use ``pixi run -e doc build-doc`` or ``pixi run -e doc build-doc-quick``
  from the root folder to generate the documentation. The advantage of using ``pixi`` is that
  it automatically sets up a virtual environment with the necessary dependencies, so you
  don't need to worry about installing them manually.

The difference between ``make html`` and ``make html-noplot`` (or between
``pixi run -e doc build-doc`` and ``pixi run -e doc build-doc-quick``) is that the
``-noplot`` or ``-quick`` versions do not execute the code in the examples, which
makes the documentation generation much faster. This is useful if you only want
to check the formatting of your example, rather than the actual output of the code
(it is assumed that you have already run the code while writing it).
Since the CI is set up to run the full documentation generation in any case,
you can safely use the ``-noplot`` or ``-quick`` versions for local testing.

After generating the documentation, you can use a web browser to open the
``index.html`` file in the ``doc/_build/html/`` folder and see how it looks.
You should check that:

- Section titles are properly formatted.
- Any formatting in docstrings or comments is rendered as intended. For example,
  Sphinx uses spaces to delimit lists and code blocks, so if you have them in the
  example, make sure that they render correctly.
- Cross-references are working. You can check the logs of the Sphinx
  generation to see if there are any broken references.


Linking your work to examples already in the documentation
----------------------------------------------------------
After generating the documentation, you can start adding references to your example
in other relevant parts of the documentation. This helps users to find your example
when they are reading about related topics.

The reason this is done after generating the documentation is that you need to know
the name of your example as it appears in the documentation, which is defined
dynamically based on the name of the file. Assuming that the name of your example
is ``99_my_example.py``, you can find the generated files in ``doc/auto_examples``,
and the reference in the file  ``doc/auto_examples/99_my_example.rst``: the correct
reference looks like ``.. _sphx_glr_auto_examples_99_my_example.py``, and the correct
way of referencing it is:

.. code-block:: rst

    :ref:`sphx_glr_auto_examples_99_my_example.py`




Merging your example
-----------------------
Finally, if everything looks good, you can commit your changes and submit a pull request
to the repository. You can find more information on how to do this in the
:ref:`contributing guide <contributing>`.

The PR will be reviewed by the maintainers of the repository, who may suggest
changes or improvements. Once the PR is approved, it will be merged into the main
branch, and your example will be part of the official documentation. Thanks!
