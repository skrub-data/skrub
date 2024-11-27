Contributing to skrub
=====================

First off, thanks for taking the time to contribute!

Below are some guidelines to help you get started.


Have a question?
----------------

If you have any questions, feel free to reach out:

- Join our community on `Discord <https://discord.gg/ABaPnm7fDC>`_ for general chat and Q&A.
- Alternatively, you can `start a discussion on GitHub <https://github.com/skrub-data/skrub/discussions>`_.

What to know before you begin
-----------------------------

To understand the purpose and goals behind skrub, please read our
`vision statement. <https://skrub-data.org/stable/vision.html>`_

If you're interested in the research behind skrub,
we encourage you to explore these papers:

- `Similarity Encoding for Learning with Dirty
  Categorical Variables <https://hal.inria.fr/hal-01806175>`_
- `Encoding High-Cardinality String Categorical
  Variables <https://hal.inria.fr/hal-02171256v4>`_.

How can I contribute?
---------------------

Reporting bugs
~~~~~~~~~~~~~~

Using the library is the best way to discover bugs and limitations. If you find one,
please:

1. **Check if an issue already exists**
   by searching the `GitHub issues <https://github.com/skrub-data/skrub/issues?q=is%3Aissue>`_

   - If **open**, leave a 👍 on the original message to signal that others are affected.
   - If closed, check for one of the following:
      - A **merged pull request** may indicate the bug is fixed. Update your
        skrub version or note if the fix is pending a release.
      - A **wontfix label** or reasoning may be provided if the issue was
        closed without a fix.
2. If the issue does not exist, `create a new one <https://github.com/skrub-data/skrub/issues/new>`_.

How to submit a bug report?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To help us resolve the issue quickly, please include:

- A **clear and descriptive title**.
- A **summary of the expected result**.
- Any **additional details** where the bug might occur or doesn't occur unexpectedly.
- A **code snippet** that reproduces the issue, if applicable.
- **Version information** for Python, skrub, and relevant dependencies (e.g., scikit-learn, numpy, pandas).

Suggesting enhancements
~~~~~~~~~~~~~~~~~~~~~~~

If you have an idea for improving skrub, whether it's a small fix
or a new feature, first:

- **Check if it has been proposed or implemented** by reviewing
  `open pull requests <https://github.com/skrub-data/skrub/pulls?q=is%3Apr>`_.
- If not, `submit a new issue <https://github.com/skrub-data/skrub/issues/new>`_
  with your proposal before writing any code.

How to submit an enhancement proposal?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When proposing an enhancement:

- **Use a clear and descriptive title**.
- **Explain the goal** of the enhancement.
- Provide a **detailed step-by-step description** of the proposed change.
- **Link to any relevant resources** that may support the enhancement.


If the enhancement proposal is validated
''''''''''''''''''''''''''''''''''''''''

Once your enhancement proposal is approved, let the maintainers know the following:

- **If you will write the code and submit a Pull Request (PR)**:
  Contributing the feature yourself is the quickest way to see it implemented.
  We're here to guide you through the process if needed! To get started,
  refer to the section :ref:`writing-your-first-pull-request`.
- **If you won't be writing the code**:
  A developer can then take over the implementation.
  However, please note that we cannot guarantee how long
  it will take for the feature to be added.


If the enhancement is refused
'''''''''''''''''''''''''''''

Although many ideas are great, not all will align with the objectives
of skrub.

If your enhancement is not accepted, consider implementing it
as a separate package that builds on top of skrub!

We would love to see your work, and in some cases, we might even
feature your package in the official repository.


.. _writing-your-first-pull-request:

Writing your first Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparing the ground
^^^^^^^^^^^^^^^^^^^^

Before writing any code, ensure you have created an issue
discussing the proposed changes with the maintainers.
See the relevant sections above on how to do this.

Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the steps in the :ref:`installation_instructions` > "From Source" section
to set up your environment, install the required development dependencies, and
run the tests.

When starting to work on a new issue, it's recommended to create a new branch:

.. code:: console

   git switch -c branch_name


.. _implementation guidelines:

Implementation Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^

When contributing, keep these project goals in mind:

- **Pure Python code**: Avoid using binary extensions, Cython, or other compiled languages.
- **Production-friendly code**:
    - Target the widest possible range of Python versions and dependencies.
    - Minimize the use of external dependencies.
    - Ensure backward compatibility as much as possible.
- **Performance over readability**:
  Optimized code may be less readable, so please include clear and detailed comments.
  Refer to this `best practice guide <https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/>`_.
- **Explicit variable/function names**: Use descriptive, verbose names for clarity.
- **Document public API components**:
    - Document all public functions, methods, variables, and class signatures.
    - The public API refers to all components available for import and use by library users. Anything that doesn't begin with an underscore is considered part of the public API.


Testing the code
~~~~~~~~~~~~~~~~

Tests for files in a given folder should be located in a sub-folder
named ``tests``: tests for Skrub objects are located in ``skrub/tests/``,
tests for the dataframe API are in ``skrub/_dataframe/tests/`` and so on.

Tests should check all functionalities of the code that you are going to
add. If needed, additional tests should be added to verify that other
objects behave correctly.

Consider an example: your contribution is for the
``AmazingTransformer``, whose code is in
``skrub/_amazing_transformer.py``. The ``AmazingTransformer`` is added
as one of the default transformers for ``TableVectorizer``.

As such, you should add a new file testing the functionality of
``AmazingTransformer`` in ``skrub/tests/test_amazing_transformer.py``,
and update the file ``skrub/tests/test_table_vectorizer.py`` so that it
takes into account the new transformer.

Additionally, you might have updated the internal dataframe API in
``skrub/_dataframe/_common.py`` with a new function,
``amazing_function``. In this case, you should also update
``skrub/_dataframe/tests/test_common.py`` to add a test for the
``amazing_function``.

Run each updated test file using ``pytest``:

.. code:: sh

   pytest -vsl skrub/tests/test_amazing_transformer.py \
   skrub/_dataframe/tests/test_common.py \
   skrub/_dataframe/tests/test_table_vectorizer.py

The ``-vsl`` flag provides more information when running the tests.

Once you are satisfied with your changes, you can run all the tests to make sure
that your change did not break code elsewhere:

.. code:: sh
    pytest -s skrub/tests

Finally, sync your changes with the remote repository and wait for CI to run.

Checking coverage on the local machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checking coverage is one of the operations that is performed after
submitting the code. As this operation may take a long time online, it
is possible to check whether the code coverage is high enough on your
local machine.

Run your tests with the ``--cov`` and ``--cov-report`` arguments:

.. code:: sh

   pytest -vsl skrub/tests/test_amazing_transformer.py --cov=skrub --cov-report=html

This will create the folder ``htmlcov``: by opening
``htmlcov/index.html`` it is possible to check what lines are covered in
each file.

Updating doctests
~~~~~~~~~~~~~~~~~

If you alter the default behavior of an object, then this might affect
the docstrings. Check for possible problems by running

.. code:: sh

   pytest skrub/path/to/file

Submitting your code
^^^^^^^^^^^^^^^^^^^^

Once you have pushed your commits to your remote repository, you can submit
a PR by clicking the "Compare & pull request" button on GitHub,
targeting the skrub repository.


Integration
^^^^^^^^^^^

Community consensus is key in the integration process. Expect a minimum
of 1 to 3 reviews depending on the size of the change before we consider
merging the PR.

Please be mindful that maintainers are volunteers, so review times may vary.

Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Github Actions**:
  Used for testing skrub across various platforms (Linux, macOS, Windows)
  and dependencies.
- **CircleCI**:
  Builds and verifies the project documentation.

If any of the following markers appears in the commit message, the following
actions are taken.

    ====================== ===================
    Commit Message Marker  Action Taken by CI
    ---------------------- -------------------
    [ci skip]              CI is skipped completely
    [skip ci]              CI is skipped completely
    [skip github]          CI is skipped completely
    [deps nightly]         CI is run with the nightly builds of dependencies
    [doc skip]             Docs are not built
    [doc quick]            Docs built, but excludes example gallery plots
    [doc build]            Docs built including example gallery plots (longer)
    ====================== ===================

Note that by default the documentation is built, but only the examples that are
directly modified by the pull request are executed.

- If the remote repository was changed, you might need to run
  ``pre-commit run --all-files`` to make sure that the formatting is
  correct.
- If a specific test environment fails, it is possible to run the tests
  in the environment that is failing by using pixi. For example if the
  env is ``ci-py309-min-optional-deps``, it is possible to replicate it
  using the following command:

.. code:: sh

   pixi run -e ci-py309-min-optional-deps  pytest skrub/tests/path/to/test



Building the documentation
--------------------------

..
  Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/doc/developers/contributing.rst

**Before submitting your pull request, ensure that your modifications haven't
introduced any new Sphinx warnings by building the documentation locally
and addressing any issues.**

First, make sure you have properly installed the development version of skrub.
You can follow the :ref:`installation_instructions` > "From source" section, if needed.

Building the documentation requires installing some additional packages:

.. code:: bash

    cd skrub
    pip install '.[doc]'

To build the documentation, you need to be in the ``doc`` folder:

.. code:: bash

    cd doc

To generate the full documentation, including the example gallery,
run the following command:

.. code:: bash

    make html

The documentation will be generated in the ``_build/html/`` directory
and are viewable in a web browser, for instance by opening the local
``_build/html/index.html`` file.

Running all the examples can take a while, so if you only want to generate
specific examples, you can use the following command with a regex pattern:

.. code:: bash

    make html EXAMPLES_PATTERN=your_regex_goes_here make html

This is especially helpful when you're only modifying or checking a few examples.
