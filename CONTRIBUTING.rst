Contributing to dirty_cat
=========================

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to
`dirty_cat <https://github.com/dirty-cat/dirty_cat>`__.

|

.. contents::
   :local:
|

I don’t want to read the whole thing I just have a question
------------------------------------------------------------

We use GitHub Discussions for general chat and Q&As. `Check it
out! <https://github.com/dirty-cat/dirty_cat/discussions>`__

What should I know before I get started?
----------------------------------------

If you want to truly understand what are the incentives behind
dirty_cat, and if scientific literature doesn’t scare you, we greatly
encourage you to read the two papers `Similarity encoding for learning
with dirty categorical variables <https://hal.inria.fr/hal-01806175>`__
and `Encoding high-cardinality string categorical
variables <https://hal.inria.fr/hal-02171256v4>`__.

How can I contribute?
---------------------

Reporting bugs
~~~~~~~~~~~~~~

Even if we unit-test our code, using the library is the best way to
discover new bugs and limitations.

If you stumble upon one, please `check if a similar or identical issue already
exists <https://github.com/dirty-cat/dirty_cat/issues?q=is%3Aissue>`__

- If yes: 

  - **The issue is still open**: leave the emote :+1: on the original message, 
    which will let us know there are several users affected by this issue 
  - **The issue has been closed**: 

    - **It has been closed by a merged pull request** (1) update your dirty_cat version, 
      or (2) the fix has not been released in a version yet
    - **Otherwise**, there might be a ``wontfix`` label, and / or a reason at the bottom of the conversation 
- If not, `file a new issue <https://github.com/dirty-cat/dirty_cat/issues/new>`__ (see following section)

How do I submit a (good) bug report?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve your issue as soon as possible, explain the problem and include
additional details to help maintainers easily reproduce the problem:

-  **Use a clear and descriptive title** which identifies the problem
-  **Describe the result you expected**
-  **Add additional details to your description problem** such as
   situations where the bug should have appeared but didn’t
-  **Include a snippet of code that reproduces the error**, as it allows
   maintainers to reproduce it in a matter of seconds!
-  **Specify versions** of Python, dirty_cat, and other dependencies
   which might be linked to the issue (e.g., scikit-learn, numpy,
   pandas, etc.). You can get these versions with ``pip3 freeze``
   (``pip freeze`` for Windows)

Of course, some of these bullet points might not apply depending on the
kind of error you’re submitting.

Suggesting enhancements
~~~~~~~~~~~~~~~~~~~~~~~

This section will guide you through submitting a new enhancement for
dirty_cat, whether it is a small fix or a new feature.

First, you should `check if the feature has not already been proposed or
implemented <https://github.com/dirty-cat/dirty_cat/pulls?q=is%3Apr>`__.

If not, the next thing you should do, before writing any code, is to
`submit a new
issue <https://github.com/dirty-cat/dirty_cat/issues/new>`__ proposing
the change.

How do I submit a (good) enhancement proposal?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Use a clear and descriptive title**
-  **Provide a quick explanation of the goal of this enhancement**
-  **Provide a step-by-step description of the suggested enhancement**
   with as many details as possible
-  **If it exists elsewhere, link resources**

Of course, some of these bullet points might not apply depending on the
kind of enhancement you’re submitting.

If the enhancement is validated
'''''''''''''''''''''''''''''''

Let maintainers know whether : 

- **You will write the code and submit a PR**. 
  Writing the feature yourself is the fastest way to getting it
  implemented in the library, and we’ll help in that process if guidance
  is needed! To go further, refer to the section *Writing your first Pull Request*.
- **You won’t be able to write the code**, in which case a
  developer interested in the feature can start working on it. Note
  however that maintainers are **volunteers**, and therefore cannot
  guarantee how much time it will take to implement the change.

If the enhancement is refused
'''''''''''''''''''''''''''''

There are specific incentives behind dirty_cat. While most enhancement
ideas are good, they don’t always fit in the context of the library.

If you’d like to implement your idea regardless, we’d be very glad if
you create a new package that builds on top of dirty_cat! In some cases,
we might even feature it on the official repository!

Writing your first Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparing the ground
^^^^^^^^^^^^^^^^^^^^

If not already done, you’ll want to create an issue first, and discuss
the changes with the project’s maintainers.

Please refer to the previous section *How do I submit a (good)
enhancement proposal?* for more information.

Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a dedicated Python environment is highly recommended!

Different solutions are available, the most common being
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`__
and `pyenv <https://github.com/pyenv/pyenv>`__.

So, first step: create your environment.

For this example, we’ll use conda:

.. code:: commandline

   conda create python=3.10 --name dirty_cat
   conda activate dirty_cat

Secondly, clone the repository (you’ll need to have ``git`` installed -
it is already on most linux distributions).

.. code:: commandline

   git clone https://github.com/dirty-cat/dirty_cat

Next, install the project dependencies. Currently, they are listed in
``requirements.txt``.

.. code:: commandline

   pip install -r requirements.txt

Code-formatting and linting is automatically done via
```pre-commit`` <https://github.com/pre-commit/pre-commit>`__. You
install this setup using:

.. code:: commandline

   pip install pre-commit
   pre-commit install

A few revisions (formatting the whole code-base for instance) better be
ignored by ``git blame`` and IDE integrations. The revisions to be
ignored are listed in ``.git-blame-ignore-revs``, which can be set in
your local repository with:

.. code:: commandline

   git config blame.ignoreRevsFile .git-blame-ignore-revs

Implementation
^^^^^^^^^^^^^^

While writing your implementation, there are a few specific project
goals to keep in mind:

- Pure Python code - no binary extensions, Cython, etc 
- Make production-friendly code

  - Try to target the broadest range of versions (Python and dependencies)
  - Use the least amount of dependencies
  - Make code as backward compatible as possible
- Prefer performance to readability

  - Optimized code might be hard to read, so 
    `please comment it <https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/>`__
- Use explicit, borderline verbose variables / function names
- Public functions / methods / variables / class signatures should be documented
  and type-hinted

  - The public API describes the components users of the
    library will import and use. It’s everything that can be imported and
    does not start with an underscore.

Submitting your code
^^^^^^^^^^^^^^^^^^^^

First, you’ll want to fork dirty_cat on Github.

That will enable you to push your commits to a branch *on your fork*.

Next, you can use the Github “Compare & pull request” button to submit
the PR.

Integration
^^^^^^^^^^^

Community consensus is key in the integration process. Expect a minimum
of 1 to 3 reviews depending on the size of the change before we consider
merging the PR.

Once again, remember that maintainers are **volunteers** and therefore
cannot guarantee how much time it will take to review the changes.
