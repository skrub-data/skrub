Contributing to skrub
=========================

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to
`skrub <https://github.com/skrub-data/skrub>`__.

|
.. contents::
   :local:

|

I just have a question
----------------------

We use GitHub Discussions for general chat and Q&As. `Check it
out! <https://github.com/skrub-data/skrub/discussions>`__

What should I know before I get started?
----------------------------------------

If you want to understand in more depth the incentives behind skrub,
read our `vision statement! <https://skrub-data.org/stable/vision.html>`__ 
If scientific literature doesn’t scare you, we greatly
encourage you to read the two following papers:
   - `Similarity encoding for learning
   with dirty categorical variables <https://hal.inria.fr/hal-01806175>`__
   - `Encoding high-cardinality string categorical
   variables <https://hal.inria.fr/hal-02171256v4>`__.

How can I contribute?
---------------------

Reporting bugs
~~~~~~~~~~~~~~

Even if we unit-test our code, using the library is the best way to
discover new bugs and limitations.

If you find one, please `check if a similar or identical issue already
exists. <https://github.com/skrub-data/skrub/issues?q=is%3Aissue>`__

- If yes:

  - **The issue is still open**: leave the emote :+1: on the original message,
    which will let us know there are several users affected by this issue.
  - **The issue has been closed**:

    - **It has been closed by a merged pull request** (1) update your skrub version,
      or (2) the fix has not been released in a version yet.
    - **Otherwise**, there might be a ``wontfix`` label, and / or a reason at the bottom of the conversation.
- If not, `file a new issue <https://github.com/skrub-data/skrub/issues/new>`__.

How do I submit a bug report?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve your issue, first explain the problem and include
additional details to help maintainers easily reproduce the problem:

-  **Use a clear and descriptive title** which identifies the problem.
-  **Describe the result you expected**.
-  **Add additional details to your description problem** such as
   situations where the bug should have appeared but didn’t.
-  **Include a snippet of code that reproduces the error**, as it allows
   maintainers to reproduce it in a matter of seconds!
-  **Specify versions** of Python, skrub, and other dependencies
   which might be linked to the issue (e.g., scikit-learn, numpy,
   pandas, etc.).

Some of these bullet points might not apply depending on the
kind of error you’re submitting.

Suggesting enhancements
~~~~~~~~~~~~~~~~~~~~~~~

This section will guide you through submitting a new enhancement for
skrub, whether it is a small fix or a new feature.

First, you should `check if the feature has not already been proposed or
implemented <https://github.com/skrub-data/skrub/pulls?q=is%3Apr>`__.

If not, before writing any code, `submit a new
issue <https://github.com/skrub-data/skrub/issues/new>`__ proposing
the change.

How do I submit an enhancement proposal?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Use a clear and descriptive title**.
-  **Provide a quick explanation of the goal of this enhancement**.
-  **Provide a step-by-step description of the suggested enhancement**
   with as many details as possible.
-  **If it exists elsewhere, link resources**.

Of course, some of these bullet points might not apply depending on the
kind of enhancement you’re submitting.

If the enhancement is validated
'''''''''''''''''''''''''''''''

Let maintainers know whether :

- **You will write the code and submit a pull request (PR)**.
  Writing the feature yourself is the fastest way to getting it
  implemented in the library, and we’ll help in that process if guidance
  is needed! To go further, refer to the section
  `*Writing your first Pull Request*<https://skrub-data.org/stable/CONTRIBUTING.html#writing-your-first-pull-request>`__.
- **You won’t be able to write the code**, in which case a
  developer can start working on it. Note however that maintainers
  are **volunteers**, and therefore cannot guarantee how much time
  it will take to implement the change.

If the enhancement is refused
'''''''''''''''''''''''''''''

There are specific incentives behind skrub. While most enhancement
ideas are good, they don’t always fit in the context of the library.

If you’d like to implement your idea regardless, we’d be very glad if
you create a new package that builds on top of skrub! In some cases,
we might even feature it on the official repository!

Writing your first Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparing the ground
^^^^^^^^^^^^^^^^^^^^

If not already done, first create an issue, and discuss
the changes with the project’s maintainers.

See in the sections above for the right way to do this.

Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

For this process, please refer to the `installation process <https://skrub-data.org/stable/install.html#advanced-usage-for-contributors>`__,
section *Advanced usage, for contributors*.

Implementation
^^^^^^^^^^^^^^

There are a few specific project goals to keep in mind:

- Pure Python code - no binary extensions, Cython, etc.
- Make production-friendly code.

  - Try to target the broadest range of versions (Python and dependencies).
  - Use the least amount of dependencies.
  - Make code as backward compatible as possible.
- Prefer performance to readability.

  - Optimized code might be hard to read, so
    `please comment it <https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/>`__
- Use explicit, borderline verbose variables / function names
- Public functions / methods / variables / class signatures should be documented
  and type-hinted.

  - The public API describes the components users of the
    library will import and use. It’s everything that can be imported and
    does not start with an underscore.

Submitting your code
^^^^^^^^^^^^^^^^^^^^

First, you’ll want to `fork skrub on Github <https://github.com/skrub-data/skrub/fork>`__.

That will enable you to push your commits to a branch *on your fork*.
It is advised to create a new branch every time you work on a new issue,
to avoid confusion.
Use the following command to create a branch:

.. code:: console

   git checkout -b branch_name

Next, you can use the Github “Compare & pull request” button to submit
your branch code as a PR.

Integration
^^^^^^^^^^^

Community consensus is key in the integration process. Expect a minimum
of 1 to 3 reviews depending on the size of the change before we consider
merging the PR.

Once again, remember that maintainers are **volunteers** and therefore
cannot guarantee how much time it will take to review the changes.
