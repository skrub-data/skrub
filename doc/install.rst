.. _installation_instructions:

=======
Install
=======

.. raw:: html

    <div class="container mt-4">

    <ul class="nav nav-pills nav-fill" id="installation" role="tablist">
        <li class="nav-item" role="presentation">
            <a class="nav-link active" id="pip-tab" data-bs-toggle="tab" data-bs-target="#pip-tab-pane" type="button" role="tab" aria-controls="pip" aria-selected="true">Using pip</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="conda-tab" data-bs-toggle="tab" data-bs-target="#conda-tab-pane" type="button" role="tab" aria-controls="conda" aria-selected="false">Using conda</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="mamba-tab" data-bs-toggle="tab" data-bs-target="#mamba-tab-pane" type="button" role="tab" aria-controls="mamba" aria-selected="false">Using mamba</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="source-tab" data-bs-toggle="tab" data-bs-target="#source-tab-pane" type="button" role="tab" aria-controls="source" aria-selected="false">From source</a>
        </li>
    </ul>

    <div class="tab-content">
        <div class="tab-pane fade show active" id="pip-tab-pane" role="tabpanel" aria-labelledby="pip-tab" tabindex="0">
            <hr />

.. code:: console

    pip install skrub -U

.. raw:: html

        </div>
        <div class="tab-pane fade" id="conda-tab-pane" role="tabpanel" aria-labelledby="conda-tab" tabindex="0">
            <hr />

.. code:: console

    conda install -c conda-forge skrub

.. raw:: html

        </div>
        <div class="tab-pane fade" id="mamba-tab-pane" role="tabpanel" aria-labelledby="mamba-tab" tabindex="0">
            <hr />

.. code:: console

    mamba install -c conda-forge skrub

.. raw:: html

        </div>
        <div class="tab-pane fade" id="source-tab-pane" role="tabpanel" aria-labelledby="source-tab" tabindex="0">
            <hr />

Advanced Usage for Contributors
-------------------------------

1. Fork the project
'''''''''''''''''''

To contribute to the project, you first need to
`fork skrub on GitHub <https://github.com/skrub-data/skrub/fork>`_.

That will enable you to push your commits to a branch *on your fork*.

2. Clone your fork
''''''''''''''''''

Clone your forked repo to your local machine:

.. code:: console

    git clone https://github.com/<YOUR_USERNAME>/skrub
    cd skrub

Next, add the *upstream* remote (i.e. the official skrub repository). This allows you
to pull the latest changes from the main repository:

.. code:: console

    git remote add upstream https://github.com/skrub-data/skrub.git

Verify that both the origin (your fork) and upstream (official repo)
are correctly set up:

.. code:: console

    git remote -v


3. Setup your environment
'''''''''''''''''''''''''

Now, setup a development environment. For example, you can use
`conda <https://docs.conda.io/en/latest/>`_  to create a virtual environment:

.. code:: console

    conda create -n skrub python=3.10 # or any later python version
    conda activate skrub

Install the local package in editable mode with development dependencies:

.. code:: console

    pip install -e ".[dev, lint, test]"

Enable pre-commit hooks to ensure code style consistency:

.. code:: console

    pre-commit install


Optionally, configure Git to ignore certain revisions in git blame and
IDE integrations. These revisions are listed in .git-blame-ignore-revs:

.. code:: console

    git config blame.ignoreRevsFile .git-blame-ignore-revs

4. Run the tests
''''''''''''''''

To ensure your environment is correctly set up, run the test suite:

.. code:: console

    pytest -s skrub/tests

Testing should take about 5 minutes.
If no errors or failures are found, your environment is ready for development!

Now that you're set up, review our :ref:`implementation guidelines<implementation guidelines>`
and start coding!

.. raw:: html

        </div>
    </div>
    </div>
