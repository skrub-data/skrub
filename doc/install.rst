==========
Installing
==========

.. raw:: html

    <div class="container mt-4">

    <ul class="nav nav-pills nav-fill" id="installation" role="tablist">
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="pip-tab" data-bs-toggle="tab" data-bs-target="#pip-tab-pane" type="button" role="tab" aria-controls="pip" aria-selected="false">Using pip</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="conda-tab" data-bs-toggle="tab" data-bs-target="#conda-tab-pane" type="button" role="tab" aria-controls="conda" aria-selected="false">Using conda</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="mamba-tab" data-bs-toggle="tab" data-bs-target="#mamba-tab-pane" type="button" role="tab" aria-controls="mamba" aria-selected="false">Using mamba</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link active" id="source-tab" data-bs-toggle="tab" data-bs-target="#source-tab-pane" type="button" role="tab" aria-controls="source" aria-selected="true">From source</a>
        </li>
    </ul>

    <div class="tab-content">
        <div class="tab-pane fade" id="pip-tab-pane" role="tabpanel" aria-labelledby="pip-tab" tabindex="0">
            <hr />

.. warning::

   skrub has not yet been released. See the "From source" tab.

.. raw:: html

        </div>
        <div class="tab-pane fade" id="conda-tab-pane" role="tabpanel" aria-labelledby="conda-tab" tabindex="0">
            <hr />

.. warning::

   skrub has not yet been released. See the "From source" tab.

.. raw:: html

        </div>
        <div class="tab-pane fade" id="mamba-tab-pane" role="tabpanel" aria-labelledby="mamba-tab" tabindex="0">
            <hr />

.. warning::

   skrub has not yet been released. See the "From source" tab.

.. raw:: html

        </div>
        <div class="tab-pane fade show active" id="source-tab-pane" role="tabpanel" aria-labelledby="source-tab" tabindex="0">
            <hr />

Recommended usage, for users
----------------------------

To install from `the source <https://github.com/skrub-data/skrub>`_ using pip,
run the following command in a shell command line:

.. code:: console

    $ pip install git+https://github.com/skrub-data/skrub.git

Advanced usage, for contributors
--------------------------------

If you want to contribute to the project, you can install the development version
of skrub from the source code:

.. code:: console

    $ git clone https://github.com/skrub-data/skrub

Create a virtual environment, here for example, using `conda <https://docs.conda.io/en/latest/>`_:

.. code:: console

    $ conda create -n skrub python=3.10
    $ conda activate skrub

Then, install the local package in editable mode,
with the development requirements:

.. code:: console

    $ cd skrub
    $ pip install -e .[dev]

Next step, enable the pre-commit hooks:

.. code:: console

    $ pre-commit install

Finally, a few revisions better be ignored by ``git blame`` and IDE integrations.
These revisions are listed in ``.git-blame-ignore-revs``,
which can be set in your local repository with:

.. code:: console

    $ git config blame.ignoreRevsFile .git-blame-ignore-revs

You're ready to go! If not already done, please have a look at
the `contributing guidelines <https://skrub-data.org/stable/CONTRIBUTING.html>`_.

.. raw:: html

        </div>
    </div>

    </div>

