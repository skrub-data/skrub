Table reports: notes for skrub maintainers
==========================================

This module generates HTML reports that summarize a dataframe's content.

Generating the report
---------------------

Information about the dataframe is collected in a dictionary by
``_summarize.summarize_dataframe``. This dictionary is then used to render a
`jinja <https://jinja.palletsprojects.com/en/3.0.x/>`_ template, filling it with
information about this particular dataframe. This is done by ``_html.to_html``.

The templates are stored in ``_data/templates/``. The top-level templates are
``standalone-report.html`` for a full html page (``TableReport.html()``), and
``inline-report.html`` for an html fragment that can be inserted into another
page or a notebook (``TableReport.html_snippet()``). Those, in turn, use the
jinja ``include`` directive to bring in inner parts of the report, including the
CSS and javascript files.

.. note::

   the files in ``_data/templates/pure-3.0.0/`` are a subset of the
   `pure.css <https://purecss.io/>`_ project. Only the ``base`` and ``tables``
   modules are included. They respectively provide some baseline default styling
   for better cross-browser consistency and some basic styling of tables such as
   thinner borders and gray stripes. Do not edit those files manually, but
   rather override any styles in the skrub css files as necessary.

To open reports in a browser (``TableReport.open()``),
``_serve.open_in_browser`` is used. It starts a local server which sends the
report to the browser then immediately shuts down.

Reports in the wild
-------------------

The reports are meant to be embedded in web pages we do not control, such as
jupyter notebooks, sphinx gallery documentation pages, etc.
This means that CSS and javascript in the surrounding page could easily break
the display or functionality of the report.

To avoid this, the report is placed in a
`shadow DOM <https://developer.mozilla.org/en-US/docs/Web/API/Web_components#shadow_dom>`_,
which insulates it from the surrounding css and javascript. The report is
implemented by a ``skrub-table-report`` custom element which loads the contents
of an html ``<template>`` into its shadow root to display it.

When the page first loads, it only shows a message saying that javascript needs
to be enabled for the report to work correctly (if in a jupyter notebook, it
needs to be trusted). A script then hides that message and shows the report. If
javascript is not enabled and the script cannot run, the user sees the message.

Testing the javascript code
---------------------------

The Python code generating the reports is tested with ``pytest`` like the rest of ``skrub``.
The functionality of the reports themselves, which happens in the web browser,
is tested with `cypress <https://www.cypress.io/>`_. Those tests are in ``js_tests/``.

To execute them, you need to install nodejs (and the javascript package manager,
npm). For example on Ubuntu::

  apt install nodejs

Then, in the ``js_tests/`` directory, use ``npm`` to install the required
dependencies (actually just ``cypress``) in the local ``js_tests/node_modules``
folder::

  cd js_tests/
  npm install

It is then possible to execute the tests. It can be done interactively while displaying
the tested page in a browser window with::

  npx cypress open

Selecting "E2E" testing in the window that opens, then select any browser. A
list of tests ("specs") appears, such as ``column-filters.cy.js``. Those
correspond to files in ``js_tests/cypress/e2e/``. Click on any of them to run
it.

Testing can also be done non-interactively with::

  npx cypress run

All the tests are executed and a report is printed on stdout. If a test fails, a
screenshot of the browser window at the point where it failed is saved in
``cypress/screenshots/``. See the Cypress documentation for details.

To add or modify the tests, edit the files in ``js_tests/cypres/e2e/`` or add
new ones.

In the CI, running the javascript tests is done by the ``test-javascript.yml``
workflow. It which relies on the cypress
`github action <https://github.com/cypress-io/github-action>`_
to install node and cypress and to run the tests .


Distributing non-Python files
-----------------------------

The html templates, javascript and CSS files are not python files so some special
steps must be taken in the build configuration to make sure they are included in
the source distribution and the wheel. At the moment (see ``pyproject.toml``)
skrub's ``build-system`` requires the ``setuptools_scm`` plugin, which packages
all files that are tracked by git. If the build backend is replaced by a
different one, refer to the new build backend's documentation on handling
"package data" (non-python files).
