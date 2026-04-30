Installation
============

Prerequisites
-------------

- Python 3.10
- Poetry
- Docker and Docker Compose for the local platform services

Environment setup
-----------------

From the ``coursework_two`` directory:

.. code-block:: bash

   poetry install

If you want to run the full pipeline, make sure Coursework One has already
populated the shared ``team_wittgenstein`` schema inputs.

Documentation dependencies
--------------------------

The Sphinx documentation uses the dev dependencies declared in
``pyproject.toml``. If the lock file is stale after dependency changes, refresh
it before building docs:

.. code-block:: bash

   poetry lock
   poetry install
