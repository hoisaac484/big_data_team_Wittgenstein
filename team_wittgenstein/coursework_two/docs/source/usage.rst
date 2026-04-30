Usage
=====

There are two supported ways to run Coursework Two:

- Quick Start: load the committed seed into Postgres and launch the dashboard.
  This is best for graders or reviewers who want to inspect results quickly.
  It does not require Coursework One.
- Full Pipeline: run Coursework One and Coursework Two from scratch to
  reproduce the data and results end to end.

Quick Start
-----------

Loads the prebuilt seed snapshot and launches the dashboard without running the
CW1 or CW2 pipelines.

.. code-block:: bash

   # 1 - start platform Postgres
   cd /path/to/repo
   docker compose up -d

   # 2 - load seed into platform Postgres
   gunzip -c team_wittgenstein/coursework_two/docker/seed/seed.sql.gz \
     | docker exec -i postgres_db_cw psql -U postgres -d fift

   # 3 - install Python deps and launch the dashboard
   cd team_wittgenstein/coursework_two
   poetry install
   poetry run streamlit run dashboard/Home.py

The seed file at ``docker/seed/seed.sql.gz`` is a snapshot of the
``team_wittgenstein`` schema after a full pipeline run. It contains all 23
backtest scenarios plus the baseline factor scores, IC weights, and portfolio
positions.

Data freshness
--------------

The seed is a frozen snapshot. The dashboard's latest rebalance date is
whatever was in the database when that snapshot was generated; it does not
advance with current calendar time. To see fresher data, run the full pipeline
below so the project recomputes outputs from live upstream data.

Full Pipeline
-------------

Use this if you want to reproduce the data yourself instead of relying on the
seed.

.. code-block:: bash

   # 1 - start platform Postgres
   cd /path/to/repo
   docker compose up -d

   # 2 - run CW1 (data ingestion, ~30-60 min)
   cd team_wittgenstein/coursework_one
   poetry install
   poetry run python main.py

   # 3 - run CW2 (factor scoring + 23 scenarios, ~1.5-2.5 hours)
   cd ../coursework_two
   poetry install
   poetry run python main.py

   # 4 - launch the dashboard
   poetry run streamlit run dashboard/Home.py

Running stages individually
---------------------------

Run the strategy pipeline from the ``coursework_two`` directory:

.. code-block:: bash

   poetry run python main.py

Dashboard
---------

Launch the Streamlit dashboard locally:

.. code-block:: bash

   poetry run streamlit run dashboard/Home.py

Documentation
-------------

Build the Sphinx docs:

.. code-block:: bash

   cd docs
   make html

Then open ``build/index.html`` in a browser.

Regenerating the seed
---------------------

If you rerun the full pipeline and want to update the committed seed, dump the
schema and gzip it from the repository root:

.. code-block:: bash

   docker exec postgres_db_cw pg_dump -U postgres -d fift \
     --schema=team_wittgenstein --no-owner --no-acl \
     | gzip > team_wittgenstein/coursework_two/docker/seed/seed.sql.gz
