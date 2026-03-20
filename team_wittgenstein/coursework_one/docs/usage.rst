Usage
=====

Running the pipeline
--------------------

From the ``coursework_one`` directory::

    poetry run python main.py

The pipeline runs through five stages:

1. **Connect** — establishes connections to PostgreSQL, MongoDB, and
   MinIO, then creates the database schema if it does not exist.
2. **Load universe** — reads the company list from the
   ``company_static`` table, applies country filters, excludes
   known-bad symbols, and normalises tickers (e.g. ``BRK.B`` →
   ``BRK-B``).
3. **Fetch** — pulls daily prices (yfinance), quarterly financials
   (waterfall or SimFin), and risk-free rates (OECD + yfinance
   fallback).
4. **Validate** — checks data quality: minimum row counts, null
   percentages, required columns, and symbol coverage. In strict
   mode the pipeline halts on validation failures.
5. **Load** — writes validated data to PostgreSQL with duplicate
   prevention (merge-style checks before insert). Logs failures to
   MongoDB.

Pipeline output includes a validation report and a summary of rows
written per table.

Re-run safety
-------------

The pipeline is designed to be re-run safely:

- **CTL files** in MinIO track what has been fetched and when. Data
  older than ``cache_ttl_days`` is re-fetched; recent data is served
  from cache.
- **Duplicate checks** before every PostgreSQL insert prevent double-
  counting. Existing rows (by primary key) are skipped.
- **Stale symbol cleanup** removes data for symbols that are no
  longer in ``company_static``.

Command-line flags
------------------

``--no-schedule``
    Run the pipeline once and exit immediately. Without this flag the
    pipeline starts a recurring scheduler (prices monthly, fundamentals
    quarterly) that keeps running in the background.

``--task prices``
    Run only the prices and risk-free rates stage.

``--task fundamentals``
    Run only the fundamentals stage.

Dev mode
--------

Set ``dev.enabled: true`` in ``config/conf.yaml`` to limit the
pipeline to the first N symbols (default 3). This avoids API rate
limits during development and testing.

Running tests
-------------

::

    poetry run pytest tests/ -x -q

With coverage::

    poetry run pytest tests/ --cov=modules --cov-report=term-missing

Building documentation
----------------------

::

    cd docs
    poetry run sphinx-build -b html . _build/html

Open ``_build/html/index.html`` in a browser to view the docs.
