Installation
============

Prerequisites
-------------

- **Python 3.10+**
- **Poetry** for dependency management
- **Docker** (recommended) for running PostgreSQL, MongoDB, and MinIO locally

Install dependencies
--------------------

Clone the repository and install with Poetry::

    git clone <repo-url>
    cd big_data_team_Wittgenstein/team_wittgenstein/coursework_one
    poetry install

This installs all runtime and development dependencies, including
pytest, flake8, black, and Sphinx.

Database setup
--------------

The pipeline requires three backing services:

**PostgreSQL** (port 5439 by default)
    Stores structured data: prices, financials, risk-free rates,
    factor metrics, and factor scores. The schema is created
    automatically on first run via ``sql/create_schema.sql``.

**MongoDB** (port 27019 by default)
    Stores raw API responses as an audit trail. Each fetch is logged
    as a timestamped document.

**MinIO** (port 9000 by default)
    Object storage used for caching fetched data as parquet files
    with companion CTL (control) files for idempotency.

Start all three services with Docker Compose (if provided) or
configure connection details in ``config/conf.yaml``.

Configuration
-------------

All settings live in ``config/conf.yaml``:

.. code-block:: yaml

    postgres:
      host: "localhost"
      port: 5439
      database: "fift"
      user: "postgres"
      password: "postgres"

    mongo:
      host: "localhost"
      port: 27019

    minio:
      host: "localhost:9000"
      access_key: "ift_bigdata"
      secret_key: "minio_password"
      secure: false

    data:
      price_period: "5y"
      fundamentals_period: "5y"
      fundamentals_source: "waterfall"
      cache_ttl_days: 7

    country_filter: "US"

    validation:
      min_price_rows: 200
      min_years: 4
      max_null_pct: 0.5
      strict: true

    dev:
      enabled: false
      max_symbols: 3

Key settings:

- **fundamentals_source**: ``waterfall`` (EDGAR → SimFin → yfinance →
  forward-fill) or ``simfin`` (SimFin only).
- **cache_ttl_days**: Re-fetch cached data older than this many days.
  Remove to disable TTL.
- **dev.enabled**: Limits the symbol list to ``max_symbols`` to avoid
  API rate limits during development.
- **country_filter**: Restrict to a single country (e.g. ``US``).
