Architecture
============

Overview
--------

The pipeline follows a three-stage architecture::

    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
    │ DataFetcher  │────▶│ DataValidator  │────▶│  DataWriter  │
    │  (input)     │     │  (processing)  │     │  (output)    │
    └──────────────┘     └───────────────┘     └──────────────┘
          │                                           │
          ▼                                           ▼
       MinIO                                    PostgreSQL
     (parquet                                  (structured
      cache)                                     tables)
                                                     │
                                                     ▼
                                                  MongoDB
                                               (audit trail)

**DataFetcher** pulls data from external APIs, caches it in MinIO,
and returns cleaned DataFrames.

**DataValidator** checks data quality (completeness, coverage,
null ratios) and returns pass/fail results.

**DataWriter** loads validated data into PostgreSQL with duplicate
prevention and logs raw responses to MongoDB.

DataFetcher — mixin composition
-------------------------------

``DataFetcher`` is assembled from eight specialised mixins, each
handling a single responsibility:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mixin
     - Responsibility
   * - ``CacheMixin``
     - MinIO parquet caching with CTL control files
   * - ``UtilsMixin``
     - Schema normalisation, period parsing, failure classification
   * - ``PriceMixin``
     - Daily price fetching via yfinance batch download
   * - ``EdgarMixin``
     - SEC EDGAR API: CIK resolution, company facts, concept extraction
   * - ``SimFinMixin``
     - SimFin API: income statements, balance sheets, weighted shares
   * - ``YFinanceMixin``
     - yfinance quarterly financials (lowest-priority fallback)
   * - ``FundamentalsMixin``
     - Waterfall orchestration, multi-source merge, forward-fill
   * - ``RatesMixin``
     - Risk-free rates from OECD API with yfinance fallback

Waterfall merge strategy
------------------------

Quarterly fundamentals are fetched using a priority waterfall::

    1. SEC EDGAR  (free, no API key, US-listed)
           │
           ▼
    2. SimFin     (free tier, global coverage)
           │
           ▼
    3. yfinance   (fallback, limited fields)
           │
           ▼
    4. Forward-fill remaining nulls from previous quarters

For each ``(symbol, fiscal_year, fiscal_quarter)`` key, null fields
in the higher-priority source are filled from the next source down.
After merging, remaining nulls are forward-filled (limit 2 quarters)
from the most recent non-null value for the same symbol.

Concurrency and rate limiting
-----------------------------

**Fundamentals fetching** uses a ``ThreadPoolExecutor`` with 4 workers
to fetch multiple symbols concurrently. Each worker runs the full
waterfall (EDGAR → SimFin → yfinance) for one symbol at a time.

**Price retry logic**: after the yfinance batch download, any symbols
that were silently dropped are retried individually with separate
``yf.download()`` calls. This prevents mismatches between price and
financial data coverage.

**Rate limiting** is enforced per API:

- **Proactive throttling**: a minimum interval (0.55s for SimFin,
  0.5s for EDGAR) between consecutive requests, enforced via
  ``threading.Lock``.
- **Reactive retry**: on HTTP 429 (rate limited), the code sleeps
  for the ``Retry-After`` header value (or 2 seconds) then retries.

CTL caching pattern
-------------------

Each fetched dataset is cached in MinIO as a parquet file with a
companion JSON control (CTL) file::

    wittgenstein-cache/
    ├── prices/
    │   ├── AAPL.parquet        (cached price data)
    │   └── AAPL.ctl            (metadata: source, timestamp, row count)
    ├── fundamentals/
    │   ├── AAPL.parquet
    │   └── AAPL.ctl
    └── risk_free_rates/
        ├── all.parquet
        └── all.ctl

The CTL file records when data was fetched and from which source.
On subsequent runs, ``_is_cached()`` checks the CTL timestamp against
``cache_ttl_days`` to decide whether to re-fetch or serve from cache.

Database schema
---------------

**PostgreSQL** (schema: ``team_wittgenstein``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Table
     - Primary key
   * - ``price_data``
     - ``(symbol, trade_date)``
   * - ``financial_data``
     - ``(symbol, fiscal_year, fiscal_quarter)``
   * - ``risk_free_rates``
     - ``(country, rate_date)``
   * - ``factor_metrics``
     - ``(symbol, calc_date)``
   * - ``factor_scores``
     - ``(symbol, score_date)``
   * - ``portfolio_positions``
     - (downstream)

**MongoDB** (database: ``wittgenstein``):

Collections prefixed with ``raw_`` store timestamped audit documents
for each API fetch (e.g. ``raw_prices``, ``raw_fundamentals``).
Failure classifications are stored in ``raw_price_failures`` and
``raw_fundamentals_failures``.
