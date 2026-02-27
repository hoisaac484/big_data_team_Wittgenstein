# Data Catalog

## 1. Document Purpose
This document supports “Develop and maintain data catalogs” by recording all key data assets in Coursework One, including storage locations, refresh frequency, owners, and downstream usage, to support team collaboration and subsequent factor research and portfolio construction in Coursework Two.

## 2. Catalog Scope
- PostgreSQL (structured master data + factor outputs)
- MinIO (raw market data lake)
- External data sources (Yahoo Finance, Alpha Vantage, OECD)

## 3. Data Asset Inventory

| Asset ID | Data Asset Name | Type | Storage Location | Primary Key / Granularity | Refresh Frequency | Upstream Source | Downstream Use | Owner |
|---|---|---|---|---|---|---|---|---|
| CAT-001 | Company static info `company_static` | Table | PostgreSQL: systematic_equity.company_static | company_id (one row per company) | On demand (company add/remove) | Course-provided database | Defines investable universe | / |
| CAT-002 | Raw daily prices `raw market csv` | Object | MinIO bucket: systematic-equity/raw/run_date=YYYY-MM-DD/{ticker}.csv | ticker + date (one row per day) | Daily/monthly batch | Yahoo Finance | Input for factor calculation (Momentum/LowVol) | / |
| CAT-003 | Fundamentals `fundamentals` | Object/Table (implementation optional) | MinIO or PostgreSQL staging | ticker/company_id + fiscal_period | Quarterly/annual | Alpha Vantage/SEC EDGAR | Input for Value/Quality factors | / |
| CAT-004 | Short-term risk-free rate `risk_free_rate` | Object/Table (implementation optional) | MinIO or PostgreSQL staging | country + month | Monthly | OECD API | Excess momentum return calculation | / |
| CAT-005 | Factor outputs `factor_values` | Table | PostgreSQL: systematic_equity.factor_values | (company_id, factor_date, factor_name) | Consistent with pipeline (daily/monthly/quarterly) | Factor computation modules | Ranking and stock selection in Coursework Two | / |

## 4. Asset Notes (By Workflow Stage)

### 4.1 Company Universe (Entry Asset)
- Asset: `company_static`
- Purpose: defines processing entities for the full pipeline (ticker/company_id)
- Risks: missing ticker values, identifier changes after delisting
- Controls: null checks during read; keep `company_id` as stable primary key

### 4.2 Raw Market Data (Data Lake)
- Asset: `systematic-equity/raw/run_date=YYYY-MM-DD/{ticker}.csv`
- Purpose: preserve traceable raw market data for recomputation and auditing
- Minimum fields: `date`, `close` (recommended extensions: `adj_close`, `volume`, `ingest_ts`)
- Controls: file path convention, overwrite policy (append or partition by run_date)

### 4.3 Factor Outputs (Serve Downstream)
- Asset: `factor_values`
- Purpose: provide standardized factor values for backtesting/live strategy modules
- Core fields: `company_id`, `factor_date`, `factor_name`, `factor_value`
- Controls: primary-key deduplication via `ON CONFLICT DO NOTHING` or upsert

## 5. Update and Maintenance Mechanism
- Maintenance cycle: update this catalog within 24 hours after any schema change, data source change, or scheduling-frequency change.
- Versioning rule: `vYYYY.MM.DD`.
- Change log template:
  - Change item (add/delete/modify)
  - Impacted asset ID
  - Impact scope (upstream/downstream)
  - Rollback plan

## 6. Data Quality and SLA (Minimum Requirements)
- Completeness: company list coverage >= 99%
- Freshness: price data delay <= 1 scheduling cycle
- Uniqueness: primary-key conflict rate in `factor_values` = 0
- Traceability: each run must map to source + run_date + pipeline version

## 7. Alignment with Coursework One
- Meets storage objective of retrieval by company and by year
- Supports extensible configurable run frequencies (daily/weekly/monthly/quarterly)
