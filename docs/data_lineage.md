# Data Lineage

## 1. Document Purpose
This document supports “Design and document data lineages” by describing the end-to-end lineage in Coursework One from data sources to factor persistence, including processing steps, inputs/outputs, quality controls, and failure handling.

## 2. End-to-End Workflow (Plain Language)
1. Read company universe from PostgreSQL (`company_static`)
2. Fetch market data by ticker (mock data can be used initially)
3. Write raw data to MinIO (Data Lake)
4. Compute factors (start with momentum)
5. Write factor outputs to PostgreSQL (`factor_values`)

## 3. Component-Level Lineage

| Step | Module/File | Input | Processing | Output | Storage |
|---|---|---|---|---|---|
| L1 | `modules/db/company_loader.py` | `systematic_equity.company_static` | Query company ID and ticker | `[(company_id, ticker)]` | Memory |
| L2 | `modules/input/market_loader.py` | ticker, run date | Fetch/mock price series | `date, close` DataFrame | Memory |
| L3 | `modules/storage/minio_client.py` | Raw price DataFrame | Serialize to CSV and upload | `raw/run_date=YYYY-MM-DD/{ticker}.csv` | MinIO |
| L4 | `modules/processing/factor_engine.py` | Price DataFrame | Compute factors (e.g., 6M+12M momentum signals and composite score) | `date, momentum_score` | Memory |
| L5 | `modules/output/factor_writer.py` | `company_id` + factor DataFrame | Batch insert (deduplicated) | `factor_values` records | PostgreSQL |
| L6 | `Main.py` | All modules | Orchestration and loop control | One complete pipeline run | Logs / database |

## 4. Field-Level Lineage (Momentum Example)
- Upstream fields:
  - `company_static.company_id`
  - `company_static.ticker`
  - `raw/run_date=YYYY-MM-DD/{ticker}.csv.date`
  - `raw/run_date=YYYY-MM-DD/{ticker}.csv.close`
- Transformation logic:
  - `MOM_{i,6,t} = (P_{i,t-1}/P_{i,t-7} - 1) - RF^{(1m)}_{c,t}`
  - `MOM_{i,12,t} = (P_{i,t-1}/P_{i,t-13} - 1) - RF^{(1m)}_{c,t}`
  - `momentum_score = 0.5*z_{i,6,t} + 0.5*z_{i,12,t}`
- Downstream field mapping:
  - `factor_values.company_id <- company_static.company_id`
  - `factor_values.factor_date <- raw.date`
  - `factor_values.factor_name <- 'momentum'`
  - `factor_values.factor_value <- momentum_score`

## 5. Temporal Lineage and Scheduling
- Configurable run frequency: daily/weekly/monthly/quarterly
- Recommended:
  - Price ingestion: daily
  - Momentum/LowVol: monthly
  - Quality: quarterly
  - Value: annually (aligned with financial statement disclosures)

## 6. Data Quality Control Points (By Node)
- L1 (company load): ticker null-rate check
- L2 (data fetch): date continuity and duplicate-date checks
- L3 (data lake write): file size > 0 and object path convention checks
- L4 (computation): output null when window length is insufficient, and log reason
- L5 (database write): primary-key conflict monitoring and written-row reconciliation

## 7. Traceability and Audit Design
- Generate `run_id` for each run (recommended format: `YYYYMMDD_HHMMSS`)
- Recommended logging fields:
  - `run_id`, `pipeline_name`, `source`, `record_count_in/out`, `error_count`
- Recommended MinIO partitioned path:
  - `raw/run_date=YYYY-MM-DD/{ticker}.csv`

## 8. Failure Handling and Recovery
- Fetch failure (single ticker): log alert and continue to next company
- MinIO write failure: retry 3 times; if still failed, mark that company as failed in this run
- PostgreSQL write failure: rollback transaction and record failed batch
- Recovery strategy: rerun failed company list by `run_id` or `run_date`

## 9. Change Management
- If data source/API, formula definition, or primary-key rules change, this lineage document must be updated synchronously.
- Recommended linkage with `CHANGELOG.md`: change time, author, impact scope, rollback plan.
