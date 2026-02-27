# Data Dictionary

## 1. Document Purpose
This document supports “Create and update data dictionaries” by defining field semantics, data types, constraints, calculation definitions, and examples for data assets in Coursework One, ensuring consistent understanding across the team.

## 2. Table: systematic_equity.company_static (Input)

| Field Name | Type (Recommended) | Nullable | Description | Example | Rule |
|---|---|---|---|---|---|
| company_id | INT | No | Unique company identifier | 10123 | Primary key / unique |
| ticker | VARCHAR(20) | No | Trading ticker | AAPL | Uppercase, trimmed |
| company_name | VARCHAR(255) | Yes | Company name | Apple Inc. | Optional metadata |
| country | VARCHAR(50) | Yes | Listing country | US | Prefer ISO country code |
| sector | VARCHAR(100) | Yes | GICS sector | Information Technology | Used for sector-neutral/group standardization |

> Note: `company_static` is a course-provided table. Actual fields should follow the real database schema; the above is the minimum set required by the pipeline.

## 3. File: MinIO raw/run_date=YYYY-MM-DD/{ticker}.csv (Raw Market Data)

| Field Name | Type (Recommended) | Nullable | Description | Example | Rule |
|---|---|---|---|---|---|
| date | DATE | No | Trading date | 2026-02-26 | Unified as UTC date |
| close | DOUBLE PRECISION | No | Closing price (or adjusted close) | 182.31 | > 0 |
| adj_close | DOUBLE PRECISION | Yes | Adjusted close | 182.31 | Recommended to keep |
| volume | BIGINT | Yes | Trading volume | 52300000 | >= 0 |
| source | VARCHAR(50) | Yes | Data source | yahoo_finance | Audit field |
| ingest_ts | TIMESTAMP | Yes | Ingestion timestamp | 2026-02-27 12:00:00 | Audit field |

## 4. Table: systematic_equity.factor_values (Output)

Suggested DDL:

```sql
CREATE TABLE IF NOT EXISTS systematic_equity.factor_values (
    company_id INT NOT NULL,
    factor_date DATE NOT NULL,
    factor_name VARCHAR(50) NOT NULL,
    factor_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, factor_date, factor_name)
);
```

Field definitions:

| Field Name | Type | Nullable | Description | Example | Rule |
|---|---|---|---|---|---|
| company_id | INT | No | Company ID, linked to `company_static` | 10123 | Logical foreign-key semantics |
| factor_date | DATE | No | Factor date | 2026-02-26 | Aligned with input market-data date |
| factor_name | VARCHAR(50) | No | Factor name | momentum | Recommended enum: value/quality/momentum/lowvol |
| factor_value | DOUBLE PRECISION | Yes | Factor value | 0.1432 | Supports negative and null values (when history is insufficient) |
| created_at | TIMESTAMP | No | Record creation timestamp | 2026-02-27 12:01:11 | Defaults to current time |

## 5. Calculated Field Dictionary (Core Factor Definitions)

### 5.1 Momentum
- Field name: `momentum_score`
- Definition (dual-signal excess momentum, excluding the most recent month):

$$
\mathrm{MOM}_{i,6,t} = \left(\frac{P_{i,t-1}}{P_{i,t-7}} - 1\right) - RF^{(1m)}_{c,t}
$$

$$
\mathrm{MOM}_{i,12,t} = \left(\frac{P_{i,t-1}}{P_{i,t-13}} - 1\right) - RF^{(1m)}_{c,t}
$$

Where:
- \(P_{i,t-1}\): adjusted close price of stock \(i\) at month \(t-1\)
- \(P_{i,t-7}\): adjusted close price of stock \(i\) at month \(t-7\)
- \(P_{i,t-13}\): adjusted close price of stock \(i\) at month \(t-13\)
- \(RF^{(1m)}_{c,t}\): 1-month risk-free rate for country \(c\) at month \(t\)

Sector z-score standardization:

$$
z_{i,6,t} = \frac{\mathrm{MOM}_{i,6,t} - \mu_{\text{sector},6,t}}{\sigma_{\text{sector},6,t}},
\qquad
z_{i,12,t} = \frac{\mathrm{MOM}_{i,12,t} - \mu_{\text{sector},12,t}}{\sigma_{\text{sector},12,t}}
$$

Composite momentum score:

$$
\mathrm{Momentum\ Score}_{i,t} = 0.5\,z_{i,6,t} + 0.5\,z_{i,12,t}
$$

Implementation note (if risk-free data is unavailable, use price momentum):

$$
\mathrm{MOM}^{\text{price}}_{i,6,t} = \frac{P_{i,t-1}}{P_{i,t-7}} - 1,
\qquad
\mathrm{MOM}^{\text{price}}_{i,12,t} = \frac{P_{i,t-1}}{P_{i,t-13}} - 1
$$

- Input fields: `date`, `close` (optional `risk_free_rate_1m`)
- Output persistence: `factor_name='momentum'`, `factor_value=momentum_score`

### 5.2 Low Volatility
- Field name: `lowvol_score`
- Daily log return:

$$
r_{i,d} = \ln\!\left(\frac{P_{i,d}}{P_{i,d-1}}\right)
$$

- Rolling volatility on window \(w\in\{63,252\}\):

$$
\sigma_{i,t}^{(w)} = \mathrm{SD}\!\left(r_{i,d}\right)_{d=t-w+1}^{t}
$$

- Sector cross-sectional standardization:

$$
Z_{i,t}^{(w)} = \frac{\sigma_{i,t}^{(w)} - \mu_{\text{sector},t}^{(w)}}{\sigma_{\text{sector},t}^{(w)}}
$$

- Composite low-vol score (lower volatility is better, so use negative sign):

$$
\mathrm{LowVol\ Score}_{i,t} = -\frac{1}{2}\left(Z_{i,t}^{(63)} + Z_{i,t}^{(252)}\right)
$$

- Dependency: adjusted close price time series with sufficient daily history

### 5.3 Value
- Field name: `value_score`
- Asset growth:

$$
\mathrm{ASSETG}_{i,t} = \frac{\mathrm{TA}_{i,t-1} - \mathrm{TA}_{i,t-2}}{\mathrm{TA}_{i,t-2}}
$$

- Book value per share and price-to-book:

$$
\mathrm{BVPS}_{i,t-1} = \frac{\mathrm{BVE}_{i,t-1}}{\mathrm{Shares}_{i,t-1}},
\qquad
\mathrm{P/B}_{i,t} = \frac{P_{i,t}}{\mathrm{BVPS}_{i,t-1}}
$$

- Cross-sectional z-score:

$$
Z_{i,h,t} = \frac{x_{i,h,t} - \mu_{h,t}}{\sigma_{h,t}}
$$

- Composite value score (lower ASSETG and lower P/B are preferred):

$$
\mathrm{Value\ Score}_{i,t} = \frac{-Z_{\mathrm{ASSETG},i,t} - Z_{\mathrm{P/B},i,t}}{2}
$$

- Dependencies: total assets, book value of equity, shares outstanding, adjusted close price

### 5.4 Quality
- Field name: `quality_score`
- Profitability metrics:

$$
\mathrm{ROE}_{i,t} = \frac{\mathrm{Net\ Income}_{i,t}}{\mathrm{Average\ Equity}_{i,t}},
\qquad
\mathrm{ROA}_{i,t} = \frac{\mathrm{Net\ Income}_{i,t}}{\mathrm{Average\ Total\ Assets}_{i,t}}
$$

- Leverage:

$$
\mathrm{LEV}_{i,t} = \frac{\mathrm{Total\ Debt}_{i,t}}{\mathrm{Book\ Equity}_{i,t}}
$$

- Earnings stability:

$$
\mathrm{EVAR}_{i,t} = \mathrm{SD}\!\left(\Delta\mathrm{EPS}_{i,t-4:t}\right)
$$

- Standardization:

$$
Z_{i,h,t} = \frac{x_{i,h,t} - \mu_{h,t}}{\sigma_{h,t}}
$$

- Composite quality score (higher ROE/ROA preferred, lower LEV/EVAR preferred):

$$
\mathrm{Quality\ Score}_{i,t} = \frac{Z_{\mathrm{ROE},i,t} + Z_{\mathrm{ROA},i,t} - Z_{\mathrm{LEV},i,t} - Z_{\mathrm{EVAR},i,t}}{4}
$$

- Dependencies: net income, total assets, equity, debt, and at least 5 years of EPS history

## 6. Data Validation Rules (Before Write)
- Primary-key fields must not be null: `company_id`, `factor_date`, `factor_name`
- Date validity: must not be later than run date `run_date`
- Numeric validity: price > 0; volatility >= 0
- Deduplication rule: keep only one record per `(company_id, factor_date, factor_name)`

## 7. Update Mechanism
- If any field is added/renamed/type-changed, update this dictionary synchronously.
- If calculation definitions change, a new “definition version” (e.g., `momentum_v2`) must be introduced and its effective date recorded.
