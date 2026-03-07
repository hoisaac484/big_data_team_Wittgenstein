#Tests implemented so far 7/03/26


Run from the coursework_one/ directory:


poetry run python3 -m pytest ./tests/ -v
What you'll see in the output:

Which tests pass/fail — each test shows PASSED or FAILED next to its name:


tests/test_data_validator.py::TestValidatePrices::test_happy_path PASSED
tests/test_main.py::TestLoadConfig::test_file_not_found FAILED
Coverage percentage — a table at the bottom shows per-file coverage:


Name                                   Stmts   Miss  Cover   Missing
modules/db/db_connection.py              135      2    99%   267, 306
modules/input/data_collector.py          257     92    64%   288, 297, 333-345...
What hasn't been tested — the Missing column shows exact line numbers not covered. For example 333-345 means lines 333 to 345 in that file were never executed during tests.

The coverage table auto-runs because of this line in pyproject.toml:


addopts = "--cov=modules --cov-report=term-missing"
Quick reference:

Cover = percentage of lines tested
Miss = number of untested lines
Missing = which specific lines are untested (you can open the file and check those line numbers)
Currently: 94 passed, 0 failed, 82% total coverage.




Test Suite Summary
94 tests, 82% coverage, all passing.

What's being tested:
1. Data Validator (28 tests) 

ValidationResult — warnings don't fail, errors do fail, summary output is correct
Price validation — catches empty data, zero/negative prices (warning for open/high/low, error if >1% of close prices are bad), short date range, too few rows per symbol, duplicate rows, high null rates, low symbol coverage (<95%)
Financials validation — catches empty data, negative total assets, duplicate quarters, high null rates
Risk-free rates validation — catches empty data, rates outside -10% to 100%, stale dates (>1 year old)
clean_prices() — strips zero/negative close prices, handles empty/null input
validate_all() — runs all three validations and returns combined results
2. Data Collector (23 tests)

_reshape_price_df — transforms yfinance output to our schema columns, handles MultiIndex columns, handles None/empty input
_safe_get — safely extracts values from DataFrames, returns None for missing keys or NaN
_classify_missing — correctly labels symbols as "delisted" vs "fetch_error" based on yfinance ticker info, handles API exceptions
Caching — CTL/parquet path generation, writing CTL files, caching DataFrames, loading from cache, marking as loaded to PostgreSQL
fetch_prices — skips cached symbols (no API call), downloads uncached symbols via yfinance
fetch_fundamentals — parallel fetching with ThreadPoolExecutor
_fetch_single_fundamental — extracts balance sheet + income statement into correct schema
3. Data Writer (16 tests)

write_prices/financials/rates — writes new rows, skips duplicates already in PostgreSQL, returns 0 for empty input
write_factor_metrics/scores — writes new rows correctly
mark_loaded — calls fetcher.mark_loaded for each written symbol
MongoDB logging — converts DataFrames to records, inserts documents, groups by symbol for batch logging, skips None input
get_table_counts — queries row counts, returns 0 on errors
4. DB Connections (20 tests)

PostgreSQL — engine creation, read_query, write_dataframe, execute SQL, execute SQL file, get_company_list, connection test (pass + fail)
MongoDB — insert_one, insert_many, find, connection test (pass + fail)
MinIO — object_exists (found + not found), bucket creation, upload/download JSON, upload/download parquet, list objects, connection test (pass + fail)
5. Main Pipeline (7 tests)

load_config — loads YAML correctly, raises FileNotFoundError when missing
setup_logging — sets correct log level
print_validation_report — outputs formatted report to console
Full pipeline — verifies correct call order: fetch → log to MongoDB → clean → validate → write
Strict mode — halts pipeline when validation fails (write_prices never called)
Connection failure — raises RuntimeError when PostgreSQL is unreachable