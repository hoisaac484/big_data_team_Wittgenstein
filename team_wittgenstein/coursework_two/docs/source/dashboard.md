# Dashboard Overview

The Streamlit app lives under `dashboard/` and is launched with:

```bash
poetry run streamlit run dashboard/Home.py
```

## Structure

- `Home.py`: landing page and high-level strategy overview
- `pages/`: page entrypoints for performance, comparison, portfolio composition,
  stock deep dives, factor analysis, and methodology
- `lib/`: reusable dashboard helpers for queries, formatting, charts,
  components, theme, and database access

## Pages

- `1_Performance.py`: scenario-level return, drawdown, turnover, and risk plots
- `2_Compare_Scenarios.py`: side-by-side scenario comparison
- `3_Strategy_Tuner.py`: parameter sensitivity exploration
- `4_Portfolio_Composition.py`: holdings, sector mix, and constraint health
- `5_Stock_Deep_Dive.py`: per-stock positions, factor history, and fundamentals
- `6_Factor_Analysis.py`: IC weights, distributions, and factor correlation views
- `7_Methodology.py`: narrative explanation of formulas and design decisions

## Helper layers

- `dashboard.lib.db`: cached SQLAlchemy connection and raw query wrapper
- `dashboard.lib.queries`: page-facing query functions with Streamlit caching
- `dashboard.lib.format`: display formatting helpers
- `dashboard.lib.components`: shared UI blocks for headers, KPI cards, badges
- `dashboard.lib.charts`: Plotly figure factories
- `dashboard.lib.theme`: central palette, template, and injected CSS

## Documentation approach

The dashboard page files are Streamlit entrypoints rather than reusable Python
APIs, so this documentation keeps them at a descriptive level. The reusable
helpers in `dashboard/lib/` can be documented further later if you want a
second API section for the UI layer.
