# Introduction

## Investment product vision

The proposed investment product is a systematic, multi-factor equity strategy designed to generate risk-adjusted excess returns through disciplined, quantitative portfolio construction. The strategy integrates four factors, Value, Quality, Momentum, and Low Volatility, into a structured investment framework. The vision of the product is to bridge academic asset pricing research with institutionally implementable portfolio design, transforming raw financial and market data into investable signals through a scalable data pipeline. Instead of relying on discretionary stock selection, the product integrates transparent factor calculations and decision rules, repeatable investment processes, and data-driven portfolio allocation.

The unique value proposition of the proposed investment product lies in its 130/30 long-short structure capitalizes relative mispricing while maintaining market participation. It also differentiates with a multi-factor architecture that integrates complementary return drivers to mitigate factor cyclicality. Lastly, the industry-neutralisation helps lower the unintended sector-tilted stock selection and concentration risk.

## Product roadmap

The product seeks to outperform the investment results of long-only multi-factor ETFs and quantitative active funds. The product seeks to achieve its investment objective by investing 130% of capital in the long portfolio and 30% in the short portfolio, resulting in a net long exposure of 100% based on the ranking via an equally-weighted composite score model. The product innovates by directly linking academic factor research to an automated data and portfolio construction pipeline, ensuring consistent, scalable, and transparent implementation. The product’s expected benefit includes enhanced diversification of return drivers, improved downside risk control, and maintainable data architecture and infrastructure suitable for institutional portfolio management.

However, the product is subject to certain risks that may affect its total return, performance, and the ability to meet its investment objectives. For instance, market risk. Since the investable universe consists mainly of U.S. firms, fluctuations in the U.S. economy may have a severe impact on the stocks to which the product is exposed. Potential costs may also arise from high portfolio turnover, transaction expenses, and short-selling financing requirements.

The product is primarily designed for institutional portfolio managers seeking systematic equity exposure and asset managers implementing factor-based strategies. It is essential to recognise that the product may not be suitable for ESG-focused investors as it doesn’t seek to take sustainable, impact, or ESG-related factors into account.

## Competitive analysis

The competitive landscape of the product includes leading multi-factor ETFs, traditional active equity funds, and quantitative hedge funds. Multi-factor ETFs, such as Vanguard U.S. Multifactor ETF VFMF, iShares Edge MSCI Multifactor USA ETF LRGF, and Xtrackers Russell 1000 Comprehensive Factor ETF DEUS, offer diversified factor exposure to stock selection while maintaining long-only positions (Bryan, 2018). Traditional active funds rely on discretionary decision-making and stock picking, which may introduce inconsistency and reduce transparency. Quantitative hedge funds employ long–short approaches similar to our product but typically involve higher fees and limited investor accessibility. By contrast, the product occupies a strategic position between passive factor investing and alternative hedge fund strategies.

Bryan, A. (2018, April 4). A Closer Look at 3 Strong Multifactor ETFs. Morningstar. Retrieved February 24, 2026, from https://www.morningstar.com/funds/closer-look-3-strong-multifactor-etfs

## 1. Investment strategy

The core objective of this study is to construct a robust quantitative investment framework by integrating four academically validated equity factors: Value, Quality, Momentum, and Low Volatility. The aim of factor investing is to capture specific premiums in order to achieve long-term excess returns. Within this framework, each factor can be considered a form of compensated risk, forming the foundational elements of the portfolio’s return (Goroshko, 2024). By employing a multi-factor approach, we seek to construct a well-rounded portfolio that captures diverse sources of returns. This strategy mitigates the cyclicality of any single factor, ensuring the portfolio remains resilient across different market conditions.

Strategic Roles of Selected Factors:

- Value Factor: Targets undervalued stocks using low P/B and disciplined Asset Growth to capture mean-reversion premiums.
- Quality Factor: Selects for high ROE, low leverage, and earnings stability to ensure resilience and avoid value traps.
- Momentum Factor: Leverages historical return persistence to align the portfolio with market trends and reduce exposure to stagnant stocks.
- Low Volatility Factor: Captures the low-volatility anomaly to minimize drawdowns and achieve a smoother equity curve.

To synthesise these diverse metrics into a tradable signal, we apply cross-sectional Z-score standardization. The process neutralises the scale differences between accounting ratios and market data. Each factor's sub-ratios are aggregated into a Composite Score, ensuring that the final stock selection reflects a symmetric exposure to valuation, fundamental strength, price trend, and risk discipline.

### 1.1 Value Factor

#### 1.1.1 Factor Overview

Value factor refers to a set of characteristics that help investors identify undervalued stocks with the potential for long-term capital appreciation. This approach is based on the principle that undervalued companies, when compared to their intrinsic value, can offer higher returns over time (Tamplin, 2024). While traditional definitions focus on low price multiples, modern applications also incorporate financial soundness to address the pitfalls of value investing, most notably “value traps”—stocks that appear cheap but which in fact do not appreciate (MSCI, 2017). By identifying companies with robust balance sheets and disciplined capital allocation, the value factor avoids these traps to ensure a “margin of safety” (Graham, 1949).

#### 1.1.2 Required Data

The composite value factor relies on a combination of accounting-based financial statement data and market pricing data.

Required raw data fields:

- Total Assets (the most recent fiscal year and the preceding one)
- Adjusted Closing Price
- Book Value of Equity (the most recent fiscal year)
- Shares Outstanding (the most recent fiscal year)

#### 1.1.3 Ratios Used

(1) Annual Firm Asset Growth Rate

$$
\mathrm{ASSETG}_{i,t} = \frac{\mathrm{TA}_{i,t-1} - \mathrm{TA}_{i,t-2}}{\mathrm{TA}_{i,t-2}}
$$

Note: The asset growth rate is calculated using the year-on-year percentage change in total assets, which captures the firm’s investment behavior from year t-2 to t-1 to avoid look-ahead bias and to ensure all accounting information was fully disclosed to the market.

(2) Price‑to‑Book Ratio

$$
\mathrm{P/B}_{i,t} = \frac{P_{i,t}}{\mathrm{BVPS}_{i,t-1}}
$$

$$
\mathrm{BVPS}_{i,t-1} = \frac{\mathrm{BVE}_{i,t-1}}{\mathrm{Shares}_{i,t-1}}
$$

Note: The P/B ratio is calculated using the current market price (t) relative to the most recent fiscal year-end book value (t-1). This time-lagged approach ensures the ratio reflects only the fundamental data that was publicly available to investors at the time of portfolio formation.

#### 1.1.4 Update Frequency

Accounting data, including total assets, book value of equity, and shares outstanding, is updated annually following the release of fiscal year-end reports to ensure information is fully disclosed and free from look-ahead bias. Conversely, adjusted closing price is updated daily to capture real-time valuations at the point of portfolio rebalancing.

#### 1.1.5 Data Sources

Accounting data are sourced from Alpha Vantage, which aggregates fundamental financial metrics directly from official SEC EDGAR filings. Simultaneously, adjusted closing prices are retrieved from Yahoo Finance via historical pricing logs.

#### 1.1.6 Z-Score Standardization

To combine the metrics into a single composite score, each raw value is standardized using cross-sectional Z-scores.

The Z-score for each firm i within a specific period is calculated as:

$$
Z_i = \frac{X_i - \mu_{\text{sector}}}{\sigma_{\text{sector}}}
$$

Where:

- Xi: The raw metric value for firm i
- μsector: The cross-sectional mean
- σsector: The cross-sectional standard deviation

To form the final ranking, a sign reversal is applied to both metrics, as lower Asset Growth and P/B ratios are empirically linked to higher returns. These adjusted Z-scores are then equally weighted to calculate a single composite score, ensuring that both valuation and investment discipline contribute symmetrically to the final stock selection.

$$
\mathrm{Value\ Score}_i = 0.5\,(-Z_{\mathrm{AG},i}) + 0.5\,(-Z_{\mathrm{P/B},i})
$$

### 1.2 Quality Factor

#### 1.2.1 Factor Overview

The Quality factor aims to identify firms with strong and sustainable fundamental characteristics. In this project, Quality is defined through three key dimensions:

- Profitability
- Low Financial Leverage
- Earnings Stability

The goal is to select firms exhibiting high Return-on-Equity (ROE), low leverage, and low earnings variability. This definition is aligned with institutional methodologies such as the MSCI World Sector Neutral Quality Index.

#### 1.2.2 Required Data

The factor relies exclusively on accounting-based financial statement data.

Required raw data fields:

- Net Income
- Total Assets
- Total Equity
- Total Debt
- Earnings Per Share (EPS)
- At least five years of historical EPS data

#### 1.2.3 Ratios Used

(1) Profitability

Return on Equity (ROE)
$$
\mathrm{ROE}_{i,t} = \frac{\mathrm{Net\ Income}_{i,t}}{\mathrm{Average\ Equity}_{i,t}}
$$

Return on Assets (ROA)
$$
\mathrm{ROA}_{i,t} = \frac{\mathrm{Net\ Income}_{i,t}}{\mathrm{Average\ Total\ Assets}_{i,t}}
$$

Higher values indicate stronger profitability.

(2) Leverage

Total debt /Equity (Book value)
$$
\mathrm{LEV}_{i,t} = \frac{\mathrm{Total\ Debt}_{i,t}}{\mathrm{Book\ Equity}_{i,t}}
$$

Lower values are preferred, indicating lower financial risk.

(3) Earnings Stability

Earnings Variability
Standard deviation of Year-on-Year EPS growth over the last five fiscal years.
$$
\mathrm{EVAR}_{i,t} = \mathrm{SD}\!\left(\Delta \mathrm{EPS}_{i,t-4:t}\right)
$$

Lower variability indicates more stable and predictable earnings.

#### 1.2.4 Update Frequency

The Quality factor will be updated on a quarterly basis, aligned with financial reporting cycles.
Quarterly updates ensure timely incorporation of new accounting information while limiting excessive portfolio turnover.

#### 1.2.5 Data Sources

All financial data will be extracted from:

- Alpha Vantage – Income Statement and Balance Sheet data
- Historical financial statement filings for at least five years

#### 1.2.6 Z-Score Standardization

To combine the three components into a single composite Quality score, each metric is standardized using cross-sectional Z-scores.

For each variable:

$$
Z_{i,h} = \frac{x_{i,h} - \mu_h}{\sigma_h}
$$

Where:

- $x_{i,h}$ = firm-level metric
- $\mu_h$ = cross-sectional mean
- $\sigma_h$ = cross-sectional standard deviation

For leverage and earnings variability (where lower values are preferred), the Z-score is multiplied by -1 to ensure higher standardized values always indicate higher quality.

The final composed Quality score is calculated as the average of the standardized component scores:

$$
\mathrm{Quality\ Score}_i = \frac{Z_{\mathrm{ROE},i} + Z_{\mathrm{ROA},i} - Z_{\mathrm{LEV},i} - Z_{\mathrm{EVAR},i}}{4}
$$

### 1.3 Momentum Factor

#### 1.3.1 Factor Overview

Momentum is a cross-sectional equity factor that captures the empirical persistence of intermediate-horizon stock performance, emphasising buying the outperforming stocks and selling the underperforming stocks (Carhart, 1997; Moskowitz & Grinblatt, 1999). Substantial evidence documents the significance of the momentum factor in delivering abnormal returns. The study of Jegadeesh and Titman (1993) demonstrates positive returns over the 3 to 12-month holding period, and Carhart (1997) also formulates the momentum factor into his four-factor model, which establishes momentum as a systematic anomaly.

#### 1.3.2 Required Data

Required raw data fields:

- Adjusted close price (Daily)
- Risk-free rate (local short-term interest rate in each country)

Daily adjusted close price for each stock over the past 5 years is required and sampled at the month-end to construct the monthly price series for the momentum factor calculations. The adjusted close price is used as it incorporates dividends, stock splits, and other corporate actions. This adjustment is critical to ensure momentum metrics reflect actual investor returns rather than mechanical price changes from corporate events.

#### 1.3.3 Ratios Used

Our momentum factor construction follows a dual-signal approach similar to BlackRock's iShare MSCI World ex Australia momentum ETF and MSCI momentum index methodology, which employs both 6-month and 12-month lookback periods (BlackRock, 2023; MSCI, 2021). Following Jegadeesh (1990), this project also mitigates the short-term reversal effect by excluding the most recent month from the calculation. Aligning with the academic approach, a 1-month risk-free rate is used to compute the excess return. The calculations are as follows,

(1) 6-month momentum

$$
\mathrm{MOM}_{i,6,t} = \left(\frac{P_{i,t-1}}{P_{i,t-7}} - 1\right) - RF_{c,t}^{(1m)}
$$

Where:

- Pt-1 is the adjusted close price 1 month ago
- Pt-7 is the adjusted close price 7 months ago
- RFlocal is the 1-month local risk-free rate in local currency of the country

(2) 12-month momentum

$$
\mathrm{MOM}_{i,12,t} = \left(\frac{P_{i,t-1}}{P_{i,t-13}} - 1\right) - RF_{c,t}^{(1m)}
$$

Where:

- Pt-1 is the adjusted close price 1 month ago
- Pt-13 is the adjusted close price 13 months ago
- RFlocal is the 1-month local risk-free rate in local currency of the country

Given the universe consists of several countries with varying interest rates, the momentum is calculated using excess returns rather than raw returns. This approach ensures fair cross-country comparison and prevents systematic bias toward high-rate countries. However, to maintain the data pipeline is simple and replicable, we employ a simple price momentum rather than using a risk-adjusted price momentum.

#### 1.3.4 Update Frequency

The momentum metrics are suggested to be updated monthly using rolling windows of historical price data to ensure the momentum signals reflect current price trends rather than stale information. Quarterly or annual updates would be infrequent, causing the indicator to miss signal changes that potentially signal momentum for stocks whose trends have already reversed.

#### 1.3.5 Data Sources

All of the price data will be extracted from Yahoo Finance, while the 1-month risk-free interest rate will be extracted from OECD API for the short-term interest rates in local currency (a proxy of risk-free interest rate).

#### 1.3.6 Z-Score Standardization

For the z-scores, it is computed within each GICS sector, not across the entire universe. The formula is as follows,

$$
z_{i,6,t} = \frac{\mathrm{MOM}_{i,6,t} - \mu_{\text{sector},6,t}}{\sigma_{\text{sector},6,t}}
$$

$$
z_{i,12,t} = \frac{\mathrm{MOM}_{i,12,t} - \mu_{\text{sector},12,t}}{\sigma_{\text{sector},12,t}}
$$

A positive z-score indicates the winning stocks that with above-avergae momentum value, while a negative z-score indicates the losing stocks that have below-average momentum value.

The final momentum score is calculated as the sum of equally-weighted z-scores of 6-month price momentum and 12-month price momentum.

$$
\mathrm{Momentum\ Score}_i = 0.5\,z_{i,6,t} + 0.5\,z_{i,12,t}
$$

### 1.4 Low Volatility Factor

#### 1.4.1 Factor Overview

Low Volatility is a risk factor based on historical return volatility. It is measured by estimating the standard deviation of historical stock returns. This project calculates annualized volatility based on past daily returns and ranks stocks to choose those with low volatility (Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X., 2006). The factor relies on high-quality, long-term continuous daily price data and requires a data structure supporting rolling calculations and periodic updates.

#### 1.4.2 Required Data

Low volatility factor based on historical price data, including daily adjusted close price for minimum 5 years history, company identifier, and trading dates.

#### 1.4.3 Ratios Used

The main ratios used to construct the low volatility factor are Quarterly Historical Volatility and Annual Historical Volatility.

(1) Quarterly Historical Volatility

Step 1: daily return calculation

$$
r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
$$

where Pi,t is the adjusted closing price of stock i at time t, adjusted prices are used to account for dividends and stock splits (Baker, M., Bradley, B., & Wurgler, J., 2011).

Step 2: Daily and Quarterly Historical Volatility

$$
\sigma^{(63)}_{i,\text{daily},t} = \mathrm{SD}\!\left(r_{i,t-62:t}\right)
$$

Where the rolling window is 63 days.

$$
\sigma_{i,3m,t} = \sqrt{63}\,\sigma^{(63)}_{i,\text{daily},t}
$$

63 is the approximate number of trading days in 3 months. This shorter-term measure captures recent changes in risk and allows the model to detect volatility regime shifts.

(2) Annual Historical Volatility

$$
r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
$$

$$
\sigma^{(252)}_{i,\text{daily},t} = \mathrm{SD}\!\left(r_{i,t-251:t}\right)
$$

Where the rolling window is 252 days.

$$
\sigma_{i,12m,t} = \sqrt{252}\,\sigma^{(252)}_{i,\text{daily},t}
$$

252 is the approximate number of trading days in one year. This measure captures medium-term risk exposure and is the primary signal used in portfolio construction.

#### 1.4.4 Update Frequency

The data and factor should be updated at different frequencies:

- Historical Price Data: Daily ingestion
- Volatility Calculation: Monthly rolling update
- Portfolio Rebalancing Signal: Monthly or Quarterly

#### 1.4.5 Data Sources

Financial data providers (e.g., Yahoo Finance, Bloomberg) – for daily adjusted prices.

#### 1.4.6 Z-Score Standardization

To make the factor comparable across companies, the volatility is standardized using cross-sectional Z-scores. For each rebalancing date:

$$
Z^{\mathrm{vol}}_{i,h,t} = \frac{\sigma_{i,h,t} - \mu^{\sigma}_{h,t}}{s^{\sigma}_{h,t}},\quad h \in \{3m,12m\}
$$

where $h \in \{3m,12m\}$, $\sigma_{i,h,t}$ is the volatility of stock $i$, $\mu^{\sigma}_{h,t}$ is the cross-sectional mean volatility, and $s^{\sigma}_{h,t}$ is the cross-sectional standard deviation of volatility. Since lower volatility is preferred (Blitz, D., & Van Vliet, P., 2007), the signal is inverted:

$$
\mathrm{LowVol\ Score}_i = -\frac{Z^{\mathrm{vol}}_{i,12m,t} + Z^{\mathrm{vol}}_{i,3m,t}}{2}
$$

This combined equation ensures to capture both long-term stability and short-term risk dynamics. And a high score of LowVol indicates lower volatility.
