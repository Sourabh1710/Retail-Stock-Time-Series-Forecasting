# Retail + Stock Time Series Forecasting

A dual-domain time series project - same methodology applied to real Kaggle retail sales data and NSE stock prices. Built to demonstrate technical depth and financial domain reasoning for Data Science roles in Finance and FinTech.

---
[Live Demo]() &nbsp;

---

## What this project covers

| Part | Data | Models | Output |
|------|------|--------|--------|
| 1 | Ecuador grocery chain - Kaggle, 3M rows, real | SARIMA + Prophet | 90-day sales forecast |
| 2 | TCS.NS stock price - yfinance, live NSE data | ARIMA + Prophet | 90-day price forecast |
| App | Any NSE stock, live | ARIMA + Prophet | Interactive Streamlit dashboard |

---

## Key findings

- **Oil price correlates with grocery sales at r = −0.47:**
  Ecuador is an oil-export economy. When global oil prices fall, government revenue drops, social spending gets cut, and consumers buy less. Found during EDA before any modelling - the kind of macro reasoning that matters in FinTech roles.

- **April 17, 2016 is the all-time sales peak at 90,558 units:**
  The day after a 7.8-magnitude earthquake hit Ecuador, nationwide panic buying drove sales to roughly 2× a normal Saturday. This is why Prophet's holiday events feature exists - the earthquake aftermath is explicitly modelled as a named 31-day event.

- **SARIMA beats Prophet on the retail test window (8.8% vs 12.1% MAPE):**
  The test period (June–August 2017) contains no major holidays. Prophet's main strength is modelling holiday effects - without events in the test window, SARIMA's autocorrelation structure wins. On a window spanning Christmas or the earthquake period, Prophet would likely recover. The lesson: model selection depends on what the forecast horizon contains, not just overall model capability.

- **auto_arima selects ARIMA(0,1,1) on real TCS.NS data:**
  Not the textbook ARIMA(0,1,0) pure random walk, but close. The MA(1) term detects a small short-lived autocorrelation in returns — consistent with weak-form market efficiency rather than strict EMH. Too small to exploit after transaction costs, but worth noting. The order varies by time window: a different date range may return (0,1,0), which is also valid.

---

## Results

### Part 1 — Retail Sales (Store 44, Quito)

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| **SARIMA(3,1,0)(2,1,0,7)** | 5,662 | 4,046 | **8.8%** ✓ |
| Prophet + 178 holiday events | 5,801 | 4,919 | 12.1% |

### Part 2 — TCS.NS Stock Price

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ARIMA(0,1,1) | 327 | 276 | 15.8% |
| **Prophet + earnings events** | 263 | 221 | **13.1%** ✓ |

---

## Project structure

```
retail_stock_forecasting/
│
├── retail_stock_forecasting.ipynb   # Main notebook - 53 cells, fully executed
├── app.py                           # Interactive NSE forecasting dashboard
├── requirements.txt                 # Notebook and app dependencies
│
├── data/
│   ├── train.csv                    # 3M rows - Kaggle Store Sales competition
│   ├── holidays_events.csv          # 350 real Ecuadorian holiday events
│   ├── oil.csv                      # WTI crude oil daily prices
│   └── stores.csv                   # 54 store metadata
│
└── outputs/
    ├── 01_eda.png                   # Daily sales series with earthquake annotation
    ├── 02_monthly_annual.png        # Monthly and annual aggregations
    ├── 02b_oil_correlation.png      # Oil price vs sales dual-axis chart (r = −0.47)
    ├── 03_decomposition.png         # Trend, seasonality, residual breakdown
    ├── 04_acf_pacf.png              # ACF/PACF plots for ARIMA parameter selection
    ├── 05_arima_forecast.png        # SARIMA test-set forecast
    ├── 06_prophet_forecast.png      # Prophet test-set forecast with confidence band
    ├── 07_prophet_components.png    # Prophet's trend, holiday, weekly, yearly components
    ├── 08_model_comparison_retail.png
    ├── 09_retail_90day_forecast.html   <- interactive Plotly chart
    ├── 10_stock_eda.png             # Price, log returns, rolling volatility
    ├── 11_stock_decomposition.png   # Log-price decomposition
    ├── 12_stock_arima.png           # ARIMA test-set forecast
    ├── 13_stock_prophet.png         # Prophet forecast with earnings date markers
    ├── 14_model_comparison_stock.png
    └── 15_stock_90day_forecast.html    <- interactive Plotly chart
```

---

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook retail_stock_forecasting.ipynb
# Kernel → Restart & Run All
```

The notebook uses real Kaggle data from `data/`. Download the dataset from the link at the bottom if you don't have `train.csv`.

---

## Running the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `localhost:8501`. Select any of 10 NSE stocks, choose a historical period, set forecast horizon (30–90 days), and click Run Forecast.

The app downloads live data via yfinance. If yfinance is unavailable (e.g. restricted network), it falls back to synthetic GBM data with the same ticker-specific seed so the shape is consistent across reloads.

---

## What the Streamlit app does

Three tabs:

**Price History** - candlestick chart with volume bars, MA20 and MA50 overlays, and six metric cards: current price, 52-week high/low, annualised return, annualised volatility, YTD return.

**Forecast** - fits ARIMA and Prophet on historical log-prices and shows a dual forecast chart with confidence bands. Displays the ARIMA order selected by auto_arima with a plain-English interpretation (pure random walk vs weak-form EMH). Shows 90-day price targets for both models with confidence intervals.

**Analysis** - daily log returns bar chart, 30-day rolling volatility, collapsible time series decomposition, and a reference table explaining what each possible ARIMA order means financially.

---

## Concepts covered

- **Data engineering for time series:**
  Raw data is 3M rows across 54 stores and 33 product families. 
  Steps: filter to one store, `groupby('date')['sales'].sum()` to aggregate families, `reindex()` to create a continuous date range (required for `seasonal_decompose`), `fillna(0)` for days the store was closed. Using `usecols` on the read keeps memory reasonable.

- **Time series decomposition:**
  `seasonal_decompose(period=7)` splits the series into trend, weekly seasonality, and residuals using a centred moving average. Residuals should look like white noise - any remaining pattern means the decomposition hasn't fully captured the signal.

- **Stationarity and ADF test:**
  ADF test null hypothesis: the series has a unit root (non-stationary).Reject if p < 0.05. Retail sales pass with p = 0.034. Stock prices fail with p = 0.62 - the classic result for financial prices. Stock log-returns pass with p ≈ 0.

- **ARIMA and SARIMA:**
  `auto_arima` runs a stepwise AIC-minimising search. For retail, it finds SARIMA(3,1,0)(2,1,0,7) - seasonal terms with m=7 capturing weekly cycles. For stocks, it finds ARIMA(0,1,1) on real TCS data - near-random walk with a small MA(1) correction. `d=1` in both cases: first differencing removes the trend and achieves stationarity.

- **Prophet holiday engineering:**
  Raw `holidays_events.csv` has 350 rows. The correct approach: filter to the store's locale (National + Quito + Pichincha), drop `transferred=True` rows (those holidays were officially moved - use the Transfer-type row on the new date instead), and group all 31 earthquake aftermath days under one event name. Creating 31 separate dummy variables for 31 training examples would overfit.

- **Model selection reasoning:**
  SARIMA wins on retail despite Prophet having access to 178 real holiday events. The reason is the test window (June–August 2017) has no major holidays. This is documented in the notebook rather than glossed over — explaining why a model wins matters more than the number itself.

- **Stock forecasting and EMH:**
  ARIMA(0,1,1) on TCS.NS is close to a pure random walk, consistent with the Efficient Market Hypothesis. Prophet adds marginal value via changepoint detection for trend regime shifts but doesn't dramatically outperform. Earnings announcement dates are added as Prophet events with a ±2 day window to capture pre-announcement drift and post-result digestion.

---

## Planned extensions

- Oil price as Prophet external regressor via `add_regressor` - motivated by the r = -0.47 EDA finding
- Multi-store forecasting across all 54 stores with performance breakdown by store type A-E
- Walk-forward cross-validation instead of a single train/test split for more robust evaluation
- GARCH model for stock volatility - complements Prophet's trend component

---

## Data source

Retail data: [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
Stock data: NSE via yfinance - no account required, downloads automatically at runtime