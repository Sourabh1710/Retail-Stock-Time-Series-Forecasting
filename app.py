"""
NSE Stock Forecaster - Streamlit App
Time Series Forecasting with Prophet + ARIMA
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

# PAGE CONFIG
st.set_page_config(
    page_title="NSE Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CONSTANTS
TICKERS = {
    "TCS.NS":        "Tata Consultancy Services",
    "INFY.NS":       "Infosys",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro",
    "ICICIBANK.NS":  "ICICI Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "TATAMOTORS.NS": "Tata Motors",
    "LTIM.NS":       "LTIMindtree",
    "HINDUNILVR.NS": "Hindustan Unilever",
}

BASE_PRICES = {
    "TCS.NS": 3200, "INFY.NS": 1500, "RELIANCE.NS": 2400,
    "HDFCBANK.NS": 1600, "WIPRO.NS": 450, "ICICIBANK.NS": 1000,
    "BAJFINANCE.NS": 7000, "TATAMOTORS.NS": 800,
    "LTIM.NS": 5000, "HINDUNILVR.NS": 2300,
}

# DATA LOADING
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker: str, years: int) -> tuple[pd.DataFrame, bool]:
    """
    Download stock data via yfinance.
    Returns (dataframe, is_live) - falls back to synthetic GBM if blocked.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=f"{years}y")
        if df.empty:
            raise ValueError("Empty response")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].copy(), True
    except Exception:
        return _synthetic_stock(ticker, years), False


def _synthetic_stock(ticker: str, years: int) -> pd.DataFrame:
    """Geometric Brownian Motion fallback - same seed per ticker for consistency."""
    np.random.seed(abs(hash(ticker)) % 2**31)
    days = pd.bdate_range(end=pd.Timestamp.today(), periods=years * 252)
    n    = len(days)
    S0   = BASE_PRICES.get(ticker, 2000)
    mu, sigma = 0.00035, 0.014
    shocks     = np.random.normal(mu - 0.5 * sigma**2, sigma, n)
    prices     = S0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame({
        "Open":   np.round(prices * np.random.uniform(0.995, 1.005, n), 2),
        "High":   np.round(prices * np.random.uniform(1.005, 1.015, n), 2),
        "Low":    np.round(prices * np.random.uniform(0.985, 0.995, n), 2),
        "Close":  np.round(prices, 2),
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=days)


# MODEL FITTING  (cached — won't refit unless inputs change)
@st.cache_data(show_spinner=False)
def fit_prophet(close_json: str, forecast_days: int):
    """Fit Prophet on log(close) and return forecast DataFrame."""
    close = pd.read_json(io.StringIO(close_json), typ="series")
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:          # strip timezone - Prophet requires tz-naive ds
        close.index = close.index.tz_convert(None)

    df_p = pd.DataFrame({"ds": close.index, "y": np.log(close.values)})
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
    )
    m.fit(df_p)
    future = m.make_future_dataframe(periods=forecast_days, freq="B")
    return m.predict(future), m


@st.cache_data(show_spinner=False)
def fit_arima(log_close_list: list, forecast_days: int):
    """Fit auto_arima on log(close); return (order, predictions, conf_intervals)."""
    model = auto_arima(
        np.array(log_close_list),
        d=1, seasonal=False,
        max_p=3, max_q=3,
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )
    preds, conf = model.predict(n_periods=forecast_days, return_conf_int=True)
    return model.order, preds, conf


# METRICS
def compute_metrics(close: pd.Series) -> dict:
    log_ret    = np.log(close / close.shift(1)).dropna()
    current    = close.iloc[-1]
    day_change = (current - close.iloc[-2]) / close.iloc[-2] * 100
    ann_vol    = log_ret.std() * np.sqrt(252) * 100
    ann_ret    = (np.exp(log_ret.mean() * 252) - 1) * 100
    ytd_close  = close[close.index.year == close.index[-1].year]
    ytd_ret    = (current / ytd_close.iloc[0] - 1) * 100
    week_high  = close.tail(52 * 5).max()   # ~52 weeks × 5 trading days
    week_low   = close.tail(52 * 5).min()
    return dict(
        current=current, day_change=day_change,
        ann_vol=ann_vol, ann_ret=ann_ret, ytd_ret=ytd_ret,
        week_high=week_high, week_low=week_low,
    )


# CHARTS
def candlestick_chart(df: pd.DataFrame, ticker: str, name: str) -> go.Figure:
    up   = df["Close"] >= df["Open"]
    col  = np.where(up, "#26a69a", "#ef5350").tolist()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=col, opacity=0.6, name="Volume",
    ), row=2, col=1)

    # 20-day and 50-day moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"].rolling(20).mean(),
        line=dict(color="orange", width=1.2),
        name="MA20", opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"].rolling(50).mean(),
        line=dict(color="royalblue", width=1.2),
        name="MA50", opacity=0.8,
    ), row=1, col=1)

    fig.update_layout(
        title=dict(text=f"{name} ({ticker}) — Price History", font=dict(size=15)),
        xaxis_rangeslider_visible=False,
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=60),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1)
    fig.update_yaxes(title_text="Volume",    row=2)
    return fig


def forecast_chart(
    close: pd.Series, fc: pd.DataFrame,
    arima_order: tuple, arima_preds: np.ndarray, arima_conf: np.ndarray,
    forecast_days: int, ticker: str,
) -> go.Figure:
    last_date    = close.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    # fc["ds"] is datetime64[us] in Prophet ≥1.1 - compare via numpy to avoid resolution mismatch
    last_date_np = np.datetime64(last_date, 'us')
    prophet_fwd  = fc[fc["ds"].values > last_date_np].head(forecast_days)

    fig = go.Figure()

    # Historical (last 180 trading days for context)
    hist = close.tail(180)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical", line=dict(color="#1f77b4", width=1.8),
    ))

    # Prophet forecast line + CI band
    fig.add_trace(go.Scatter(
        x=prophet_fwd["ds"], y=np.exp(prophet_fwd["yhat"]),
        name="Prophet", line=dict(color="#2ca02c", width=2.2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([prophet_fwd["ds"], prophet_fwd["ds"].iloc[::-1]]),
        y=np.concatenate([
            np.exp(prophet_fwd["yhat_upper"].values),
            np.exp(prophet_fwd["yhat_lower"].values[::-1]),
        ]),
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Prophet 80% CI",
    ))

    # ARIMA forecast line + CI band
    n = min(len(arima_preds), len(future_dates))
    fig.add_trace(go.Scatter(
        x=future_dates[:n], y=np.exp(arima_preds[:n]),
        name=f"ARIMA{arima_order}", line=dict(color="#d62728", width=2.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates[:n], future_dates[:n][::-1]]),
        y=np.concatenate([np.exp(arima_conf[:n, 1]), np.exp(arima_conf[:n, 0][::-1])]),
        fill="toself", fillcolor="rgba(214,39,40,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="ARIMA 95% CI",
    ))

    # Vertical divider at forecast start
    fig.add_vline(x=str(last_date.date()), line_dash="dot", line_color="grey", opacity=0.6)
    fig.add_annotation(
        x=str(last_date.date()), y=close.tail(180).max(),
        text="Forecast →", showarrow=False,
        xanchor="left", font=dict(size=10, color="grey"),
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — {forecast_days}-Day Forward Forecast", font=dict(size=15)),
        xaxis_title="Date", yaxis_title="Price (₹)",
        height=480, template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified", margin=dict(t=60),
    )
    return fig


def decomposition_chart(close: pd.Series) -> go.Figure:
    result = seasonal_decompose(np.log(close.values), model="additive", period=5)
    titles = ["Observed (log price)", "Trend", "Weekly Seasonality", "Residual"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    comps  = [result.observed, result.trend, result.seasonal, result.resid]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.06)
    for i, (comp, color) in enumerate(zip(comps, colors), 1):
        fig.add_trace(go.Scatter(
            x=close.index, y=comp,
            line=dict(color=color, width=0.9), showlegend=False,
        ), row=i, col=1)

    fig.update_layout(
        height=620, template="plotly_white",
        title="Log-Price Decomposition  (period = 5 trading days)",
        margin=dict(t=60),
    )
    return fig


def returns_chart(close: pd.Series) -> go.Figure:
    log_ret  = np.log(close / close.shift(1)).dropna()
    roll_vol = log_ret.rolling(30).std() * np.sqrt(252) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Daily Log Returns", "30-Day Rolling Annualised Volatility (%)"],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Bar(
        x=log_ret.index, y=log_ret.values,
        marker_color=np.where(log_ret.values >= 0, "#26a69a", "#ef5350").tolist(),
        name="Log Return",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        fill="toself", fillcolor="rgba(100,149,237,0.3)",
        line=dict(color="cornflowerblue", width=1.2),
        name="30d Vol",
    ), row=2, col=1)

    fig.update_layout(
        height=420, template="plotly_white",
        showlegend=False, margin=dict(t=50),
    )
    return fig

# MAIN APP
def main():
    # Custom CSS 
    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stMetric label  { font-size: 0.78rem; color: #888; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    col_title, col_tag = st.columns([3, 1])
    with col_title:
        st.markdown("## 📈 NSE Stock Forecaster")
        st.caption("Prophet + ARIMA · Live NSE Data via yfinance · 90-Day Forward Forecast")
    with col_tag:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("`Time Series · Finance DS Portfolio`")

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        ticker = st.selectbox(
            "Stock",
            options=list(TICKERS.keys()),
            format_func=lambda x: f"{TICKERS[x]}  ({x})",
            index=0,
        )
        years = st.selectbox("Historical period", options=[2, 3, 5], index=1,
                             format_func=lambda x: f"{x} years")
        forecast_days = st.slider("Forecast horizon (trading days)", 30, 90, 90, step=10)

        st.divider()
        st.markdown("**About the models**")
        st.markdown("""
**ARIMA** fits on log(price) with d=1 (differencing removes the random walk trend).
`auto_arima` selects p and q via AIC - often finds ARIMA(0,1,0) to (0,1,2) on stocks,
consistent with the Efficient Market Hypothesis.

**Prophet** decomposes the series into trend + weekly seasonality, using
changepoint detection for structural regime shifts. Better at longer horizons.
        """)

        st.divider()
        st.caption("Built with: Prophet · pmdarima · yfinance · Streamlit · Plotly")

    # Load data
    stock_name = TICKERS[ticker]

    with st.spinner(f"Loading {ticker}..."):
        df, is_live = load_stock_data(ticker, years)

    if df.empty:
        st.error("No data returned. Try a different ticker.")
        return

    if is_live:
        st.success(f"Live data loaded - {len(df):,} trading days from NSE")
    else:
        st.info(
            " yfinance unavailable in this environment - showing synthetic data. "
            "Run locally to get live prices."
        )

    close = df["Close"]

    # Metric cards
    m = compute_metrics(close)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",         f"₹{m['current']:,.1f}",
                               f"{m['day_change']:+.2f}% today")
    c2.metric("52w High",      f"₹{m['week_high']:,.0f}")
    c3.metric("52w Low",       f"₹{m['week_low']:,.0f}")
    c4.metric("Ann. Return",   f"{m['ann_ret']:+.1f}%")
    c5.metric("Ann. Volatility", f"{m['ann_vol']:.1f}%")
    c6.metric("YTD Return",    f"{m['ytd_ret']:+.1f}%")

    st.divider()

    # Tabs
    tab_price, tab_forecast, tab_analysis = st.tabs([
        "📊 Price History",
        "🔮 Forecast",
        "🔬 Analysis",
    ])

    # Tab 1: Price History 
    with tab_price:
        st.plotly_chart(candlestick_chart(df, ticker, stock_name),
                        use_container_width=True)
        st.caption("Green/red candles = up/down days. Orange = MA20, Blue = MA50.")

    # Tab 2: Forecast
    with tab_forecast:
        col_desc, col_btn = st.columns([4, 1])
        with col_desc:
            st.markdown(f"Fitting **ARIMA** and **Prophet** on {len(df):,} days of {ticker} history.")
            st.caption("First run takes ~30 seconds. Results are cached - switching tickers is instant after.")
        with col_btn:
            run = st.button("▶  Run Forecast", type="primary", use_container_width=True)

        if run:
            st.session_state["forecast_done"] = True
            st.session_state["forecast_ticker"] = ticker
            st.session_state["forecast_years"]  = years
            st.session_state["forecast_days"]   = forecast_days

        # Re-run if settings changed
        same_settings = (
            st.session_state.get("forecast_ticker") == ticker and
            st.session_state.get("forecast_years")  == years  and
            st.session_state.get("forecast_days")   == forecast_days
        )
        if st.session_state.get("forecast_done") and not same_settings:
            st.session_state["forecast_done"] = False
            st.info("Settings changed - click Run Forecast to update.")

        if st.session_state.get("forecast_done") and same_settings:
            with st.spinner("Fitting Prophet (Bayesian sampling)..."):
                fc, prophet_model = fit_prophet(
                    close.to_json(date_format="iso"), forecast_days
                )
            with st.spinner("Running auto_arima..."):
                arima_order, arima_preds, arima_conf = fit_arima(
                    np.log(close.values).tolist(), forecast_days
                )

            # ARIMA interpretation banner
            if arima_order == (0, 1, 0):
                st.info(
                    f"**auto_arima selected ARIMA(0,1,0)** - pure random walk. "
                    "Past prices carry no predictive signal after differencing. "
                    "Consistent with the Efficient Market Hypothesis."
                )
            else:
                st.info(
                    f"**auto_arima selected ARIMA{arima_order}** - "
                    f"small MA({arima_order[2]}) error-correction term detected. "
                    "Slight short-lived autocorrelation in returns (weak-form EMH)."
                )

            # Main forecast chart
            st.plotly_chart(
                forecast_chart(close, fc, arima_order, arima_preds,
                               arima_conf, forecast_days, ticker),
                use_container_width=True,
            )

            # Price targets
            st.markdown("#### Price Targets")
            prophet_fwd = fc[fc["ds"].values > np.datetime64(close.index[-1], 'us')].head(forecast_days)
            p_target = np.exp(prophet_fwd["yhat"].iloc[-1])
            p_lo     = np.exp(prophet_fwd["yhat_lower"].iloc[-1])
            p_hi     = np.exp(prophet_fwd["yhat_upper"].iloc[-1])
            n        = min(len(arima_preds), forecast_days)
            a_target = np.exp(arima_preds[n - 1])
            a_lo     = np.exp(arima_conf[n - 1, 0])
            a_hi     = np.exp(arima_conf[n - 1, 1])

            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**Prophet**")
                st.metric(
                    f"{forecast_days}-day target",
                    f"₹{p_target:,.1f}",
                    f"{(p_target / m['current'] - 1) * 100:+.1f}% from today",
                )
                st.caption(f"80% CI: ₹{p_lo:,.0f} — ₹{p_hi:,.0f}")

            with t2:
                st.markdown(f"**ARIMA{arima_order}**")
                st.metric(
                    f"{forecast_days}-day target",
                    f"₹{a_target:,.1f}",
                    f"{(a_target / m['current'] - 1) * 100:+.1f}% from today",
                )
                st.caption(f"95% CI: ₹{a_lo:,.0f} — ₹{a_hi:,.0f}")

            st.warning(
                "For educational purposes only. Not financial advice. "
                "Stock prices are inherently unpredictable - "
                "wider confidence intervals at longer horizons reflect genuine uncertainty."
            )

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#888;">
                <p style="font-size:2rem">🔮</p>
                <p>Click <b>Run Forecast</b> above to fit models and generate the {}-day forecast.</p>
            </div>
            """.format(forecast_days), unsafe_allow_html=True)

    # Tab 3: Analysis
    with tab_analysis:
        st.markdown("#### Returns & Volatility")
        st.plotly_chart(returns_chart(close), use_container_width=True)

        with st.expander("📉 Time Series Decomposition  (click to expand)"):
            st.markdown(
                "Separates log(price) into **Trend**, **Weekly Seasonality**, and **Residual**. "
                "Uses `period=5` (5 trading days per week)."
            )
            st.plotly_chart(decomposition_chart(close), use_container_width=True)

        with st.expander("📖 What does the ARIMA order mean?"):
            st.markdown("""
| Order | Meaning |
|-------|---------|
| `ARIMA(0,1,0)` | Pure random walk - EMH in model form. Past prices carry no signal. |
| `ARIMA(0,1,1)` | Random walk + small MA(1) error correction. Weak-form efficiency. |
| `ARIMA(1,1,0)` | Slight momentum in log-returns. Uncommon on large-cap stocks. |
| `ARIMA(p,1,q)` | `d=1` is almost always correct for stock prices (confirmed by ADF test). |

The `d=1` parameter means we difference once: `y'(t) = log(P_t) - log(P_{t-1})` = log-return.
This transforms a non-stationary price series into a stationary returns series.
            """)


if __name__ == "__main__":
    main()