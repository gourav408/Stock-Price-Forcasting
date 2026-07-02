# GOOG Stock Price Forecasting — Hybrid ARIMA + LSTM

A time-series forecasting pipeline that combines a classical **ARIMA** model with an **LSTM** neural
network to forecast the daily closing price of Alphabet Inc. (GOOG). ARIMA captures the linear
structure of the series through a rolling walk-forward forecast; the LSTM is layered on top to learn
any non-linear structure left behind in the ARIMA residuals.

The project is built around a simple principle: **a low forecast error does not automatically mean a
model has predictive skill.** Daily equity prices are close to a random walk, so naive metrics like
RMSE or MAPE can look impressive while hiding a model that's really just predicting "no change." This
notebook is designed to surface that gap rather than paper over it — every result is benchmarked
against a naive persistence baseline and checked against a directional-accuracy / trading backtest,
not just reported as a standalone error number.

## What's inside

- **Data** — 10 years of daily GOOG OHLCV data pulled via `yfinance`, split chronologically 80/20 into
  train and test sets (no shuffling, no leakage).
- **Stationarity & decomposition** — ADF testing to confirm the differencing order, plus an additive
  seasonal decomposition and ACF/PACF plots to reason about ARIMA order candidates.
- **ARIMA order selection** — a small AIC grid search over `(p, d, q)` rather than a hand-picked order.
- **Walk-forward ARIMA** — a genuine one-step-ahead rolling forecast: the model is fit once and
  extended with `append(..., refit=False)` as each new actual price is observed, avoiding both
  look-ahead leakage and the cost of a full refit every day.
- **LSTM residual model** — a 2-layer LSTM (with dropout and early stopping) trained to predict the
  ARIMA model's residuals, using a scaler fit only on training data.
- **Hybrid forecast** — `ARIMA forecast + LSTM residual correction`, evaluated against:
  - a **naive persistence baseline** (tomorrow = today),
  - **ARIMA alone**,
  - the **full hybrid**.
- **Reality check** — directional (up/down) accuracy and a simple long/short backtest compared to
  buy-and-hold, because low price error and real trading edge are not the same thing.

## Key finding

The hybrid model achieves a low headline MAPE (~1.4%), but so does the naive persistence baseline —
because daily price *levels* are highly autocorrelated. The RMSE/MAE improvement from the LSTM stage
over ARIMA alone is marginal, and its validation loss is flat almost immediately, indicating there is
little non-linear structure left in the ARIMA residuals for it to learn. The directional win rate sits
just above chance. The honest takeaway isn't "this model beats the market" — it's a correctly built,
leak-free forecasting pipeline whose diagnostics reveal *why* a hybrid ARIMA+LSTM struggles to beat a
random walk on daily equity prices.

## Tech stack

- `yfinance` — data acquisition
- `statsmodels` — ADF test, seasonal decomposition, ACF/PACF, ARIMA
- `scikit-learn` — MinMax scaling
- `TensorFlow / Keras` — LSTM residual model
- `pandas`, `NumPy`, `matplotlib` — data handling and visualization

## Getting started

```bash
pip install yfinance statsmodels scikit-learn tensorflow pandas numpy matplotlib
jupyter notebook GOOG_stock_price_forecasting_notebook.ipynb
```

Run all cells top to bottom. The notebook downloads fresh data on each run, so results will drift
slightly over time as the underlying GOOG price history updates.

## Disclaimer

This project is for research and educational purposes only and is not financial advice. The backtest
does not account for transaction costs, slippage, or market impact, and results from a single test
window should not be assumed to generalize to other time periods or tickers.
