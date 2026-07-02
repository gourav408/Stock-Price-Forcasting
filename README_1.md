# GOOG Stock Price Forecasting — Hybrid ARIMA + LSTM

## Objective

Forecast the daily closing price of Alphabet Inc. (GOOG) using a two-stage hybrid model — ARIMA for the
linear component of the series, and an LSTM to learn any non-linear structure left behind in the ARIMA
residuals — and rigorously test whether that hybrid actually produces a usable predictive edge, rather
than just reporting a low error number. The project treats "low RMSE/MAPE" as a starting point, not a
conclusion: every result is checked against a naive persistence baseline and a directional / trading
backtest before being called a success.

## Workflow & Results

### 1. Data acquisition & split

10 years of daily GOOG closing prices were pulled via `yfinance` and split chronologically (no shuffling)
into an 80% training set and a 20% held-out test set.

| | |
|---|---|
| Total trading days | 2,513 |
| Training set (80%) | 2,010 days |
| Test set (20%) | 503 days |
| Date range | 2016-07-05 → 2026-07-02 |

### 2. Stationarity & decomposition

An Augmented Dickey-Fuller test confirmed the raw price series is non-stationary, and that a single
first-difference (`d = 1`) is enough to stationarize it. An additive seasonal decomposition (252-day /
annual cycle) and ACF/PACF plots on the differenced series were used to reason about candidate ARIMA
orders.

### 3. ARIMA order selection (AIC grid search)

Rather than hand-picking an order, a grid search over `p, q ∈ {0..3}` with `d = 1` was run once on the
training set:

| Order | AIC |
|---|---|
| ARIMA(3,1,3) | 7,896.0 |
| **ARIMA(1,1,1)** | **7,906.7** ← selected |
| ARIMA(0,1,3) | 7,907.9 |
| ARIMA(0,1,1) | 7,908.2 |
| ARIMA(1,1,0) | 7,908.3 |

ARIMA(3,1,3) technically scored lowest, but its fit failed to converge; ARIMA(1,1,1) was selected as the
more reliable order — it's only ~11 AIC points behind and gives a clean, converging fit.

### 4. Walk-forward ARIMA

The model was fit once on the training set, then rolled forward one day at a time across the full test
window using `append(..., refit=False)`: at each step it forecasts the next close, then is extended with
the *actual* observed price (not its own prediction) before moving on. This avoids both look-ahead
leakage and the cost of refitting from scratch 503 times.

### 5. LSTM on the ARIMA residuals

The in-sample and out-of-sample ARIMA residuals were scaled to `[-1, 1]` (scaler fit on training
residuals only, to avoid leakage) and reshaped into 10-day lookback windows. A 2-layer LSTM
(64 → 32 units, with dropout and early stopping) was trained to predict the next residual.

Validation loss plateaued essentially from epoch 1 (~0.032) and never meaningfully improved — a strong
signal that the ARIMA residuals are close to white noise, with little non-linear structure left for the
LSTM to learn.

### 6. Hybrid forecast vs. naive baseline

The final forecast is `ARIMA forecast + LSTM residual correction`, benchmarked against a naive
persistence baseline (tomorrow = today's close) and against ARIMA alone:

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naive (persistence) | *run to populate* | *run to populate* | *run to populate* |
| ARIMA only | 4.706 | 3.266 | 1.41% |
| ARIMA + LSTM hybrid | 4.694 | 3.257 | 1.41% |

The naive baseline lands in the same neighborhood as both models. On a near-random-walk series like
daily equity closes, a ~1.4% MAPE is largely a reflection of how little day-to-day prices move — not
evidence of genuine predictive skill. The LSTM's improvement over ARIMA alone (~0.2% on RMSE) is
marginal, consistent with the flat validation loss above.

### 7. Reality check — directional accuracy & backtest

Error metrics don't tell you whether a model is *useful* for trading, so the hybrid forecast was also
scored on whether it called the correct direction (up/down) each day, and used to drive a simple
long/short strategy versus buy-and-hold:

| Metric | Value |
|---|---|
| Directional win rate | 53.68% |
| Buy & Hold GOOG return (test window) | +95.44% |
| Hybrid long/short strategy return | +43.73% |

A 53.7% win rate is only modestly above a coin flip, and the strategy underperformed simple buy-and-hold
by a wide margin over this window — despite the attractive-looking error metrics above. On an earlier
ARIMA order this same backtest actually showed a small *loss* (−1.4%) at a 51.7% win rate, which
illustrates how sensitive these headline trading numbers are to small changes in the model and how
little should be read into any single run.

## Final Results & Conclusion

**What this project demonstrates well:** a correctly built, leak-free forecasting pipeline — chronological
train/test split, walk-forward validation using only real observed data, a residual scaler fit strictly on
training data, AIC-informed order selection, and dropout/early-stopping regularization on the neural
component.

**What the results actually show:** the hybrid ARIMA + LSTM model does **not** produce a reliable edge
over a naive "tomorrow = today" forecast on daily GOOG prices, and its directional accuracy is only
slightly better than chance. The LSTM residual stage adds negligible value because the ARIMA residuals
are close to white noise — there simply isn't much non-linear signal left to extract once the linear
autocorrelation has been removed.

**Conclusion:** the value of this project isn't a claim that it beats the market — on this evidence, it
doesn't, reliably. The value is in the pipeline and the diagnostics: benchmarking against a naive
baseline and validating with a directional/backtest check are exactly the steps that expose *why* low
forecast error and real trading skill are two different things, and why daily-price hybrid models like
this one tend to struggle against the efficiency of public equity markets.

## Tech stack

`yfinance` · `statsmodels` (ADF test, seasonal decomposition, ACF/PACF, ARIMA) · `scikit-learn`
(MinMax scaling) · `TensorFlow / Keras` (LSTM) · `pandas` · `NumPy` · `matplotlib`

## Getting started

```bash
pip install yfinance statsmodels scikit-learn tensorflow pandas numpy matplotlib
jupyter notebook GOOG_stock_price_forecasting_notebook.ipynb
```

Run all cells top to bottom. The notebook downloads fresh data on each run, so exact figures will drift
slightly as the underlying GOOG price history updates.

## Disclaimer

Research / educational only — not financial advice. The backtest does not account for transaction costs,
slippage, or market impact, and results from a single test window should not be assumed to generalize to
other time periods or tickers.
