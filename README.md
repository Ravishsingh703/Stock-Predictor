# Stock Price Predictor (ML + Streamlit)

🚀 **Live App:** https://stock-predictor-32hss6pkwvozushvzneyxd.streamlit.app/

A portfolio project that builds an end-to-end stock prediction pipeline using 5-year daily market data, simple time-series feature engineering, and a Ridge Regression model — wrapped in an interactive Streamlit web app.

> **Disclaimer:** This project is for learning/portfolio purposes only and is **not financial advice**.

---

## What you can do in the app

### 1) Next-day prediction (single company)
- Enter a ticker (e.g., `AAPL`, `TSLA`, `MSFT`)
- View a price chart and a **next-day close prediction**

### 2) Accuracy over the last 30 trading days (simple % metrics)
The app evaluates recent performance by predicting day-by-day and comparing predictions to what actually happened:
- **Up/Down Accuracy (%):** How often the model correctly predicted direction vs the previous day
- **Within X% Accuracy (%):** How often predictions were within a chosen error tolerance (default 2%)
- **MAPE (%):** Average percent error over the evaluation period

You can optionally enable **Recursive mode**, where predictions feed into future inputs (harder and more realistic, errors can compound).

### 3) Backtest 10+ companies (recursive simulation)
Runs a multi-ticker backtest using **5 years of data** and produces:
- A summary table across companies (error + direction metrics)
- Per-company charts: **Actual vs Predicted** and **Absolute Error** over time

---

## Model + features

### Model
- **Ridge Regression (scikit-learn)**  
  A regularized linear regression model that is fast, stable, and easy to explain.

### Features (computed from Close price history)
The model uses interpretable time-series features:
- `ret_1`: 1-day return (short-term momentum)
- `ret_5`: 5-day return (weekly momentum)
- `ma_5`: 5-day moving average (short trend)
- `ma_20`: 20-day moving average (medium trend)
- `vol_20`: 20-day volatility (std of returns)

### Target
- **Next-day closing price**

---

## Project structure

All code lives in `code/`. Generated outputs are saved under `data/`.