# Stock Price Predictor (ML + Streamlit)

A one-day portfolio project that builds an end-to-end stock prediction pipeline: fetch 5-year daily stock data, engineer time-series features, train a regression model, evaluate against a baseline, and serve results through a Streamlit web app.

> Disclaimer: This project is for learning/portfolio purposes only and is **not** financial advice.

## Demo
- **Live Streamlit App:** https://YOUR_STREAMLIT_APP_URL_HERE  
- **Showcase Website (GitHub Pages):** https://YOUR_GITHUB_PAGES_URL_HERE  

## What this project does
1. Downloads historical stock data using `yfinance` (OHLCV).
2. Creates simple, explainable features (momentum, trend, volatility).
3. Trains a **Ridge Regression** model with **time-series cross-validation**.
4. Compares performance to a naive baseline: **tomorrow’s close = today’s close**.
5. Saves outputs (CSV, plots, model, and metrics JSON) into organized subfolders.

## Features used
The model uses a small set of interpretable time-series features:
- `ret_1`: 1-day return (short-term momentum)
- `ret_5`: 5-day return (weekly momentum)
- `ma_5`: 5-day moving average (short trend)
- `ma_20`: 20-day moving average (medium trend)
- `vol_20`: 20-day return standard deviation (volatility)

**Target:** next-day closing price (`y_next_close`).

## Model
- **Ridge Regression (scikit-learn)**  
  Ridge is a regularized linear model that is fast, stable, and a strong baseline for noisy financial time-series.
- **TimeSeriesSplit Cross-Validation**  
  We avoid data leakage by ensuring training data always comes **before** test data in time.

## Project structure
All code lives in `code/`. Generated outputs are written into `data/`.