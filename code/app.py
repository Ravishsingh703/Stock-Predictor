import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# --- Paths (repo-root aware) ---
APP_PATH = Path(__file__).resolve()
ROOT_DIR = APP_PATH.parent.parent if APP_PATH.parent.name == "code" else APP_PATH.parent

DATA_DIR = ROOT_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
GRAPHS_DIR = DATA_DIR / "graphs"
LEARNING_DIR = DATA_DIR / "learning"
MODELS_DIR = DATA_DIR / "models"

for d in [CSV_DIR, GRAPHS_DIR, LEARNING_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def fetch_data(ticker="AAPL", period="5y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # target: next day close
    df["y_next_close"] = df["Close"].shift(-1)

    return df.dropna()


def make_xy(df: pd.DataFrame):
    features = ["ret_1", "ret_5", "ma_5", "ma_20", "vol_20"]
    X = df[features].values
    y = df["y_next_close"].values
    return X, y, features


def baseline_predict_next_close(df_feat: pd.DataFrame):
    # baseline: predict tomorrow close = today close
    y_true = df_feat["y_next_close"].values
    y_pred = df_feat["Close"].values
    return y_true, y_pred


def cv_mae_ridge(X, y, alpha=1.0, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    maes = []
    model = Ridge(alpha=alpha)

    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))

    model.fit(X, y)  # final fit on all data
    return float(np.mean(maes)), model


st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Prediction (Portfolio Demo)")
st.caption("Educational project — not financial advice.")


ticker = st.text_input("Ticker", value="TSLA").strip().upper()
period = st.selectbox("Period", ["1y", "2y", "5y", "10y"], index=2)
alpha = st.slider("Ridge alpha (regularization)", 0.1, 10.0, 1.0, 0.1)

run = st.button("Run prediction")

if run:
    try:
        # 1) data
        df = fetch_data(ticker, period)

        # Save raw CSV
        raw_csv_path = CSV_DIR / f"raw_{ticker}_{period}.csv"
        df.to_csv(raw_csv_path)

        # 2) features
        df_feat = add_features(df)
        feat_csv_path = LEARNING_DIR / f"features_{ticker}_{period}.csv"
        df_feat.to_csv(feat_csv_path)

        # 3) baseline
        y_true_b, y_pred_b = baseline_predict_next_close(df_feat)
        baseline_mae = mean_absolute_error(y_true_b, y_pred_b)

        # 4) ridge model
        X, y, feat_names = make_xy(df_feat)
        ridge_mae, model = cv_mae_ridge(X, y, alpha=alpha, splits=5)

        # 5) predict next day (from last row features)
        pred_next = float(model.predict(X[-1].reshape(1, -1))[0])
        last_close = float(df_feat["Close"].iloc[-1])

        # 6) save model
        model_path = MODELS_DIR / f"ridge_{ticker}_{period}.joblib"
        joblib.dump(model, model_path)

        # 7) chart (saved + shown)
        fig = plt.figure()
        plt.plot(df_feat.index, df_feat["Close"])
        plt.title(f"{ticker} Close ({period}) + Next Close Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.tight_layout()

        graph_path = GRAPHS_DIR / f"{ticker}_{period}_prediction_plot.png"
        plt.savefig(graph_path, dpi=200, bbox_inches="tight")

        # 8) save metrics json
        metrics = {
            "ticker": ticker,
            "period": period,
            "run_time_utc": datetime.utcnow().isoformat() + "Z",
            "features": feat_names,
            "baseline_mae": float(baseline_mae),
            "ridge_cv_mae": float(ridge_mae),
            "last_close": float(last_close),
            "predicted_next_close": float(pred_next),
            "raw_csv": str(raw_csv_path),
            "features_csv": str(feat_csv_path),
            "model_path": str(model_path),
            "graph_path": str(graph_path),
        }
        metrics_path = LEARNING_DIR / f"metrics_{ticker}_{period}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # --- UI ---
        col1, col2 = st.columns(2)
        col1.metric("Last Close", f"{last_close:.2f}")
        col2.metric("Predicted Next Close", f"{pred_next:.2f}")

        col3, col4 = st.columns(2)
        col3.metric("Baseline MAE", f"{baseline_mae:.4f}")
        col4.metric("Ridge CV MAE", f"{ridge_mae:.4f}")

        st.pyplot(fig)

        st.write("Features used:", feat_names)
        st.success(f"Saved graph + metrics: {graph_path.name}, {metrics_path.name}")

    except Exception as e:
        st.error(str(e))