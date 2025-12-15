import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# --- Paths (repo-root aware) ---
SCRIPT_PATH = Path(__file__).resolve()
# If this file lives in /code, repo root is one level up
ROOT_DIR = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "code" else SCRIPT_PATH.parent

DATA_DIR = ROOT_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
GRAPHS_DIR = DATA_DIR / "graphs"
LEARNING_DIR = DATA_DIR / "learning"
MODELS_DIR = DATA_DIR / "models"

for d in [CSV_DIR, GRAPHS_DIR, LEARNING_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    ticker = input("Enter ticker (e.g., AAPL, MSFT, TSLA): ").strip().upper()
    if not ticker:
        ticker = "AAPL"

    period = input("Enter period (default 5y): ").strip().lower()
    if not period:
        period = "5y"

    # 1) data
    df = fetch_data(ticker, period)
    raw_csv_path = CSV_DIR / f"raw_{ticker}_{period}.csv"
    df.to_csv(raw_csv_path)
    print(f"Saved raw CSV: {raw_csv_path}")

    # 2) features
    df_feat = add_features(df)
    feat_csv_path = LEARNING_DIR / f"features_{ticker}_{period}.csv"
    df_feat.to_csv(feat_csv_path)
    print(f"Saved feature dataset: {feat_csv_path}")

    # 3) baseline
    y_true_b, y_pred_b = baseline_predict_next_close(df_feat)
    baseline_mae = mean_absolute_error(y_true_b, y_pred_b)

    # 4) ridge model
    X, y, feat_names = make_xy(df_feat)
    ridge_mae, model = cv_mae_ridge(X, y, alpha=1.0, splits=5)

    # 5) predict next day (from last row features)
    pred_next = float(model.predict(X[-1].reshape(1, -1))[0])
    last_close = float(df_feat["Close"].iloc[-1])

    # 6) save model
    model_path = MODELS_DIR / f"ridge_{ticker}_{period}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

    print("\n=== Results ===")
    print("Features:", feat_names)
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Ridge CV MAE: {ridge_mae:.4f}")
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted next close: {pred_next:.2f}")
    
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

    print(f"Saved metrics: {metrics_path}")

    # 7) chart (saved to graphs/)
    plt.figure()
    plt.plot(df_feat.index, df_feat["Close"])
    plt.title(f"{ticker} Close ({period}) + Next Close Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()

    graph_path = GRAPHS_DIR / f"{ticker}_{period}_prediction_plot.png"
    plt.savefig(graph_path, dpi=200, bbox_inches="tight")
    print(f"Saved graph: {graph_path}")

    plt.show()