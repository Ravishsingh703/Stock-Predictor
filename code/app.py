import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ----------------------------
# Paths (repo-root aware)
# ----------------------------
APP_PATH = Path(__file__).resolve()
ROOT_DIR = APP_PATH.parent.parent if APP_PATH.parent.name == "code" else APP_PATH.parent

DATA_DIR = ROOT_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
GRAPHS_DIR = DATA_DIR / "graphs"
LEARNING_DIR = DATA_DIR / "learning"
MODELS_DIR = DATA_DIR / "models"

for d in [CSV_DIR, GRAPHS_DIR, LEARNING_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Data + features (Close-only)
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_close_series(ticker: str, period: str = "5y") -> pd.Series:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.index.name = "Date"
    close.name = "Close"
    return close


def features_from_history(close_history: np.ndarray) -> np.ndarray:
    """
    One feature row from the LAST values of a close series.
    Needs >= 21 points.

    Features:
      - ret_1, ret_5, ma_5, ma_20, vol_20
    """
    if len(close_history) < 21:
        raise ValueError("Need at least 21 days of history for features.")

    c = close_history.astype(float)
    ret_1 = (c[-1] / c[-2]) - 1.0
    ret_5 = (c[-1] / c[-6]) - 1.0
    ma_5 = float(np.mean(c[-5:]))
    ma_20 = float(np.mean(c[-20:]))

    # last 20 daily returns
    r = (c[-20:] / c[-21:-1]) - 1.0
    vol_20 = float(np.std(r))

    return np.array([ret_1, ret_5, ma_5, ma_20, vol_20], dtype=float).reshape(1, -1)


def make_training_set(close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised dataset from close series:
      X(t) from history up to t
      y(t) = close(t+1)
    """
    values = close.values.astype(float)
    X_list, y_list = [], []

    for t in range(20, len(values) - 1):
        hist = values[: t + 1]
        X_list.append(features_from_history(hist).flatten())
        y_list.append(values[t + 1])

    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)


def direction_accuracy(actual: np.ndarray, pred: np.ndarray, last_known: np.ndarray) -> float:
    true_dir = np.sign(actual - last_known)
    pred_dir = np.sign(pred - last_known)
    return float((true_dir == pred_dir).mean())


# ----------------------------
# Backtests
# ----------------------------
def recursive_backtest(close: pd.Series, test_days: int = 252, alpha: float = 1.0):
    """
    Train on older data, then simulate the last `test_days` days recursively.
    Each day we append our prediction into history (not the actual).
    """
    if len(close) < 21 + test_days + 50:
        raise ValueError("Not enough data for this backtest length.")

    split_idx = len(close) - test_days - 1
    train_close = close.iloc[: split_idx + 1]
    test_close = close.iloc[split_idx + 1 :]

    X_train, y_train = make_training_set(train_close)
    model = Ridge(alpha=alpha).fit(X_train, y_train)

    history = train_close.values.astype(float).tolist()
    rows = []

    for dt, actual_next in test_close.items():
        last_known = float(history[-1])
        x = features_from_history(np.array(history, dtype=float))
        pred_next = float(model.predict(x)[0])

        rows.append({
            "date": dt,
            "last_known_close": last_known,
            "actual_close": float(actual_next),
            "pred_close": pred_next,
            "abs_error": abs(float(actual_next) - pred_next),
        })

        history.append(pred_next)  # recursive feed

    bt = pd.DataFrame(rows).set_index("date")

    mae = float(mean_absolute_error(bt["actual_close"], bt["pred_close"]))
    mape = float(
        (np.abs((bt["actual_close"] - bt["pred_close"]) / bt["actual_close"]))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean() * 100.0
    )
    dir_acc = direction_accuracy(
        actual=bt["actual_close"].values,
        pred=bt["pred_close"].values,
        last_known=bt["last_known_close"].values,
    )

    metrics = {
        "mae": mae,
        "mape_pct": mape,
        "direction_accuracy": dir_acc,  # 0..1
        "test_days": int(test_days),
        "alpha": float(alpha),
        "recursive": True,
    }
    return bt, metrics, model


def backtest_last_n_days(
    close: pd.Series,
    n_days: int = 30,
    alpha: float = 1.0,
    tol_pct: float = 0.02,
    recursive: bool = False,
):
    """
    Train on older data, predict last n_days one-step ahead.
    - recursive=False: feed ACTUALS back in (cleaner evaluation)
    - recursive=True: feed PREDICTIONS back in (harder, errors compound)

    Returns bt_df and metrics with % accuracies.
    """
    if len(close) < 21 + n_days + 50:
        raise ValueError("Not enough data to run this backtest.")

    split_idx = len(close) - n_days - 1
    train_close = close.iloc[: split_idx + 1]
    test_close = close.iloc[split_idx + 1 :]

    X_train, y_train = make_training_set(train_close)
    model = Ridge(alpha=alpha).fit(X_train, y_train)

    history = train_close.values.astype(float).tolist()
    rows = []

    for dt, actual_next in test_close.items():
        last_known = float(history[-1])
        x = features_from_history(np.array(history, dtype=float))
        pred_next = float(model.predict(x)[0])

        abs_err = abs(float(actual_next) - pred_next)
        pct_err = abs_err / float(actual_next) if float(actual_next) != 0 else np.nan

        rows.append({
            "date": dt,
            "last_known_close": last_known,
            "actual_close": float(actual_next),
            "pred_close": pred_next,
            "abs_error": abs_err,
            "pct_error": pct_err,
            "correct_direction": int(np.sign(float(actual_next) - last_known) == np.sign(pred_next - last_known)),
            "within_tol": int((pct_err <= tol_pct) if not np.isnan(pct_err) else 0),
        })

        history.append(pred_next if recursive else float(actual_next))

    bt = pd.DataFrame(rows).set_index("date")

    mae = float(mean_absolute_error(bt["actual_close"], bt["pred_close"]))
    mape = float(bt["pct_error"].replace([np.inf, -np.inf], np.nan).dropna().mean() * 100.0)

    direction_acc_pct = float(bt["correct_direction"].mean() * 100.0)
    within_tol_acc_pct = float(bt["within_tol"].mean() * 100.0)

    metrics = {
        "mae": mae,
        "mape_pct": mape,
        "direction_accuracy_pct": direction_acc_pct,
        "within_tol_accuracy_pct": within_tol_acc_pct,
        "n_days": int(n_days),
        "tol_pct": float(tol_pct),
        "alpha": float(alpha),
        "recursive": bool(recursive),
    }
    return bt, metrics


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Predictor (Portfolio Demo)")
st.caption("Educational project — not financial advice.")

tabs = st.tabs(["Simple (one company)", "Backtest 10+ companies"])

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "JNJ", "XOM"]

with tabs[0]:
    st.subheader("Simple demo")
    st.write("Pick a ticker to see a next-day prediction and a simple accuracy report for the last 30 trading days.")

    c1, c2 = st.columns(2)
    ticker = c1.text_input("Ticker", value="TSLA").strip().upper()
    period = c2.selectbox("Period for chart", ["1y", "2y", "5y", "10y"], index=2)

    alpha = st.slider("Model strength (Ridge alpha)", 0.1, 10.0, 1.0, 0.1)

    # --- Next-day prediction ---
    if st.button("Run next-day prediction", key="btn_single_pred"):
        close = fetch_close_series(ticker, period)

        close_csv = CSV_DIR / f"close_{ticker}_{period}.csv"
        close.to_frame().to_csv(close_csv)

        X, y = make_training_set(close)
        model = Ridge(alpha=alpha).fit(X, y)

        last_close = float(close.values[-1])
        pred_next = float(model.predict(features_from_history(close.values.astype(float)))[0])

        model_path = MODELS_DIR / f"ridge_{ticker}_{period}.joblib"
        joblib.dump(model, model_path)

        fig = plt.figure()
        plt.plot(close.index, close.values)
        plt.title(f"{ticker} Close ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.tight_layout()

        graph_path = GRAPHS_DIR / f"{ticker}_{period}_close.png"
        plt.savefig(graph_path, dpi=200, bbox_inches="tight")

        a, b = st.columns(2)
        a.metric("Last close", f"{last_close:.2f}")
        b.metric("Predicted next close", f"{pred_next:.2f}")

        st.pyplot(fig)
        st.success("Saved CSV/model/graph into the data/ folder.")

    st.markdown("---")
    st.subheader("✅ Accuracy over the last 30 trading days")
    st.write("This checks how well the model did recently by comparing predictions to what actually happened.")

    d1, d2, d3 = st.columns(3)
    n_days = d1.slider("Days", 10, 90, 30, 5)
    tol_pct = 0.02
    recursive_mode = False
    with st.expander("Advanced settings"):
        tol_pct = st.slider("Within (%)", 1, 10, 2, 1) / 100.0
        recursive_mode = st.checkbox("Recursive (harder)", value=False)

    if st.button("Run last-days accuracy", key="btn_last_days"):
        close5y = fetch_close_series(ticker, "5y")
        bt, m = backtest_last_n_days(
            close5y,
            n_days=n_days,
            alpha=alpha,
            tol_pct=tol_pct,
            recursive=recursive_mode,
        )

        bt_path = LEARNING_DIR / f"last_{n_days}d_backtest_{ticker}_5y.csv"
        bt.to_csv(bt_path)

        metrics_payload = {
            "ticker": ticker,
            "period": "5y",
            "run_time_utc": datetime.utcnow().isoformat() + "Z",
            **m,
            "backtest_csv": str(bt_path),
        }
        metrics_path = LEARNING_DIR / f"last_{n_days}d_metrics_{ticker}_5y.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Up/Down Accuracy", f"{m['direction_accuracy_pct']:.1f}%")
        m2.metric(f"Within {int(tol_pct*100)}% Accuracy", f"{m['within_tol_accuracy_pct']:.1f}%")
        m3.metric("MAPE (avg % error)", f"{m['mape_pct']:.2f}%")

        fig = plt.figure()
        plt.plot(bt.index, bt["actual_close"], label="Actual")
        plt.plot(bt.index, bt["pred_close"], label="Predicted")
        plt.title(f"{ticker} — last {n_days} days (Actual vs Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(f"Saved: {bt_path.name} and {metrics_path.name}")
        st.dataframe(bt.tail(min(30, len(bt))), use_container_width=True)


with tabs[1]:
    st.subheader("Backtest 10+ companies (recursive)")
    st.write(
        "This simulates real usage: we train on older data and then predict day-by-day. "
        "Each day’s next input uses the **previous prediction**, so errors can compound."
    )

    tickers = st.multiselect("Companies", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    test_days = st.slider("Test window (days)", 30, 504, 252, 21)
    with st.expander("Advanced settings"):
        alpha_bt = st.slider("Model strength (Ridge alpha)", 0.1, 10.0, 1.0, 0.1)

    if st.button("Run 10+ company backtest", key="btn_multi_bt"):
        summary_rows = []
        per_ticker_results = {}

        prog = st.progress(0)
        for i, t in enumerate(tickers):
            prog.progress((i + 1) / max(1, len(tickers)))
            try:
                close = fetch_close_series(t, "5y")
                bt_df, m, _ = recursive_backtest(close, test_days=test_days, alpha=alpha_bt)

                bt_path = LEARNING_DIR / f"backtest_{t}_5y.csv"
                bt_df.to_csv(bt_path)

                metrics_payload = {
                    "ticker": t,
                    "period": "5y",
                    "run_time_utc": datetime.utcnow().isoformat() + "Z",
                    **m,
                    "backtest_csv": str(bt_path),
                }
                metrics_path = LEARNING_DIR / f"backtest_metrics_{t}_5y.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics_payload, f, indent=2)

                summary_rows.append({
                    "ticker": t,
                    "avg_error_price": m["mae"],
                    "avg_error_pct": m["mape_pct"],
                    "correct_updown_pct": m["direction_accuracy"] * 100.0,
                })
                per_ticker_results[t] = bt_df

            except Exception as e:
                st.warning(f"{t}: {e}")

        summary = pd.DataFrame(summary_rows).sort_values("avg_error_price")
        summary_path = LEARNING_DIR / "summary_backtest_5y.csv"
        summary.to_csv(summary_path, index=False)

        st.success(f"Done. Summary saved as {summary_path.name}")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### View one company")

        if not per_ticker_results:
            st.error("No backtest results to display. Try fewer tickers or a shorter test window.")
        else:
            pick = st.selectbox("Ticker", list(per_ticker_results.keys()))
            bt = per_ticker_results[pick]

        fig1 = plt.figure()
        plt.plot(bt.index, bt["actual_close"], label="Actual")
        plt.plot(bt.index, bt["pred_close"], label="Predicted")
        plt.title(f"{pick} — recursive backtest ({test_days} days)")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(bt.index, bt["abs_error"])
        plt.title(f"{pick} — absolute error over time")
        plt.xlabel("Date")
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        st.pyplot(fig2)