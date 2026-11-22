"""
Stock Market Trend Classification: Decision Tree vs LSTM
---------------------------------------------------------
Script version of notebooks/stock_classification.ipynb.

Predicts whether Apple (AAPL) stock movement is bullish (1) or bearish (0)
from daily price data and technical indicators, then compares a Decision
Tree classifier against an LSTM neural network.

Usage:
    python src/stock_classifier.py                       # uses data/aapl_2014_2023.csv
    python src/stock_classifier.py --data path/to.csv --no-plots
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_7", "rsi_14", "cci_7", "cci_14",
    "sma_50", "ema_50", "sma_100", "ema_100",
    "macd", "bollinger", "TrueRange", "atr_7", "atr_14",
]
SEED = 42


def set_seeds(seed=SEED):
    """Make results repeatable (the LSTM changes run to run without this)."""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    print("Dataset shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    return df


def plot_eda(df, out_dir):
    """Recreate the 7 exploratory charts and save them as PNGs."""
    os.makedirs(out_dir, exist_ok=True)

    def save(name):
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=100)
        plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], color="blue")
    plt.title("Apple Closing Price Over Time (2014–2023)")
    plt.xlabel("Date"); plt.ylabel("Closing Price (USD)")
    save("closing_price_over_time")

    plt.figure(figsize=(10, 5))
    plt.hist(df["TrueRange"], bins=50, color="purple", edgecolor="black")
    plt.title("Distribution of Daily True Range (Volatility)")
    plt.xlabel("True Range"); plt.ylabel("Frequency")
    save("true_range_distribution")

    plt.figure(figsize=(12, 6))
    df.groupby(df["date"].dt.year)["close"].mean().plot(kind="bar", color="orange")
    plt.title("Yearly Average Closing Price")
    plt.xlabel("Year"); plt.ylabel("Average Closing Price (USD)")
    save("yearly_avg_close")

    plt.figure(figsize=(10, 6))
    plt.scatter(df["volume"], df["close"], alpha=0.5, color="green")
    plt.title("Volume vs Closing Price")
    plt.xlabel("Volume"); plt.ylabel("Closing Price (USD)")
    save("volume_vs_close")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["rsi_7"], color="cyan")
    plt.title("Distribution of RSI 7")
    save("rsi7_boxplot")

    plt.figure(figsize=(10, 5))
    sns.kdeplot(df["atr_14"], fill=True, color="red")
    plt.title("Density Plot of ATR 14 (Volatility)")
    save("atr14_density")

    corr_cols = ["open", "high", "low", "close", "volume",
                 "rsi_7", "rsi_14", "cci_7", "cci_14", "macd", "TrueRange"]
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Selected Features")
    save("correlation_heatmap")


def prepare(df):
    """Select features, drop 'neutral' rows, split 80/20 in time order."""
    X = df[FEATURE_COLS]
    y = df["target"].astype(str)

    keep = y != "neutral"
    X, y = X[keep], y[keep].astype(int)
    print("Rows after removing neutral targets:", len(X))

    # shuffle=False keeps time order: train on the past, test on the future
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def plot_confusion(y_true, y_pred, title, cmap, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def train_decision_tree(X_train, y_train, X_test):
    model = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_lstm(X_train, y_train, X_test, epochs=20):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

    # Scale with statistics from the training set only (avoids data leakage)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # LSTM expects 3D input: (samples, time_steps, features)
    X_train_s = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_s = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    model = Sequential([
        Input(shape=(X_train_s.shape[1], X_train_s.shape[2])),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        X_train_s, y_train.astype(np.float32).values,
        epochs=epochs, batch_size=32, validation_split=0.1, verbose=0,
    )
    return (model.predict(X_test_s, verbose=0) > 0.5).astype(int).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/aapl_2014_2023.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    set_seeds()
    df = load_data(args.data)
    if not args.no_plots:
        plot_eda(df, args.out)

    X_train, X_test, y_train, y_test = prepare(df)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    results = {}
    for name, predict, cmap in [
        ("Decision Tree", lambda: train_decision_tree(X_train, y_train, X_test), "Blues"),
        ("LSTM", lambda: train_lstm(X_train, y_train, X_test, args.epochs), "Greens"),
    ]:
        y_pred = predict()
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        if not args.no_plots:
            fname = "confusion_matrix_" + name.lower().replace(" ", "_") + ".png"
            plot_confusion(y_test, y_pred, f"{name} Confusion Matrix", cmap,
                           os.path.join(args.out, fname))

    print("\n=== Comparison ===")
    print(pd.DataFrame(results).T.round(4))


if __name__ == "__main__":
    main()
