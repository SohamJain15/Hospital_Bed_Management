"""
model.py
--------
Linear Regression model that predicts tomorrow's occupied beds.

Features (inputs) used for a day:
    day_number     -> how many days since the first record (captures the trend)
    prev_occupied  -> occupied beds on the PREVIOUS day (yesterday)

Target (output):
    occupied_beds  -> occupied beds on that day

Only yesterday's value is used to predict today, so the model never sees
future information.
"""

import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "occupancy.csv")
FEATURES = ["day_number", "prev_occupied"]
TARGET = "occupied_beds"


def load_data(csv_path=CSV_PATH):
    """Step 1-3: load the CSV, turn dates into numbers, add yesterday's value."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["day_number"] = (df["date"] - df["date"].min()).dt.days
    df["prev_occupied"] = df[TARGET].shift(1)     # yesterday's occupancy
    return df.dropna().reset_index(drop=True)      # first day has no "yesterday"


def train_model(df, test_ratio=0.2):
    """
    Step 4-7: split by date (oldest 80% train, newest 20% test),
    fit the model, predict the test days and measure MAE.
    Finally re-fit on ALL data so the forecast uses the latest trend.
    """
    split = int(len(df) * (1 - test_ratio))
    train, test = df.iloc[:split], df.iloc[split:]

    eval_model = LinearRegression()
    eval_model.fit(train[FEATURES], train[TARGET])
    test_predictions = eval_model.predict(test[FEATURES])
    mae = mean_absolute_error(test[TARGET], test_predictions)

    final_model = LinearRegression()
    final_model.fit(df[FEATURES], df[TARGET])

    return {
        "model": final_model,
        "mae": mae,
        "train": train,
        "test": test.assign(predicted=test_predictions),
        "intercept": final_model.intercept_,
        "coef_day": final_model.coef_[0],
        "coef_prev": final_model.coef_[1],
    }


def forecast(model, df, capacity, days=7):
    """
    Predict the next `days` days one at a time.
    Each prediction becomes the "yesterday" value for the next day.
    Results are rounded and kept between 0 and the hospital's capacity.
    """
    last_date = df["date"].iloc[-1]
    day_number = int(df["day_number"].iloc[-1])
    prev = float(df[TARGET].iloc[-1])
    rows = []
    for step in range(1, days + 1):
        x = pd.DataFrame([[day_number + step, prev]], columns=FEATURES)
        predicted = float(model.predict(x)[0])
        predicted = min(max(predicted, 0), capacity)       # stay within capacity
        rows.append({
            "date": last_date + pd.Timedelta(days=step),
            "predicted_occupied": int(round(predicted)),
        })
        prev = predicted
    result = pd.DataFrame(rows)
    result["predicted_occupied"] = result["predicted_occupied"].clip(upper=capacity)
    result["predicted_available"] = capacity - result["predicted_occupied"]
    result["predicted_pct"] = (result["predicted_occupied"] / capacity * 100).round(1) if capacity else 0.0
    return result


def run_prediction(capacity, csv_path=CSV_PATH, days=7):
    """Convenience function used by the app: load, train and forecast."""
    df = load_data(csv_path)
    results = train_model(df)
    results["data"] = df
    results["forecast"] = forecast(results["model"], df, capacity, days)
    return results


if __name__ == "__main__":
    out = run_prediction(capacity=80)
    print(f"MAE on test days: {out['mae']:.2f} beds")
    print(out["forecast"])
