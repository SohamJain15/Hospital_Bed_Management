"""
generate_data.py
----------------
Creates SIMULATED historical occupancy data for the demo.

IMPORTANT: This is NOT real hospital data. It is generated with random
numbers so the project has something to train the ML model on.

How the simulation works (easy to explain):
  * The hospital has 80 beds.
  * Every day some patients are admitted (random, a bit higher on weekdays).
  * Every day roughly 20% of the patients already in beds are discharged
    (that means an average stay of about 5 days).
  * Occupied today = Occupied yesterday + Admissions - Discharges
  * A slow upward trend is added so Linear Regression has a trend to learn.

Run:  python generate_data.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

TOTAL_BEDS = 80
NUM_DAYS = 270
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "occupancy.csv")


def generate_occupancy_data(num_days=NUM_DAYS, total_beds=TOTAL_BEDS, seed=5):
    """Return a DataFrame of simulated daily occupancy ending yesterday."""
    rng = np.random.default_rng(seed)

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=num_days - 1)

    discharge_rate = 0.20          # ~20% of in-patients leave each day
    occupied = 50                  # starting occupancy
    rows = []

    for i in range(num_days):
        current_date = start_date + timedelta(days=i)

        # Target occupancy slowly rises from about 48 to about 60 beds
        target = 48 + 12 * (i / num_days)

        # Weekdays get slightly more admissions than weekends
        weekday_boost = 1.5 if current_date.weekday() < 5 else -2.5
        expected_admissions = max(1, target * discharge_rate + weekday_boost)
        admissions = int(rng.poisson(expected_admissions))

        discharges = int(rng.binomial(occupied, discharge_rate))

        new_occupied = occupied + admissions - discharges

        # Keep occupancy within the hospital's capacity
        if new_occupied > total_beds:
            admissions -= new_occupied - total_beds   # extra patients were redirected
            new_occupied = total_beds
        new_occupied = max(new_occupied, 0)

        rows.append({
            "date": current_date.isoformat(),
            "total_beds": total_beds,
            "occupied_beds": new_occupied,
            "admissions": admissions,
            "discharges": discharges,
        })
        occupied = new_occupied

    return pd.DataFrame(rows)


def save_dataset(path=CSV_PATH):
    """Generate the dataset and save it as a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_occupancy_data()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = save_dataset()
    print(f"Saved {len(data)} days of SIMULATED data to {CSV_PATH}")
    print(data.tail())
