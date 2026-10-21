# Hospital Bed Management System with ML-Based Occupancy Prediction

A college mini-project built with **Python, Streamlit, SQLite, scikit-learn and Plotly**.
It manages hospital beds and patients and uses **Linear Regression** to estimate how many
beds will be occupied tomorrow and over the next 7 days.

> All patient records and historical occupancy data are **simulated / fictional** and exist
> only for demonstration. The forecast is an educational estimate, not a planning tool.

---

## 1. Setup and Run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The browser opens at `http://localhost:8501`. On the first run the app automatically:

1. creates `hospital.db` with the three tables,
2. creates `data/occupancy.csv` if it is missing,
3. loads 80 sample beds, ~60 fictional admitted patients and the 270-day history.

Useful extra commands:

```bash
python generate_data.py        # re-create the simulated CSV (ends yesterday)
python database.py --reset     # delete the database and reload sample data
python model.py                # train the model in the terminal and print MAE + forecast
```

After `generate_data.py`, run `python database.py --reset` so the database history matches the new CSV.
Everything runs offline once the packages are installed.

---

## 2. Project Structure

```
hospital-bed-management/
├── app.py               # Streamlit UI: sidebar navigation + 5 pages
├── database.py          # SQLite tables, bed/patient operations, sample data
├── model.py             # load CSV, train Linear Regression, MAE, 7-day forecast
├── generate_data.py     # creates the simulated occupancy history
├── requirements.txt
├── README.md
├── .streamlit/config.toml   # light blue/white theme
└── data/occupancy.csv   # simulated historical dataset (270 days)
```

`hospital.db` is created automatically when the app first runs.

---

## 3. Architecture

```
            ┌──────────────────────── app.py (Streamlit UI) ────────────────────────┐
            │ Dashboard │ Bed Management │ Patients │ ML Prediction │ Analytics      │
            └─────┬───────────────┬──────────────┬──────────────┬─────────────────┘
                  │               │              │              │
                  ▼               ▼              ▼              ▼
            database.py  ◄──── read / write ────►  hospital.db (SQLite)
            (beds, patients,                       ├── beds
             occupancy_history)                    ├── patients
                                                   └── occupancy_history
                  model.py  ◄── reads ── data/occupancy.csv ◄── generate_data.py
```

* **UI layer (`app.py`)** – only displays data and calls functions. Each page is one function.
* **Data layer (`database.py`)** – every change returns `(success, message)`, so the UI shows a
  green success or red error box.
* **ML layer (`model.py`)** – pure functions: `load_data → train_model → forecast`.

### Database schema

| Table | Columns |
|---|---|
| `beds` | `bed_id` (primary key), `ward`, `status` (Available / Occupied) |
| `patients` | `patient_id`, `name`, `age`, `ward`, `bed_id`, `admission_date`, `discharge_date`, `status` (Admitted / Discharged) |
| `occupancy_history` | `date`, `total_beds`, `occupied_beds`, `admissions`, `discharges` |

### How consistency is guaranteed

* **Admission** runs in one transaction: the bed is set to *Occupied* **only if it is still
  Available**, then the patient row is inserted. If either step fails, both are rolled back.
* **Discharge** runs in one transaction: the patient is marked *Discharged* and the bed becomes
  *Available*.
* A **unique index** (`one_patient_per_bed`) makes it impossible for two admitted patients to
  share a bed, even if the app had a bug.
* A bed that belongs to an admitted patient cannot be manually marked *Available* – the user must
  discharge the patient.
* Bed IDs are primary keys, so duplicates are rejected.
* Dashboard numbers are always **counted from the `beds` table**, never typed in separately.

---

## 4. Pages

| Page | What it does |
|---|---|
| **Dashboard** | Total / occupied / available beds and occupancy %, donut chart, ward bar chart, ward table, next-day prediction and ward capacity warnings |
| **Bed Management** | Filter beds by ward and status, add a bed, mark beds Occupied / Available |
| **Patients** | Admit a patient to an available bed, discharge, search by ID or name |
| **ML Prediction** | Tomorrow's predicted occupied / available beds and %, 7-day forecast, historical vs predicted chart, MAE, model equation |
| **Analytics** | Date-range filter, occupancy trend, weekly admissions vs discharges, ward-wise occupancy |

The **capacity alert threshold** (default 85%) is set in the sidebar and used by the dashboard and
prediction warnings.

---

## 5. The Machine Learning Part

### The dataset
`generate_data.py` simulates 270 days for an 80-bed hospital:

* admissions are random (a little higher on weekdays),
* about 20% of in-patients are discharged each day (≈ 5-day average stay),
* `occupied today = occupied yesterday + admissions − discharges`,
* a slow upward trend is added.

### Linear Regression in simple words
Linear Regression finds the **best straight-line formula** linking inputs to an output.
Our formula is:

```
occupied_today = a + b × day_number + c × occupied_yesterday
```

* `day_number` (0, 1, 2 …) lets the model learn the long-term **trend**.
* `occupied_yesterday` tells it where occupancy currently is.
* `fit()` finds the values of `a`, `b` and `c` that make the line's errors as small as possible
  (least squares). The app shows the learned values on the prediction page.

### Steps in `model.py`
1. Load the CSV with Pandas.
2. Convert dates into `day_number`.
3. Add `prev_occupied` (yesterday's value). Only past data is used – no future leakage.
4. Split **chronologically**: oldest 80% = training, newest 20% = testing.
5. `model.fit(train)`.
6. `model.predict(test)`.
7. Measure **MAE** on the test days.
8. Re-fit on all data and forecast 7 days ahead. Each forecast day uses the previous forecast as
   its "yesterday". Predictions are rounded and clipped between 0 and the total number of beds.

### MAE (Mean Absolute Error)
MAE = the average of |actual − predicted| over the test days.
An MAE of 3.6 means *"on days the model had never seen, it was wrong by about 3–4 beds on
average."* Lower is better, and it is in the same unit as the data (beds), so it is easy to
explain.

### Limitations (say these in the viva)
* The data is simulated, so the accuracy says nothing about a real hospital.
* A straight-line model cannot capture sudden events (epidemics, accidents, holidays).
* Ward-level forecasts are estimated by sharing the hospital forecast according to each ward's
  current share of occupied beds – a simple approximation.

---

## 6. Ten Likely Viva Questions

1. **What does your project do?**
   It tracks hospital beds and patients, shows live availability, and predicts tomorrow's and the
   next week's bed occupancy using Linear Regression.

2. **Why Streamlit?**
   It builds a web UI with only Python – no HTML/JavaScript – so we could focus on logic and ML.

3. **Why SQLite?**
   It is a file-based database that needs no server, is built into Python and keeps data after the
   app restarts.

4. **How do you make sure two patients never get the same bed?**
   The bed is only updated if its status is still *Available*, inside a transaction, and a unique
   index on `bed_id` for admitted patients blocks duplicates at the database level.

5. **What is a transaction?**
   A group of database steps that either all succeed or all fail. Admission updates the bed and
   inserts the patient together, so the data can never be half-updated.

6. **Why Linear Regression?**
   It is simple, fast and explainable – we can show the exact formula. For a trend-based estimate
   on a small dataset it is a reasonable first model.

7. **What are your features and target?**
   Features: `day_number` and yesterday's occupied beds. Target: today's occupied beds.

8. **Why not split the data randomly?**
   It is time-series data. A random split would let the model train on future days and test on
   past days, which is cheating. We train on older dates and test on newer dates.

9. **What is MAE and what does your value mean?**
   The average absolute difference between actual and predicted values. Ours (shown in the app)
   means the prediction is off by that many beds on average on unseen days.

10. **How could you improve the project?**
    Use real anonymised data, add features such as weekday and holidays, compare with other models
    (e.g. Random Forest or ARIMA), predict per ward, and add user login.

---

## 7. Project Report Outline

1. Title page, certificate, acknowledgement
2. Abstract
3. Introduction – problem of bed shortages, motivation, objectives
4. Literature review / existing systems (brief)
5. Requirements – hardware, software, functional and non-functional requirements
6. System design – architecture diagram, database schema, flow of admission/discharge, page flow
7. Dataset – how the simulated data was generated and why it is labelled as simulated
8. Methodology – Linear Regression, features, chronological split, MAE
9. Implementation – module-wise explanation (`app.py`, `database.py`, `model.py`, `generate_data.py`)
10. Results – screenshots of each page, MAE value, forecast charts
11. Testing – test cases (duplicate bed ID, invalid status change, admit to full ward, discharge, persistence)
12. Limitations and future scope
13. Conclusion
14. References

## 8. Presentation Slide Outline (10–12 slides)

1. Title, team, guide
2. Problem statement – why hospitals need live bed tracking and forecasting
3. Objectives
4. Technology stack
5. System architecture
6. Database design
7. Key features – dashboard, bed management, patients
8. Dataset (simulated) and ML approach
9. Linear Regression + MAE explained with our formula
10. Results – screenshots and forecast chart
11. Limitations and future scope
12. Conclusion and Q&A
