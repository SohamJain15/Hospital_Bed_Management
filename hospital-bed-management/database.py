"""
database.py
-----------
All SQLite work lives here: creating tables, beds, patients and history.

Every function that changes data returns (success, message) so the UI can
show a clear success or error message.

Run `python database.py --reset` to delete everything and reload the
sample data.
"""

import os
import random
import re
import sqlite3
import sys
from datetime import date, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hospital.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "occupancy.csv")

# Ward name -> (number of sample beds, bed ID prefix)
WARDS = {
    "General Ward": (40, "GW"),
    "ICU": (12, "ICU"),
    "Emergency": (15, "ER"),
    "Private Ward": (13, "PW"),
}
WARD_NAMES = list(WARDS.keys())


# ---------------------------------------------------------------- setup
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    """Create the three tables if they do not exist yet."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS beds (
                bed_id  TEXT PRIMARY KEY,
                ward    TEXT NOT NULL,
                status  TEXT NOT NULL CHECK (status IN ('Available', 'Occupied'))
            );

            CREATE TABLE IF NOT EXISTS patients (
                patient_id      TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                age             INTEGER NOT NULL CHECK (age BETWEEN 0 AND 120),
                ward            TEXT NOT NULL,
                bed_id          TEXT NOT NULL REFERENCES beds(bed_id),
                admission_date  TEXT NOT NULL,
                discharge_date  TEXT,
                status          TEXT NOT NULL DEFAULT 'Admitted'
                                CHECK (status IN ('Admitted', 'Discharged'))
            );

            -- A bed can belong to only ONE currently admitted patient
            CREATE UNIQUE INDEX IF NOT EXISTS one_patient_per_bed
                ON patients(bed_id) WHERE status = 'Admitted';

            CREATE TABLE IF NOT EXISTS occupancy_history (
                date           TEXT PRIMARY KEY,
                total_beds     INTEGER NOT NULL,
                occupied_beds  INTEGER NOT NULL,
                admissions     INTEGER NOT NULL,
                discharges     INTEGER NOT NULL
            );
        """)


# ---------------------------------------------------------------- beds
def get_beds(ward=None, status=None):
    """Return beds as a DataFrame, optionally filtered by ward and status."""
    query = """
        SELECT b.bed_id AS "Bed ID", b.ward AS "Ward", b.status AS "Status",
               COALESCE(p.patient_id, '-') AS "Patient ID",
               COALESCE(p.name, '-') AS "Patient Name"
        FROM beds b
        LEFT JOIN patients p ON p.bed_id = b.bed_id AND p.status = 'Admitted'
        WHERE 1 = 1
    """
    params = []
    if ward and ward != "All":
        query += " AND b.ward = ?"
        params.append(ward)
    if status and status != "All":
        query += " AND b.status = ?"
        params.append(status)
    query += " ORDER BY b.ward, b.bed_id"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_bed_ids(status, ward=None):
    """List of bed IDs with the given status (optionally in one ward)."""
    query = "SELECT bed_id FROM beds WHERE status = ?"
    params = [status]
    if ward:
        query += " AND ward = ?"
        params.append(ward)
    query += " ORDER BY bed_id"
    with get_connection() as conn:
        return [row[0] for row in conn.execute(query, params)]


def add_bed(bed_id, ward):
    """Add a new (available) bed. Rejects empty, badly formatted or duplicate IDs."""
    bed_id = bed_id.strip().upper()
    if not bed_id:
        return False, "Bed ID cannot be empty."
    if not re.fullmatch(r"[A-Z0-9-]{2,15}", bed_id):
        return False, "Bed ID may contain only letters, numbers and '-' (2–15 characters)."
    if ward not in WARD_NAMES:
        return False, "Please choose a valid ward."
    try:
        with get_connection() as conn:
            conn.execute("INSERT INTO beds (bed_id, ward, status) VALUES (?, ?, 'Available')",
                         (bed_id, ward))
        return True, f"Bed {bed_id} added to {ward}."
    except sqlite3.IntegrityError:
        return False, f"Bed ID {bed_id} already exists. Please use a different ID."


def update_bed_status(bed_id, new_status):
    """Change a bed between Available and Occupied, with validation."""
    if new_status not in ("Available", "Occupied"):
        return False, "Invalid status."
    with get_connection() as conn:
        row = conn.execute("SELECT status FROM beds WHERE bed_id = ?", (bed_id,)).fetchone()
        if row is None:
            return False, f"Bed {bed_id} does not exist."
        if row[0] == new_status:
            return False, f"Bed {bed_id} is already {new_status}."
        if new_status == "Available":
            patient = conn.execute(
                "SELECT patient_id, name FROM patients WHERE bed_id = ? AND status = 'Admitted'",
                (bed_id,)).fetchone()
            if patient:
                return False, (f"Bed {bed_id} is assigned to {patient[1]} ({patient[0]}). "
                               "Discharge the patient from the Patients page instead.")
        conn.execute("UPDATE beds SET status = ? WHERE bed_id = ?", (new_status, bed_id))
    return True, f"Bed {bed_id} marked as {new_status}."


def get_ward_summary():
    """Total / occupied / available beds and occupancy % for each ward."""
    query = """
        SELECT ward AS "Ward",
               COUNT(*) AS "Total Beds",
               SUM(CASE WHEN status = 'Occupied' THEN 1 ELSE 0 END) AS "Occupied",
               SUM(CASE WHEN status = 'Available' THEN 1 ELSE 0 END) AS "Available"
        FROM beds GROUP BY ward
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)
    # Keep wards in a fixed, sensible order
    df["order"] = df["Ward"].apply(lambda w: WARD_NAMES.index(w) if w in WARD_NAMES else 99)
    df = df.sort_values("order").drop(columns="order").reset_index(drop=True)
    df["Occupancy %"] = (df["Occupied"] / df["Total Beds"] * 100).round(1)
    return df


def get_overall_stats():
    """Hospital-wide numbers, calculated from the actual bed records."""
    with get_connection() as conn:
        total, occupied = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = 'Occupied'), 0) FROM beds").fetchone()
    available = total - occupied
    percent = round(occupied / total * 100, 1) if total else 0.0
    return {"total": total, "occupied": occupied, "available": available, "percent": percent}


# ---------------------------------------------------------------- patients
def next_patient_id():
    """Generate the next ID like P0001, P0002, ..."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(CAST(SUBSTR(patient_id, 2) AS INTEGER)) FROM patients").fetchone()
    return f"P{(row[0] or 0) + 1:04d}"


def add_patient(name, age, ward, bed_id, admission_date):
    """Admit a patient and mark their bed Occupied in ONE transaction."""
    name = name.strip()
    if not name:
        return False, "Patient name cannot be empty."
    if not (0 <= int(age) <= 120):
        return False, "Age must be between 0 and 120."
    if not bed_id:
        return False, f"No available bed selected in {ward}."
    if admission_date > date.today():
        return False, "Admission date cannot be in the future."

    patient_id = next_patient_id()
    conn = get_connection()
    try:
        with conn:  # commits if everything succeeds, rolls back on any error
            # Only succeeds if the bed is still Available in the chosen ward
            cur = conn.execute(
                "UPDATE beds SET status = 'Occupied' "
                "WHERE bed_id = ? AND ward = ? AND status = 'Available'",
                (bed_id, ward))
            if cur.rowcount != 1:
                raise ValueError(f"Bed {bed_id} is no longer available.")
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, ward, bed_id, admission_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, name, int(age), ward, bed_id, admission_date.isoformat()))
        return True, f"{name} admitted as {patient_id} to bed {bed_id} ({ward})."
    except (ValueError, sqlite3.IntegrityError) as error:
        return False, f"Admission failed: {error}"
    finally:
        conn.close()


def discharge_patient(patient_id):
    """Discharge a patient and free their bed in ONE transaction."""
    conn = get_connection()
    try:
        with conn:
            row = conn.execute(
                "SELECT bed_id, name FROM patients WHERE patient_id = ? AND status = 'Admitted'",
                (patient_id,)).fetchone()
            if row is None:
                raise ValueError(f"{patient_id} is not a currently admitted patient.")
            bed_id, name = row
            conn.execute(
                "UPDATE patients SET status = 'Discharged', discharge_date = ? WHERE patient_id = ?",
                (date.today().isoformat(), patient_id))
            conn.execute("UPDATE beds SET status = 'Available' WHERE bed_id = ?", (bed_id,))
        return True, f"{name} ({patient_id}) discharged. Bed {bed_id} is now available."
    except ValueError as error:
        return False, str(error)
    finally:
        conn.close()


def get_patients(search="", show_discharged=False):
    """Patients table, searchable by ID or name."""
    query = """
        SELECT patient_id AS "Patient ID", name AS "Name", age AS "Age",
               ward AS "Ward", bed_id AS "Bed", admission_date AS "Admission Date",
               COALESCE(discharge_date, '-') AS "Discharge Date", status AS "Status"
        FROM patients WHERE 1 = 1
    """
    params = []
    if not show_discharged:
        query += " AND status = 'Admitted'"
    if search.strip():
        query += " AND (patient_id LIKE ? OR name LIKE ?)"
        term = f"%{search.strip()}%"
        params += [term, term]
    query += " ORDER BY patient_id DESC"
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


# ---------------------------------------------------------------- history
def load_history_from_csv(csv_path=CSV_PATH):
    """Copy the simulated CSV into the occupancy_history table."""
    df = pd.read_csv(csv_path)
    with get_connection() as conn:
        conn.execute("DELETE FROM occupancy_history")
        conn.executemany(
            "INSERT INTO occupancy_history VALUES (?, ?, ?, ?, ?)",
            df[["date", "total_beds", "occupied_beds", "admissions", "discharges"]]
            .itertuples(index=False, name=None))


def get_history(start=None, end=None):
    query = "SELECT * FROM occupancy_history WHERE 1 = 1"
    params = []
    if start:
        query += " AND date >= ?"
        params.append(str(start))
    if end:
        query += " AND date <= ?"
        params.append(str(end))
    query += " ORDER BY date"
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    df["occupancy_pct"] = (df["occupied_beds"] / df["total_beds"] * 100).round(1)
    return df


# ---------------------------------------------------------------- sample data
FIRST_NAMES = ["Aarav", "Diya", "Rohan", "Ananya", "Kabir", "Meera", "Arjun", "Isha",
               "Vihaan", "Sara", "Aditya", "Neha", "Kiran", "Priya", "Rahul", "Tara",
               "Dev", "Pooja", "Nikhil", "Riya", "Manav", "Sneha", "Om", "Kavya"]
LAST_NAMES = ["Sharma", "Patil", "Iyer", "Desai", "Khan", "Joshi", "Nair", "Gupta",
              "Kulkarni", "Reddy", "Shah", "Pillai", "Mehta", "Rao", "Verma", "Das"]


def init_sample_data():
    """Fill an empty database with sample beds and FICTIONAL patients."""
    with get_connection() as conn:
        if conn.execute("SELECT COUNT(*) FROM beds").fetchone()[0] > 0:
            return  # already has data

    if not os.path.exists(CSV_PATH):
        from generate_data import save_dataset
        save_dataset(CSV_PATH)
    load_history_from_csv()

    rnd = random.Random(9)
    beds = []
    for ward, (count, prefix) in WARDS.items():
        beds += [(f"{prefix}-{n:02d}", ward) for n in range(1, count + 1)]

    # Start with today's occupancy equal to the last day in the history,
    # so the dashboard and the forecast tell the same story.
    last_occupied = int(pd.read_csv(CSV_PATH)["occupied_beds"].iloc[-1])
    last_occupied = min(last_occupied, len(beds))
    # Randomly choose which beds start occupied
    occupied = set(rnd.sample(range(len(beds)), last_occupied))

    with get_connection() as conn:
        for index, (bed_id, ward) in enumerate(beds):
            conn.execute("INSERT INTO beds VALUES (?, ?, 'Available')", (bed_id, ward))
        for number, index in enumerate(sorted(occupied), start=1):
            bed_id, ward = beds[index]
            name = f"{rnd.choice(FIRST_NAMES)} {rnd.choice(LAST_NAMES)}"
            admitted = date.today() - timedelta(days=rnd.randint(0, 9))
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, ward, bed_id, admission_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"P{number:04d}", name, rnd.randint(4, 88), ward, bed_id, admitted.isoformat()))
            conn.execute("UPDATE beds SET status = 'Occupied' WHERE bed_id = ?", (bed_id,))


def setup_database():
    """Called when the app starts."""
    create_tables()
    init_sample_data()


def reset_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    setup_database()


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_database()
        print("Database reset and sample data loaded.")
    else:
        setup_database()
        print("Database ready:", DB_PATH)
    print(get_overall_stats())
