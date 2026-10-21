"""
app.py
------
Hospital Bed Management System with ML-Based Occupancy Prediction.
Run with:  streamlit run app.py
"""

from datetime import date

import plotly.graph_objects as go
import streamlit as st

import database as db
import model

st.set_page_config(page_title="Hospital Bed Management", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------- styling
DARK_BLUE = "#0B3C6F"
BLUE = "#1F5FAD"
LIGHT_BLUE = "#A9CCF0"
ORANGE = "#E8833A"
GREY = "#6B7A90"

st.markdown("""
<style>
    .block-container {padding-top: 3.5rem; padding-bottom: 2rem; max-width: 1300px;}
    h1, h2, h3, h4 {color: #0B3C6F !important;}
    h4 {font-size: 1.1rem !important; font-weight: 650 !important; padding-bottom: .4rem !important;}
    [data-testid="stSidebar"] {background-color: #FFFFFF; border-right: 1px solid #DCE6F2;}
    .page-title {font-size: 1.9rem; line-height: 1.3; font-weight: 700; color: #0B3C6F; margin-bottom: 0;}
    .page-desc {color: #5A6B82; margin-top: .2rem; margin-bottom: 1.4rem;}
    .metric-card {background: #FFFFFF; border: 1px solid #DCE6F2; border-radius: 12px;
                  padding: 16px 18px; display: flex; gap: 14px; align-items: center;
                  box-shadow: 0 1px 2px rgba(11,60,111,.04); height: 100%;}
    .metric-icon {width: 46px; height: 46px; border-radius: 10px; background: #EAF2FC;
                  display: flex; align-items: center; justify-content: center; font-size: 1.4rem;}
    .metric-label {color: #5A6B82; font-size: .85rem;}
    .metric-value {color: #0B3C6F; font-size: 1.6rem; font-weight: 700; line-height: 1.2;}
    .metric-note {color: #7C8BA1; font-size: .78rem;}
    .brand {font-size: 1.15rem; font-weight: 700; color: #0B3C6F;}
    .brand-sub {color: #5A6B82; font-size: .8rem; margin-bottom: 1rem;}
    .demo-note {font-size: .78rem; color: #7C8BA1;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def page_header(title, description):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-desc">{description}</div>',
                unsafe_allow_html=True)


def metric_card(column, icon, label, value, note="&nbsp;"):
    column.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        </div>""", unsafe_allow_html=True)


def show_overall_metrics():
    stats = db.get_overall_stats()
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "🏥", "Total Beds", stats["total"], "All wards")
    metric_card(c2, "🛏️", "Occupied Beds", stats["occupied"], "Currently in use")
    metric_card(c3, "✅", "Available Beds", stats["available"], "Ready for admission")
    metric_card(c4, "📊", "Occupancy", f'{stats["percent"]}%', "Occupied ÷ total")
    return stats


def set_message(kind, text):
    """Store a message so it is still visible after the page reloads."""
    st.session_state["message"] = (kind, text)


def show_message():
    if "message" in st.session_state:
        kind, text = st.session_state.pop("message")
        getattr(st, kind)(text)


def style_chart(fig, height=340):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=10, r=10, t=80, b=10),
                      font=dict(color="#1B2B41"),
                      title=dict(y=0.97, yanchor="top", font=dict(color=DARK_BLUE, size=16)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def get_predictions():
    capacity = db.get_overall_stats()["total"]
    return model.run_prediction(capacity=capacity), capacity


def ward_forecast(ward_df, predicted_total):
    """
    Estimate tomorrow's occupancy per ward by sharing the hospital-level
    prediction according to each ward's current share of occupied beds.
    """
    df = ward_df.copy()
    weights = df["Occupied"] if df["Occupied"].sum() > 0 else df["Total Beds"]
    df["Predicted"] = (predicted_total * weights / weights.sum()).round().astype(int)
    df["Predicted"] = df[["Predicted", "Total Beds"]].min(axis=1)
    df["Predicted %"] = (df["Predicted"] / df["Total Beds"] * 100).round(1)
    return df


# ---------------------------------------------------------------- pages
def dashboard_page(threshold):
    page_header("Hospital Overview",
                "Live bed availability calculated from bed records, with tomorrow's ML estimate.")
    stats = show_overall_metrics()
    wards = db.get_ward_summary()
    st.write("")

    left, right = st.columns([1, 1.4])
    with left, st.container(border=True):
        fig = go.Figure(go.Pie(labels=["Occupied", "Available"],
                               values=[stats["occupied"], stats["available"]],
                               hole=0.62, marker=dict(colors=[BLUE, LIGHT_BLUE]), sort=False))
        fig.add_annotation(text=f'<b>{stats["percent"]}%</b><br>occupied', showarrow=False,
                           font=dict(size=16, color=DARK_BLUE))
        st.plotly_chart(style_chart(fig.update_layout(title="Bed Status")), width="stretch")
    with right, st.container(border=True):
        fig = go.Figure([
            go.Bar(name="Occupied", x=wards["Ward"], y=wards["Occupied"], marker_color=BLUE),
            go.Bar(name="Available", x=wards["Ward"], y=wards["Available"], marker_color=LIGHT_BLUE),
        ])
        fig.update_layout(barmode="stack", title="Occupancy by Ward", yaxis_title="Beds")
        st.plotly_chart(style_chart(fig), width="stretch")

    left, right = st.columns([1.4, 1])
    with left, st.container(border=True):
        st.markdown("#### Ward Summary")
        st.dataframe(wards, hide_index=True, width="stretch", column_config={
            "Occupancy %": st.column_config.ProgressColumn(
                "Occupancy %", min_value=0, max_value=100, format="%.1f%%")})

    with right, st.container(border=True):
        st.markdown("#### Next-Day Prediction")
        results, capacity = get_predictions()
        tomorrow = results["forecast"].iloc[0]
        st.caption(f'For {tomorrow["date"]:%d %b %Y} · Linear Regression · MAE ≈ {results["mae"]:.1f} beds')
        c1, c2 = st.columns(2)
        c1.metric("Predicted occupied", int(tomorrow["predicted_occupied"]))
        c2.metric("Predicted occupancy", f'{tomorrow["predicted_pct"]}%')

        busy = ward_forecast(wards, tomorrow["predicted_occupied"])
        busy = busy[busy["Predicted %"] >= threshold]
        if busy.empty:
            st.success(f"No ward is expected to reach {threshold}% occupancy.")
        for _, row in busy.iterrows():
            st.warning(f'**{row["Ward"]}** may reach about {row["Predicted %"]}% '
                       f'({row["Predicted"]}/{row["Total Beds"]} beds).')
        st.markdown('<p class="demo-note">Ward estimates share the hospital forecast by each '
                    "ward's current share of occupied beds. Educational estimate only.</p>",
                    unsafe_allow_html=True)


def bed_management_page():
    page_header("Bed Management", "View, add and update beds. All changes are saved to the database.")
    show_message()
    show_overall_metrics()
    st.write("")

    with st.container(border=True):
        st.markdown("#### All Beds")
        f1, f2, _ = st.columns([1, 1, 2])
        ward = f1.selectbox("Filter by ward", ["All"] + db.WARD_NAMES)
        status = f2.selectbox("Filter by status", ["All", "Available", "Occupied"])
        beds = db.get_beds(ward, status)
        st.dataframe(beds, hide_index=True, width="stretch", height=320)
        st.caption(f"Showing {len(beds)} bed(s).")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("#### Add New Bed")
        with st.form("add_bed", clear_on_submit=True):
            new_id = st.text_input("Bed ID", placeholder="e.g. GW-41")
            new_ward = st.selectbox("Ward", db.WARD_NAMES)
            if st.form_submit_button("Add bed", type="primary", width="stretch"):
                ok, msg = db.add_bed(new_id, new_ward)
                set_message("success" if ok else "error", msg)
                st.rerun()

    with c2, st.container(border=True):
        st.markdown("#### Mark as Occupied")
        available = db.get_bed_ids("Available")
        bed = st.selectbox("Available bed", available, key="to_occupy",
                           placeholder="No available beds")
        st.caption("Use for beds blocked without a registered patient (e.g. cleaning).")
        if st.button("Mark occupied", width="stretch", disabled=not available):
            ok, msg = db.update_bed_status(bed, "Occupied")
            set_message("success" if ok else "error", msg)
            st.rerun()

    with c3, st.container(border=True):
        st.markdown("#### Mark as Available")
        occupied = db.get_bed_ids("Occupied")
        bed = st.selectbox("Occupied bed", occupied, key="to_free",
                           placeholder="No occupied beds")
        st.caption("Beds with an admitted patient must be released by discharging the patient.")
        if st.button("Mark available", width="stretch", disabled=not occupied):
            ok, msg = db.update_bed_status(bed, "Available")
            set_message("success" if ok else "error", msg)
            st.rerun()


def admit_patient():
    """Button callback: runs before the page redraws, so the form can be cleared."""
    ok, msg = db.add_patient(st.session_state["p_name"], st.session_state["p_age"],
                             st.session_state["p_ward"], st.session_state.get("p_bed"),
                             st.session_state["p_date"])
    set_message("success" if ok else "error", msg)
    if ok:
        st.session_state["p_name"] = ""
        st.session_state["p_age"] = 30


def patients_page():
    page_header("Patient Management",
                "Admit patients to available beds and discharge them to release beds. "
                "All patient records are fictional demo data.")
    show_message()

    c1, c2 = st.columns([1.3, 1])
    with c1, st.container(border=True):
        st.markdown("#### Admit Patient")
        st.caption(f"New patient ID will be **{db.next_patient_id()}**")
        a, b = st.columns([2, 1])
        st.session_state.setdefault("p_name", "")
        st.session_state.setdefault("p_age", 30)
        a.text_input("Patient name", key="p_name")
        b.number_input("Age", min_value=0, max_value=120, step=1, key="p_age")
        a, b = st.columns(2)
        ward = a.selectbox("Ward", db.WARD_NAMES, key="p_ward")
        beds = db.get_bed_ids("Available", ward)
        b.selectbox("Available bed", beds, key="p_bed", placeholder="No beds free in this ward")
        st.date_input("Admission date", value=date.today(), max_value=date.today(), key="p_date")
        if not beds:
            st.warning(f"{ward} has no available beds right now.")
        st.button("Admit patient", type="primary", on_click=admit_patient, disabled=not beds)

    with c2, st.container(border=True):
        st.markdown("#### Discharge Patient")
        admitted = db.get_patients()
        options = {f'{r["Patient ID"]} — {r["Name"]} ({r["Bed"]})': r["Patient ID"]
                   for _, r in admitted.iterrows()}
        choice = st.selectbox("Admitted patient", list(options), placeholder="No admitted patients")
        st.caption("Discharging marks the patient's bed as Available.")
        if st.button("Discharge", width="stretch", disabled=not options):
            ok, msg = db.discharge_patient(options[choice])
            set_message("success" if ok else "error", msg)
            st.rerun()

    with st.container(border=True):
        st.markdown("#### Registered Patients")
        s1, s2 = st.columns([2, 1])
        search = s1.text_input("Search by patient ID or name", placeholder="e.g. P0012 or Sharma")
        s2.write("")
        show_all = s2.checkbox("Include discharged patients")
        patients = db.get_patients(search, show_all)
        if patients.empty:
            st.info("No patients match your search.")
        else:
            st.dataframe(patients, hide_index=True, width="stretch", height=330)
            st.caption(f"{len(patients)} patient(s) found.")


def prediction_page(threshold):
    page_header("Occupancy Prediction (Machine Learning)",
                "Linear Regression trained on simulated historical data to estimate occupied beds.")
    results, capacity = get_predictions()
    fc = results["forecast"]
    tomorrow = fc.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "📅", "Forecast date", f'{tomorrow["date"]:%d %b}', "Day after last record")
    metric_card(c2, "🛏️", "Predicted occupied", int(tomorrow["predicted_occupied"]), f"of {capacity} beds")
    metric_card(c3, "✅", "Predicted available", int(tomorrow["predicted_available"]), "beds")
    metric_card(c4, "📊", "Predicted occupancy", f'{tomorrow["predicted_pct"]}%', f"Alert at {threshold}%")
    st.write("")

    over = fc[fc["predicted_pct"] >= threshold]
    if over.empty:
        st.success(f"Predicted occupancy stays below the {threshold}% alert threshold for the next 7 days.")
    else:
        days = ", ".join(f"{d:%d %b}" for d in over["date"])
        st.warning(f"Predicted occupancy reaches {threshold}% or more on: {days}. Plan bed availability early.")

    left, right = st.columns([1, 1.5])
    with left, st.container(border=True):
        fig = go.Figure(go.Bar(x=fc["date"].dt.strftime("%a %d %b"), y=fc["predicted_occupied"],
                               marker_color=[ORANGE if p >= threshold else BLUE for p in fc["predicted_pct"]],
                               text=fc["predicted_occupied"], textposition="outside"))
        fig.add_hline(y=capacity * threshold / 100, line_dash="dot", line_color=ORANGE,
                      annotation_text=f"{threshold}% threshold")
        fig.update_layout(title="7-Day Forecast", yaxis_title="Occupied beds",
                          yaxis_range=[0, capacity * 1.1])
        st.plotly_chart(style_chart(fig, 380), width="stretch")

    with right, st.container(border=True):
        data, test = results["data"], results["test"]
        fig = go.Figure([
            go.Scatter(x=data["date"], y=data["occupied_beds"], name="Actual (history)",
                       line=dict(color=BLUE, width=1.5)),
            go.Scatter(x=test["date"], y=test["predicted"], name="Model prediction (test days)",
                       line=dict(color=ORANGE, width=2, dash="dash")),
            go.Scatter(x=fc["date"], y=fc["predicted_occupied"], name="7-day forecast",
                       mode="lines+markers", line=dict(color=DARK_BLUE, width=2, dash="dot")),
        ])
        fig.update_layout(title="Historical vs Predicted Occupancy", yaxis_title="Occupied beds")
        st.plotly_chart(style_chart(fig, 380), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("#### Model Performance")
        test = results["test"]
        st.metric("Mean Absolute Error (MAE)", f'{results["mae"]:.2f} beds')
        st.write(f"The model was trained on the oldest **{len(results['train'])} days** and tested on "
                 f"the newest **{len(test)} days** ({test['date'].min():%d %b} – {test['date'].max():%d %b %Y}), "
                 "which it had never seen.")
        st.write(f"On those test days its prediction was off by about **{results['mae']:.1f} beds** "
                 "on average. Lower MAE means better predictions.")

    with right, st.container(border=True):
        st.markdown("#### How the Model Works")
        st.write("Linear Regression finds the straight-line relationship that best fits the history:")
        st.code(f"occupied_today = {results['intercept']:.2f}\n"
                f"               + {results['coef_day']:.4f} × day_number\n"
                f"               + {results['coef_prev']:.3f} × occupied_yesterday", language="text")
        st.write("Each forecast day uses the previous day's prediction as its *yesterday* value. "
                 "Only past information is used for every prediction.")

    st.info("**Educational estimate only.** The model learns a trend from *simulated* data. "
            "It is not suitable for real hospital planning without real data and proper validation.")


def analytics_page():
    page_header("Historical Analytics", "Trends from the simulated occupancy history.")
    history = db.get_history()
    first, last = history["date"].min().date(), history["date"].max().date()

    with st.container(border=True):
        picked = st.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    if not isinstance(picked, tuple) or len(picked) != 2:
        st.info("Select both a start and an end date.")
        return
    data = db.get_history(picked[0], picked[1])
    if data.empty:
        st.warning("No records in the selected range.")
        return

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "📅", "Days selected", len(data), f"{picked[0]:%d %b} – {picked[1]:%d %b}")
    metric_card(c2, "➕", "Total admissions", int(data["admissions"].sum()))
    metric_card(c3, "➖", "Total discharges", int(data["discharges"].sum()))
    metric_card(c4, "📊", "Average occupancy", f'{data["occupancy_pct"].mean():.1f}%')
    st.write("")

    with st.container(border=True):
        fig = go.Figure(go.Scatter(x=data["date"], y=data["occupancy_pct"], fill="tozeroy",
                                   line=dict(color=BLUE), fillcolor="rgba(31,95,173,0.12)",
                                   name="Occupancy %"))
        fig.update_layout(title="Occupancy Trend", yaxis_title="Occupancy %", yaxis_range=[0, 105])
        st.plotly_chart(style_chart(fig), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        weekly = data.set_index("date")[["admissions", "discharges"]].resample("W").sum().reset_index()
        fig = go.Figure([
            go.Bar(name="Admissions", x=weekly["date"], y=weekly["admissions"], marker_color=BLUE),
            go.Bar(name="Discharges", x=weekly["date"], y=weekly["discharges"], marker_color=LIGHT_BLUE),
        ])
        fig.update_layout(title="Weekly Admissions vs Discharges", barmode="group", yaxis_title="Patients")
        st.plotly_chart(style_chart(fig), width="stretch")

    with right, st.container(border=True):
        wards = db.get_ward_summary()
        fig = go.Figure(go.Bar(x=wards["Occupancy %"], y=wards["Ward"], orientation="h",
                               marker_color=BLUE, text=wards["Occupancy %"].astype(str) + "%",
                               textposition="outside"))
        fig.update_layout(title="Ward-wise Occupancy (current bed records)",
                          xaxis_title="Occupancy %", xaxis_range=[0, 115])
        st.plotly_chart(style_chart(fig), width="stretch")

    st.caption("History is simulated demonstration data, not records from a real hospital.")


# ---------------------------------------------------------------- main
def main():
    db.setup_database()

    with st.sidebar:
        st.markdown('<div class="brand">🏥 CarePoint BMS</div>'
                    '<div class="brand-sub">Hospital Bed Management System</div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigation", ["Dashboard", "Bed Management", "Patients",
                                       "ML Prediction", "Analytics"], label_visibility="collapsed")
        st.divider()
        threshold = st.slider("Capacity alert threshold (%)", 60, 100, 85, step=5,
                              help="Warnings appear when predicted occupancy reaches this level.")
        st.divider()
        st.caption("College mini-project · Demo data only\n\nStreamlit · SQLite · scikit-learn")

    if page == "Dashboard":
        dashboard_page(threshold)
    elif page == "Bed Management":
        bed_management_page()
    elif page == "Patients":
        patients_page()
    elif page == "ML Prediction":
        prediction_page(threshold)
    else:
        analytics_page()


main()
