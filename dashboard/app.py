# dashboard/app.py

import streamlit as st
import sys
import os

# Fix path for agents import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import Orchestrator

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Agentic AI Based Healthcare Decision Support System",
    layout="wide"
)

orch = Orchestrator()

# ==========================================
# SESSION STATE INIT
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None


# ==========================================
# INPUT PAGE
# ==========================================
def input_page():

    st.title("🧠 Agentic AI Based Healthcare Decision Support System")
    st.markdown("### Enter Patient Clinical Parameters")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        gender = st.selectbox("Gender", [0, 1],
                              format_func=lambda x: "Male" if x == 1 else "Female")
        hr = st.number_input("Heart Rate", 30, 200, 90)
        bp = st.number_input("Blood Pressure", 60, 200, 120)
        spo2 = st.number_input("SpO2", 70, 100, 96)

    with col2:
        temp = st.number_input("Temperature", 34.0, 42.0, 37.0)
        rr = st.number_input("Respiratory Rate", 8, 40, 18)
        glucose = st.number_input("Glucose", 50, 400, 110)
        wbc = st.number_input("WBC", 1.0, 50.0, 7.5)
        creatinine = st.number_input("Creatinine", 0.1, 15.0, 1.0)
        icu_days = st.number_input("ICU Stay Days", 0, 30, 2)

    if st.button("🚀 Run AI Assessment"):

        st.session_state.patient_data = {
            "age": age,
            "gender": gender,
            "hr": hr,
            "bp": bp,
            "spo2": spo2,
            "temp": temp,
            "rr": rr,
            "glucose": glucose,
            "wbc": wbc,
            "creatinine": creatinine,
            "icu_stay_days": icu_days
        }

        st.session_state.page = "output"
        st.rerun()


# ==========================================
# OUTPUT PAGE
# ==========================================
def output_page():

    data = st.session_state.patient_data

    if data is None:
        st.error("No patient data found.")
        return

    result = orch.run(data)

    st.title("📊 AI Risk Intelligence Dashboard")

    # =========================
    # TOP METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Probability", f"{result['probability']:.3f}")
    col2.metric("Model Confidence", f"{result['confidence']:.3f}")
    col3.metric("Fairness Score", f"{result['fairness_score']:.3f}")

    st.divider()

    # =========================
    # CLINICAL AGENT OUTPUT
    # =========================
    st.subheader("🧠 Clinical Agent Output")

    if result["risk_level"] == "HIGH":
        st.error("🚨 HIGH RISK PATIENT")
    elif result["risk_level"] == "MEDIUM":
        st.warning("⚠ MEDIUM RISK PATIENT")
    else:
        st.success("✅ LOW RISK PATIENT")

    st.write(f"**Predicted Risk Level:** {result['risk_level']}")
    st.write(f"**Mortality Probability:** {result['probability']:.3f}")
    st.write(f"**Severity Score:** {result['severity_score']}")
    st.write(f"**Model Confidence:** {result['confidence']:.3f}")

    st.divider()

    # =========================
    # ADMIN AGENT OUTPUT
    # =========================
    st.subheader("🏥 Admin Agent – Clinical Protocol Recommendation")
    st.info(result["clinical_guideline"])

    st.divider()

    # =========================
    # OPERATIONS AGENT OUTPUT
    # =========================
    st.subheader("⚙ Operations Agent – Resource Allocation Plan")
    st.warning(result["operations_plan"])

    st.divider()

    # =========================
    # ORCHESTRATOR FINAL SUMMARY
    # =========================
    st.subheader("📋 Orchestrator Final Decision Summary")

    st.success(
        f"""
        Overall Condition: **{result['risk_level']} Risk**

        Based on clinical prediction, hospital protocol knowledge model,
        and operational resource policy model, the system recommends
        the above coordinated action plan.

        Final Decision Confidence: **{result['confidence']:.3f}**
        """
    )

    st.divider()

    # =========================
    # RAW DEBUG VIEW
    # =========================
    with st.expander("🔍 View Raw Agent Outputs"):
        st.json(result)

    # =========================
    # RESET BUTTON
    # =========================
    if st.button("🔄 Evaluate Another Patient"):
        st.session_state.page = "input"
        st.session_state.patient_data = None
        st.rerun()


# ==========================================
# ROUTER
# ==========================================
if st.session_state.page == "input":
    input_page()
else:
    output_page()