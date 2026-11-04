"""
StudyFlow — Interactive Streamlit Web Application
Provides real-time individual student risk forecasting and classroom batch analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from studyflow import config
from studyflow.schemas import StudentInput
from studyflow.models.forest import RandomForestStudentPredictor
from studyflow.data.loader import load_student_data
from studyflow.data.validator import validate_dataframe
from studyflow.reporting.reporter import generate_batch_report, summarize_class_health

# --- Page Configuration ---
st.set_page_config(
    page_title="StudyFlow — Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cache the ML Predictor in RAM ---
@st.cache_resource
def load_ml_predictor():
    return RandomForestStudentPredictor()

predictor = load_ml_predictor()

# --- Main App Header ---
st.title("🎓 StudyFlow: Academic Performance & Early-Warning Platform")
st.markdown(
    "Identify academic risk early, monitor learning momentum, and allocate constructive educational support."
)

# --- Navigation Tabs ---
tab1, tab2 = st.tabs(["👤 Single Student Predictor", "📁 Batch Classroom Upload & Analytics"])

# ==============================================================================
# TAB 1: SINGLE STUDENT PREDICTOR
# ==============================================================================
with tab1:
    st.subheader("Individual Student Performance Forecast")
    st.write("Adjust the student's profile and term grades to predict final exam performance and assess risk.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Student Academic Profile")
        g1 = st.slider("Term 1 Grade (G1)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        g2 = st.slider("Term 2 Grade (G2)", min_value=0.0, max_value=20.0, value=13.0, step=0.5)
        studytime = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x],
            index=1
        )
        failures = st.slider("Past Class Failures", min_value=0, max_value=4, value=0)
        absences = st.number_input("School Absences", min_value=0, max_value=93, value=4, step=1)
        age = st.slider("Age", min_value=15, max_value=22, value=17)

        with st.expander("Socioeconomic & Support Factors"):
            higher = st.radio("Plans for Higher Education?", ["yes", "no"], index=0, horizontal=True)
            schoolsup = st.radio("Extra Educational Support?", ["no", "yes"], index=0, horizontal=True)
            medu = st.slider("Mother's Education (0: None to 4: Higher)", 0, 4, 2)
            fedu = st.slider("Father's Education (0: None to 4: Higher)", 0, 4, 2)

    with col2:
        st.markdown("### 📊 Prediction & Trajectory Analysis")

        # Build schema and predict
        student_input = StudentInput(
            age=int(age),
            studytime=int(studytime),
            failures=int(failures),
            absences=int(absences),
            G1=float(g1),
            G2=float(g2),
            higher=higher,
            Medu=int(medu),
            Fedu=int(fedu),
            schoolsup=schoolsup
        )
        result = predictor.predict_student(student_input)
        momentum = round(g2 - g1, 2)

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Final Mark (G3)", f"{result.predicted_grade} / 20")
        m2.metric("Status", "Passing" if result.passed else "Failing")
        m3.metric("Risk Level", result.risk_level)
        m4.metric("Academic Momentum", f"{momentum:+0.1f}", delta=f"{momentum:+0.1f}")

        # Trajectory Visualization
        st.markdown("#### 📈 Grade Trajectory")
        trajectory_df = pd.DataFrame({
            "Evaluation Period": ["Term 1 (G1)", "Term 2 (G2)", "Predicted Final (G3)"],
            "Grade (0-20)": [g1, g2, result.predicted_grade]
        })
        st.line_chart(trajectory_df.set_index("Evaluation Period"))

        # Advisory Recommendation Box
        st.markdown("#### 💡 Early Intervention Advisory")
        if result.risk_level == "High Risk":
            st.error(f"⚠️ **High Risk Alert**: {result.summary_message}")
        elif result.risk_level == "Moderate Risk":
            st.warning(f"🔔 **Monitoring Recommended**: {result.summary_message}")
        else:
            st.success(f"✅ **On Track**: {result.summary_message}")

# ==============================================================================
# TAB 2: BATCH CLASSROOM UPLOAD & ANALYTICS
# ==============================================================================
with tab2:
    st.subheader("Classroom Batch Assessment")
    st.write("Upload a class CSV file to evaluate risk across an entire student cohort.")

    uploaded_file = st.file_uploader("Upload Classroom CSV (semicolon or comma delimited)", type=["csv"])

    col_btn, _ = st.columns([1, 4])
    load_benchmark = col_btn.button("Load Benchmark Dataset (student-mat.csv)")

    df_to_process = None
    if uploaded_file is not None:
        try:
            # Try semicolon first (UCI standard), fallback to comma
            uploaded_file.seek(0)
            try:
                df_to_process = pd.read_csv(uploaded_file, sep=";")
                if len(df_to_process.columns) < 5:
                    uploaded_file.seek(0)
                    df_to_process = pd.read_csv(uploaded_file, sep=",")
            except Exception:
                uploaded_file.seek(0)
                df_to_process = pd.read_csv(uploaded_file, sep=",")
            st.success(f"Successfully uploaded file with {len(df_to_process)} records.")
        except Exception as err:
            st.error(f"Error reading uploaded CSV: {err}")
    elif load_benchmark:
        df_to_process = load_student_data()
        st.info(f"Loaded benchmark dataset with {len(df_to_process)} students.")

    if df_to_process is not None:
        is_valid, errors = validate_dataframe(df_to_process)
        if not is_valid:
            st.error("Validation failed for the uploaded dataset:")
            for err in errors:
                st.write(f"- {err}")
        else:
            # Run batch predictions and summarization
            with st.spinner("Processing batch predictions..."):
                report_df = generate_batch_report(df_to_process, predictor)
                summary = summarize_class_health(report_df)

            st.markdown("### 🏫 Cohort Overview")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Cohort Size", f"{summary['total_students']} Students")
            k2.metric("Class Average (G3)", f"{summary['avg_predicted_grade']} / 20")
            k3.metric("Projected Pass Rate", f"{summary['overall_pass_rate_pct']}%")
            k4.metric("High Risk Count", f"{summary['high_risk_count']} ({summary['high_risk_pct']}%)")

            # Risk Distribution Breakdown
            st.markdown("### 📊 Risk Distribution")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("**Risk Tier Breakdown**")
                risk_counts = report_df["risk_level"].value_counts()
                st.bar_chart(risk_counts)
            with chart_col2:
                st.markdown("**Predicted Grade Distribution (0-20)**")
                grade_bins = pd.cut(report_df["predicted_G3"], bins=[0, 9, 12, 16, 20], labels=["<9 (Fail)", "9-12 (Border)", "12-16 (Good)", "16-20 (Top)"])
                st.bar_chart(grade_bins.value_counts().sort_index())

            # Data Table & Export
            st.markdown("### 📋 Student Risk Roster")
            st.dataframe(
                report_df[["age", "studytime", "failures", "absences", "G1", "G2", "predicted_G3", "risk_level"]],
                use_container_width=True
            )

            # Export Button
            csv_data = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated Risk Report (CSV)",
                data=csv_data,
                file_name="studyflow_student_risk_report.csv",
                mime="text/csv"
            )