"""
AcademIQ — Institutional Academic Performance & Early-Warning Platform
Executive analytics system for forecasting student final marks and assessing risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from studyflow import config
from studyflow.schemas import StudentInput
from studyflow.models.forest import RandomForestStudentPredictor
from studyflow.data.loader import load_student_data
from studyflow.data.validator import validate_dataframe
from studyflow.reporting.reporter import generate_batch_report, summarize_class_health

# --- Page Configuration ---
st.set_page_config(
    page_title="AcademIQ | Institutional Performance Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Professional Design System & Multi-Font CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@700;800&family=Space+Grotesk:wght@600;700&display=swap');

    /* Color Tokens */
    :root {
        --canvas-bg: #0B132B;
        --surface-card: #1C2541;
        --border-subtle: #3A506B;
        --accent-sky: #5BC0DE;
        --accent-cream: #F4D6B0;
    }

    /* Base Canvas */
    .stApp {
        background-color: #0B132B;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    /* Brand Header */
    .brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin-bottom: 0.1rem;
    }

    .brand-title span {
        color: #5BC0DE;
    }

    .brand-subtitle {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #5BC0DE;
        margin-bottom: 1.5rem;
    }

    /* Card Panels */
    .academ-card {
        background-color: #1C2541;
        border: 1px solid #3A506B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .card-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: #F4D6B0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid #3A506B;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* KPI Metric Tiles */
    .kpi-tile {
        background: linear-gradient(145deg, #1C2541, #151D33);
        border: 1px solid #3A506B;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .kpi-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #5BC0DE;
    }

    .kpi-value.cream {
        color: #F4D6B0;
    }

    .kpi-value.risk-high {
        color: #E63946;
    }

    .kpi-value.risk-moderate {
        color: #F4D6B0;
    }

    .kpi-value.risk-ontrack {
        color: #5BC0DE;
    }

    /* Institutional Advisory Box */
    .advisory-box {
        background-color: #161F36;
        border-left: 4px solid #5BC0DE;
        border-top: 1px solid #3A506B;
        border-right: 1px solid #3A506B;
        border-bottom: 1px solid #3A506B;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-top: 1rem;
    }

    .advisory-box.risk-high {
        border-left-color: #E63946;
    }

    .advisory-box.risk-moderate {
        border-left-color: #F4D6B0;
    }

    .advisory-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #F4D6B0;
        margin-bottom: 0.25rem;
    }

    .advisory-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: #F1F5F9;
        line-height: 1.5;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #3A506B;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #94A3B8;
        padding: 10px 20px;
        background-color: #1C2541;
        border-radius: 6px 6px 0 0;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #5BC0DE !important;
        border-color: #3A506B #3A506B #0B132B #3A506B !important;
        background-color: #0B132B !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Cache ML Predictor ---
@st.cache_resource
def load_ml_predictor():
    return RandomForestStudentPredictor()

predictor = load_ml_predictor()

# --- Header Section ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="brand-title">Academ<span>IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Institutional Academic Risk & Performance Forecasting Platform</div>', unsafe_allow_html=True)
with col_head2:
    st.markdown("""
    <div style="text-align: right; margin-top: 10px;">
        <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; letter-spacing: 0.08em; color: #5BC0DE; border: 1px solid #3A506B; padding: 4px 8px; border-radius: 4px; background: #1C2541;">
            SYSTEM STATUS: ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

# --- Primary Workspace Tabs ---
tab_single, tab_batch = st.tabs(["[ SINGLE STUDENT ASSESSMENT ]", "[ COHORT BATCH ANALYTICS ]"])

# ==============================================================================
# TAB 1: SINGLE STUDENT ASSESSMENT
# ==============================================================================
with tab_single:
    col_input, col_output = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("""
        <div class="academ-card">
            <div class="card-heading">Academic Performance Parameters</div>
        </div>
        """, unsafe_allow_html=True)

        g1 = st.slider("Term 1 Grade (G1)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        g2 = st.slider("Term 2 Grade (G2)", min_value=0.0, max_value=20.0, value=13.0, step=0.5)
        studytime = st.selectbox(
            "Weekly Study Commitment",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Tier 1 (< 2 Hours)", 2: "Tier 2 (2 - 5 Hours)", 3: "Tier 3 (5 - 10 Hours)", 4: "Tier 4 (> 10 Hours)"}[x],
            index=1
        )
        failures = st.slider("Historical Class Failures", min_value=0, max_value=4, value=0)
        absences = st.number_input("Recorded Absences", min_value=0, max_value=93, value=4, step=1)
        age = st.slider("Student Age", min_value=15, max_value=22, value=17)

        st.markdown("""
        <div class="academ-card">
            <div class="card-heading">Demographic & Educational Capital</div>
        </div>
        """, unsafe_allow_html=True)

        higher = st.radio("Higher Education Aspirations", ["yes", "no"], index=0, horizontal=True)
        schoolsup = st.radio("Remedial Educational Support", ["no", "yes"], index=0, horizontal=True)
        medu = st.slider("Maternal Educational Level (0-4)", 0, 4, 2)
        fedu = st.slider("Paternal Educational Level (0-4)", 0, 4, 2)

    with col_output:
        # Build schema & predict
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

        # Risk color class
        risk_class = "risk-ontrack"
        if result.risk_level == "High Risk":
            risk_class = "risk-high"
        elif result.risk_level == "Moderate Risk":
            risk_class = "risk-moderate"

        # KPI Metric Tiles Row
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Predicted Mark (G3)</div>
                <div class="kpi-value">{result.predicted_grade} <span style="font-size: 0.9rem; color: #94A3B8;">/ 20</span></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c2:
            status_text = "PASSING" if result.passed else "FAILING"
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Academic Standing</div>
                <div class="kpi-value {'cream' if not result.passed else ''}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c3:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Risk Category</div>
                <div class="kpi-value {risk_class}">{result.risk_level.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c4:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Term Momentum</div>
                <div class="kpi-value cream">{momentum:+0.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Interactive Plotly Trajectory Graph ---
        st.markdown('<div class="card-heading">Trajectory & Longitudinal Progression</div>', unsafe_allow_html=True)
        
        periods = ["Term 1 (G1)", "Term 2 (G2)", "Projected Final (G3)"]
        grades = [g1, g2, result.predicted_grade]

        fig_trajectory = go.Figure()

        # Benchmark line at 10.0
        fig_trajectory.add_hline(
            y=10.0, 
            line_dash="dot", 
            line_color="#3A506B", 
            annotation_text="Passing Threshold (10.0)", 
            annotation_position="bottom right",
            annotation_font=dict(color="#94A3B8", family="Space Grotesk", size=10)
        )

        # Trajectory Spline Curve
        fig_trajectory.add_trace(go.Scatter(
            x=periods,
            y=grades,
            mode="lines+markers+text",
            line=dict(color="#5BC0DE", width=3, shape="spline"),
            marker=dict(color="#F4D6B0", size=10, line=dict(color="#1C2541", width=2)),
            text=[f"{v:.1f}" for v in grades],
            textposition="top center",
            textfont=dict(family="JetBrains Mono", size=12, color="#F4D6B0"),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f} / 20<extra></extra>"
        ))

        fig_trajectory.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=25, b=25),
            height=280,
            yaxis=dict(
                range=[0, 21],
                gridcolor="#1C2541",
                tickfont=dict(family="JetBrains Mono", color="#94A3B8"),
                zeroline=False
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(family="Space Grotesk", color="#F8FAFC")
            )
        )
        st.plotly_chart(fig_trajectory, use_container_width=True)

        # Institutional Advisory Card
        st.markdown(f"""
        <div class="advisory-box {risk_class}">
            <div class="advisory-title">Institutional Advisory Guidance</div>
            <div class="advisory-text">{result.summary_message}</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: COHORT BATCH ANALYTICS
# ==============================================================================
with tab_batch:
    st.markdown("""
    <div class="academ-card">
        <div class="card-heading">Cohort Ingestion & Batch Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Institutional Dataset (CSV format)",
        type=["csv"],
        help="Delimited by semicolon (;) or comma (,)"
    )

    col_btn, _ = st.columns([1, 4])
    load_benchmark = col_btn.button("Load Standard Benchmark Dataset (395 Students)")

    df_to_process = None
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            try:
                df_to_process = pd.read_csv(uploaded_file, sep=";")
                if len(df_to_process.columns) < 5:
                    uploaded_file.seek(0)
                    df_to_process = pd.read_csv(uploaded_file, sep=",")
            except Exception:
                uploaded_file.seek(0)
                df_to_process = pd.read_csv(uploaded_file, sep=",")
            st.success(f"Dataset Loaded Successfully: {len(df_to_process)} records detected.")
        except Exception as err:
            st.error(f"Ingestion Failure: {err}")
    elif load_benchmark:
        df_to_process = load_student_data()
        st.info(f"Benchmark Cohort Loaded: {len(df_to_process)} records loaded.")

    if df_to_process is not None:
        is_valid, errors = validate_dataframe(df_to_process)
        if not is_valid:
            st.error("Data Validation Failed:")
            for err in errors:
                st.write(f"- {err}")
        else:
            with st.spinner("Executing cohort inference..."):
                report_df = generate_batch_report(df_to_process, predictor)
                summary = summarize_class_health(report_df)

            # Executive Metrics Summary
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Cohort Population</div>
                    <div class="kpi-value">{summary['total_students']}</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Cohort Mean (G3)</div>
                    <div class="kpi-value">{summary['avg_predicted_grade']} <span style="font-size: 0.9rem; color: #94A3B8;">/ 20</span></div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="kpi-tile">
                    <div class="kpi-label">Projected Pass Rate</div>
                    <div class="kpi-value cream">{summary['overall_pass_rate_pct']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="kpi-tile">
                    <div class="kpi-label">At-Risk Count</div>
                    <div class="kpi-value risk-high">{summary['high_risk_count']} <span style="font-size: 0.85rem; color: #94A3B8;">({summary['high_risk_pct']}%)</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Visual Analytics (Plotly Donut + Histogram) ---
            chart_col1, chart_col2 = st.columns(2, gap="medium")

            with chart_col1:
                st.markdown('<div class="card-heading">Risk Tier Distribution</div>', unsafe_allow_html=True)
                risk_counts = report_df["risk_level"].value_counts()
                
                # Assign colors based on category names
                tier_colors = []
                for label in risk_counts.index:
                    if label == "High Risk":
                        tier_colors.append("#E63946")
                    elif label == "Moderate Risk":
                        tier_colors.append("#F4D6B0")
                    else:
                        tier_colors.append("#5BC0DE")

                fig_donut = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.6,
                    marker=dict(colors=tier_colors),
                    textinfo="label+percent",
                    textfont=dict(family="Space Grotesk", size=11, color="#FFFFFF")
                )])
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with chart_col2:
                st.markdown('<div class="card-heading">Predicted Final Grade Distribution</div>', unsafe_allow_html=True)
                
                fig_hist = px.histogram(
                    report_df,
                    x="predicted_G3",
                    nbins=15,
                    color_discrete_sequence=["#5BC0DE"]
                )
                fig_hist.add_vline(x=10.0, line_dash="dot", line_color="#F4D6B0", annotation_text="Passing (10.0)", annotation_font_color="#F4D6B0")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=260,
                    xaxis=dict(title="Score (0-20)", gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
                    yaxis=dict(title="Frequency", gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8"))
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Roster Table & Export
            st.markdown('<div class="card-heading">Student Cohort Roster & Risk Stratification</div>', unsafe_allow_html=True)
            st.dataframe(
                report_df[["age", "studytime", "failures", "absences", "G1", "G2", "predicted_G3", "risk_level"]],
                use_container_width=True
            )

            csv_export = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="DOWNLOAD ANNOTATED COHORT REPORT (CSV)",
                data=csv_export,
                file_name="academiq_cohort_risk_report.csv",
                mime="text/csv"
            )
