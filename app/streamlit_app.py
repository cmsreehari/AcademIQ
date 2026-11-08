"""
AcademIQ — Institutional Academic Performance & Early-Warning Platform
Executive analytics system featuring SHAP Explainability, Goal Planning,
What-If Counterfactual Simulation, and AI Study Coaching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from studyflow import config
from studyflow.schemas import StudentInput
from studyflow.models.forest import RandomForestStudentPredictor
from studyflow.models.explainer import StudentModelExplainer
from studyflow.analytics.planner import plan_target_grade
from studyflow.analytics.simulator import simulate_intervention
from studyflow.tutor.coach import generate_recovery_plan
from studyflow.data.loader import load_student_data
from studyflow.data.validator import validate_dataframe
from studyflow.reporting.reporter import generate_batch_report, summarize_class_health

# --- Page Configuration ---
st.set_page_config(
    page_title="AcademIQ | Institutional Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Professional Design System & Multi-Font CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&family=Outfit:wght@600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Color Tokens */
    :root {
        --canvas-bg: #0B132B;
        --surface-card: #1C2541;
        --surface-dark: #121A30;
        --border-subtle: #3A506B;
        --accent-sky: #5BC0DE;
        --accent-cream: #F4D6B0;
        --alert-crimson: #E63946;
        --text-muted: #94A3B8;
        --text-bright: #F8FAFC;
    }

    /* Base Canvas & Global Defaults */
    .stApp {
        background-color: var(--canvas-bg);
        color: var(--text-bright);
        font-family: 'Inter', sans-serif;
    }

    /* 1. BRAND & HEADER TYPOGRAPHY */
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .brand-title span { color: var(--accent-sky); }

    .brand-subtext {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent-sky);
        margin-top: 0.2rem;
    }

    .meta-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        background: var(--surface-card);
        border: 1px solid var(--border-subtle);
        padding: 4px 10px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }

    /* 2. SECTION & CARD HEADINGS */
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-cream);
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0.5rem;
        margin-bottom: 0.9rem;
    }

    .sub-section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }

    /* 3. STRUCTURED CARD CONTAINERS */
    .academ-panel {
        background-color: var(--surface-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.15rem;
        margin-bottom: 1rem;
    }

    /* 4. KPI STAT TILES */
    .kpi-box {
        background: linear-gradient(160deg, var(--surface-card), var(--surface-dark));
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem 0.8rem;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .kpi-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--accent-sky);
        line-height: 1.2;
    }
    .kpi-number.cream { color: var(--accent-cream); }
    .kpi-number.crimson { color: var(--alert-crimson); }
    .kpi-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* 5. INSTITUTIONAL ADVISORY MEMORANDUM */
    .memo-card {
        background-color: var(--surface-dark);
        border-left: 4px solid var(--accent-sky);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-top: 0.8rem;
    }
    .memo-card.risk-high { border-left-color: var(--alert-crimson); }
    .memo-card.risk-moderate { border-left-color: var(--accent-cream); }

    .memo-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--accent-cream);
        margin-bottom: 0.4rem;
    }
    .memo-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #E2E8F0;
        line-height: 1.55;
    }

    /* 6. STREAMLIT WIDGET INTEGRATION */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--text-muted);
        padding: 8px 18px;
        background-color: var(--surface-card);
        border-radius: 6px 6px 0 0;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-sky) !important;
        border-color: var(--border-subtle) var(--border-subtle) var(--canvas-bg) var(--border-subtle) !important;
        background-color: var(--canvas-bg) !important;
    }

    /* Native Inputs Polishing */
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stRadio"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
    }

    /* Primary Buttons */
    .stButton button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        background-color: var(--surface-card) !important;
        color: var(--accent-sky) !important;
        border: 1px solid var(--accent-sky) !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton button:hover {
        background-color: var(--accent-sky) !important;
        color: var(--canvas-bg) !important;
        border-color: var(--accent-sky) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Cache Model Engines in RAM ---
@st.cache_resource
def load_prediction_engine():
    predictor = RandomForestStudentPredictor()
    explainer = StudentModelExplainer(predictor)
    return predictor, explainer

predictor, explainer = load_prediction_engine()

# --- TOP NAVIGATION & INSTITUTIONAL HEADER ---
header_col1, header_col2 = st.columns([3, 2])
with header_col1:
    st.markdown('<div class="brand-title">Academ<span>IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtext">Institutional Academic Early-Warning & Forecasting Suite</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 12px; display: flex; gap: 8px; justify-content: flex-end;">
        <span class="meta-tag">ENGINE: RANDOM FOREST REGRESSOR</span>
        <span class="meta-tag">STATUS: ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-bottom: 1px solid #3A506B; margin: 10px 0 16px 0;'>", unsafe_allow_html=True)

# --- PRIMARY WORKSPACE TABS ---
tab_single, tab_batch = st.tabs(["[ 1. INDIVIDUAL STUDENT ASSESSMENT ]", "[ 2. COHORT BATCH ANALYTICS ]"])

# ==============================================================================
# TAB 1: INDIVIDUAL STUDENT ASSESSMENT
# ==============================================================================
with tab_single:
    col_input, col_output = st.columns([1, 2], gap="large")

    # --- LEFT COLUMN: CONTROL & PARAMETER PANEL ---
    with col_input:
        st.markdown('<div class="card-title">Parameter Console</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="sub-section-title">Academic Baseline Marks</div>', unsafe_allow_html=True)
            g1 = st.slider("Term 1 Evaluation (G1)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
            g2 = st.slider("Term 2 Evaluation (G2)", min_value=0.0, max_value=20.0, value=9.0, step=0.5)

            st.markdown('<div class="sub-section-title" style="margin-top: 14px;">Behavioral & Engagement Metrics</div>', unsafe_allow_html=True)
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Tier 1 (< 2 Hours/Week)", 2: "Tier 2 (2 - 5 Hours/Week)", 3: "Tier 3 (5 - 10 Hours/Week)", 4: "Tier 4 (> 10 Hours/Week)"}[x],
                index=0
            )
            failures = st.slider("Past Course Failures", min_value=0, max_value=4, value=1)
            absences = st.number_input("Recorded Absences", min_value=0, max_value=93, value=10, step=1)
            age = st.slider("Student Age", min_value=15, max_value=22, value=17)

        with st.expander("Socioeconomic & Support Context"):
            higher = st.radio("University Aspirations", ["yes", "no"], index=0, horizontal=True)
            schoolsup = st.radio("Remedial School Support", ["no", "yes"], index=0, horizontal=True)
            medu = st.slider("Maternal Education Level (0-4)", 0, 4, 2)
            fedu = st.slider("Paternal Education Level (0-4)", 0, 4, 2)

    # --- RIGHT COLUMN: EXECUTIVE INTELLIGENCE SUITE ---
    with col_output:
        # Construct input and run prediction
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

        risk_color_class = "cream"
        if result.risk_level == "High Risk":
            risk_color_class = "crimson"
        elif result.risk_level == "On Track":
            risk_color_class = ""

        # --- ROW 1: 4 KPI STAT TILES ---
        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        with kpi_1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Forecast Mark (G3)</div>
                <div class="kpi-number">{result.predicted_grade:.2f}</div>
                <div class="kpi-sub">Scale: 0.00 – 20.00</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_2:
            status_text = "PASSING" if result.passed else "FAILING"
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Academic Status</div>
                <div class="kpi-number {'crimson' if not result.passed else ''}">{status_text}</div>
                <div class="kpi-sub">Benchmark: 10.00</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Risk Category</div>
                <div class="kpi-number {risk_color_class}">{result.risk_level.upper()}</div>
                <div class="kpi-sub">Classification Tier</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_4:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Term Momentum</div>
                <div class="kpi-number cream">{momentum:+0.1f}</div>
                <div class="kpi-sub">Delta (G2 - G1)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

        # --- ROW 2: WORKSPACE SUB-TABS (Ordered & Uncluttered) ---
        studio_tab_diag, studio_tab_plan, studio_tab_coach = st.tabs([
            "[ ANALYTICAL DIAGNOSTICS & SHAP ]",
            "[ GOAL PLANNER & WHAT-IF STUDIO ]",
            "[ AI RECOVERY ROADMAP ]"
        ])

        # SUB-TAB 1: DIAGNOSTICS & SHAP
        with studio_tab_diag:
            diag_col1, diag_col2 = st.columns(2, gap="medium")

            with diag_col1:
                st.markdown('<div class="card-title">Longitudinal Trajectory</div>', unsafe_allow_html=True)
                periods = ["Term 1 (G1)", "Term 2 (G2)", "Forecast (G3)"]
                grades = [g1, g2, result.predicted_grade]

                fig_trajectory = go.Figure()
                fig_trajectory.add_hline(
                    y=10.0, line_dash="dot", line_color="#3A506B",
                    annotation_text="Pass Threshold (10.0)", annotation_position="bottom right",
                    annotation_font=dict(color="#94A3B8", family="Space Grotesk", size=10)
                )
                fig_trajectory.add_trace(go.Scatter(
                    x=periods, y=grades, mode="lines+markers+text",
                    line=dict(color="#5BC0DE", width=3, shape="spline"),
                    marker=dict(color="#F4D6B0", size=9, line=dict(color="#1C2541", width=2)),
                    text=[f"{v:.1f}" for v in grades], textposition="top center",
                    textfont=dict(family="JetBrains Mono", size=11, color="#F4D6B0"),
                    hovertemplate="<b>%{x}</b><br>Score: %{y:.1f} / 20<extra></extra>"
                ))
                fig_trajectory.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=15, r=15, t=20, b=20), height=240,
                    yaxis=dict(range=[0, 21], gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8", size=10)),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="Space Grotesk", color="#F8FAFC", size=11))
                )
                st.plotly_chart(fig_trajectory, use_container_width=True)

            with diag_col2:
                st.markdown('<div class="card-title">SHAP Factor Attribution</div>', unsafe_allow_html=True)
                shap_res = explainer.explain_student(student_input)

                factors = []
                point_impacts = []
                bar_colors = []

                # Top negative detractors
                for feat, val in reversed(shap_res.top_negative_factors[:4]):
                    factors.append(feat)
                    point_impacts.append(val)
                    bar_colors.append("#E63946")

                # Top positive drivers
                for feat, val in shap_res.top_positive_factors[:4]:
                    factors.append(feat)
                    point_impacts.append(val)
                    bar_colors.append("#5BC0DE")

                fig_shap = go.Figure(go.Bar(
                    x=point_impacts, y=factors, orientation='h',
                    marker=dict(color=bar_colors),
                    text=[f"{v:+0.2f}" for v in point_impacts], textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=10, color="#F8FAFC"),
                    hovertemplate="Factor: <b>%{y}</b><br>Impact: <b>%{x:+0.2f} pts</b><extra></extra>"
                ))
                fig_shap.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=15, r=35, t=20, b=20), height=240,
                    xaxis=dict(title="Delta vs. 10.3 Baseline", gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8", size=9)),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="Space Grotesk", color="#F8FAFC", size=11))
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            # Institutional Advisory Notice
            memo_class = "risk-high" if result.risk_level == "High Risk" else ("risk-moderate" if result.risk_level == "Moderate Risk" else "")
            st.markdown(f"""
            <div class="memo-card {memo_class}">
                <div class="memo-header">Institutional Advisory Notice</div>
                <div class="memo-body">{result.summary_message}</div>
            </div>
            """, unsafe_allow_html=True)

        # SUB-TAB 2: GOAL PLANNER & SIMULATOR
        with studio_tab_plan:
            st.markdown('<div class="card-title">Prescriptive Intervention & Scenario Simulation</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2, gap="medium")

            with p_col1:
                st.markdown('<div class="sub-section-title">Target Goal Planner (Reverse Optimization)</div>', unsafe_allow_html=True)
                target_g3 = st.slider("Designated Target Final Mark", min_value=10.0, max_value=20.0, value=12.0, step=0.5)
                plan = plan_target_grade(student_input, target_g3, predictor)

                feasibility_color = "#5BC0DE" if plan.is_achievable else "#F4D6B0"
                st.markdown(f"""
                <div class="academ-panel" style="margin-top: 6px;">
                    <div style="font-family: 'Space Grotesk'; font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Feasibility Assessment</div>
                    <div style="font-family: 'Outfit'; font-size: 1.15rem; color: {feasibility_color}; font-weight: 700;">
                        {'FEASIBLE ACTION PLAN' if plan.is_achievable else 'ELEVATED DIFFICULTY TARGET'}
                    </div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #E2E8F0; margin-top: 4px;">
                        Point Deficit: <b>{plan.point_gap:+0.1f} marks</b>
                    </div>
                    <ul style="font-family: 'Inter'; font-size: 0.84rem; color: #CBD5E1; margin-top: 8px; padding-left: 18px;">
                        {"".join([f"<li>{step}</li>" for step in plan.action_steps])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with p_col2:
                st.markdown('<div class="sub-section-title">What-If Counterfactual Sandbox</div>', unsafe_allow_html=True)
                sim_st = st.selectbox("Simulated Weekly Study Tier", [1, 2, 3, 4], index=min(3, student_input.studytime + 1), format_func=lambda x: f"Tier {x}")
                sim_abs = st.number_input("Simulated Target Absences", min_value=0, max_value=int(student_input.absences), value=min(2, int(student_input.absences)))
                sim_result = simulate_intervention(student_input, new_studytime=sim_st, new_absences=sim_abs, predictor=predictor)

                delta_color = "#5BC0DE" if sim_result.grade_delta >= 0 else "#E63946"
                st.markdown(f"""
                <div class="academ-panel" style="margin-top: 6px;">
                    <div style="font-family: 'Space Grotesk'; font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Counterfactual Projection</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.25rem; font-weight: 700; color: {delta_color};">
                        {sim_result.simulated_grade:.2f} / 20.00
                        <span style="font-size: 0.85rem; font-family: 'Space Grotesk';">({sim_result.grade_delta:+0.2f} pts)</span>
                    </div>
                    <div style="font-family: 'Inter'; font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
                        Projected Risk Tier: <b style="color: #F8FAFC;">{sim_result.new_risk_level.upper()}</b>
                    </div>
                    <div style="font-family: 'Inter'; font-size: 0.78rem; color: var(--text-muted); margin-top: 8px;">
                        Note: Study time and absence reductions act as multipliers on exam readiness. Consistent concept mastery on unit tests yields maximum score acceleration.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # SUB-TAB 3: AI COACH
        with studio_tab_coach:
            st.markdown('<div class="card-title">AI Academic Coach & Recovery Roadmap</div>', unsafe_allow_html=True)
            st.write("Synthesizes student longitudinal metrics and SHAP constraints into an institutional 4-week recovery memorandum.")

            if st.button("SYNTHESIZE 4-WEEK RECOVERY ROADMAP", use_container_width=True):
                with st.spinner("Compiling pedagogical recovery roadmap..."):
                    coach_plan = generate_recovery_plan(
                        student=student_input,
                        target_grade=target_g3,
                        current_prediction=result.predicted_grade,
                        explanation=shap_res
                    )
                    st.markdown(f"""
                    <div class="memo-card">
                        <div class="memo-header">Primary Bottleneck: {coach_plan.primary_bottleneck}</div>
                        <div class="memo-body">{coach_plan.raw_markdown}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: COHORT BATCH ANALYTICS
# ==============================================================================
with tab_batch:
    st.markdown('<div class="card-title">Cohort Ingestion & Batch Risk Analytics</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Institutional Dataset (CSV format)",
        type=["csv"],
        help="Semicolon (;) or Comma (,) delimited"
    )

    col_btn, _ = st.columns([1, 4])
    load_benchmark = col_btn.button("LOAD BENCHMARK COHORT (395 STUDENTS)")

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
            st.success(f"Cohort Ingested: {len(df_to_process)} student records identified.")
        except Exception as err:
            st.error(f"Ingestion Failure: {err}")
    elif load_benchmark:
        df_to_process = load_student_data()
        st.info(f"Standard Benchmark Cohort Active: {len(df_to_process)} records loaded.")

    if df_to_process is not None:
        is_valid, errors = validate_dataframe(df_to_process)
        if not is_valid:
            st.error("Data Validation Exceptions:")
            for err in errors:
                st.write(f"- {err}")
        else:
            with st.spinner("Processing batch cohort inference..."):
                report_df = generate_batch_report(df_to_process, predictor)
                summary = summarize_class_health(report_df)

            # 4 Cohort KPI Tiles
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-label">Cohort Size</div>
                    <div class="kpi-number">{summary['total_students']}</div>
                    <div class="kpi-sub">Total Active Students</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-label">Cohort Mean (G3)</div>
                    <div class="kpi-number">{summary['avg_predicted_grade']:.2f}</div>
                    <div class="kpi-sub">Scale: 0.00 – 20.00</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-label">Projected Pass Rate</div>
                    <div class="kpi-number cream">{summary['overall_pass_rate_pct']}%</div>
                    <div class="kpi-sub">Threshold: >= 10.00</div>
                </div>
                """, unsafe_allow_html=True)
            with b4:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-label">At-Risk Count</div>
                    <div class="kpi-number crimson">{summary['high_risk_count']}</div>
                    <div class="kpi-sub">{summary['high_risk_pct']}% of Total Cohort</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

            # Twin Distribution Visualizations
            chart_col1, chart_col2 = st.columns(2, gap="medium")
            with chart_col1:
                st.markdown('<div class="card-title">Risk Stratification Breakdown</div>', unsafe_allow_html=True)
                risk_counts = report_df["risk_level"].value_counts()
                tier_colors = []
                for label in risk_counts.index:
                    if label == "High Risk": tier_colors.append("#E63946")
                    elif label == "Moderate Risk": tier_colors.append("#F4D6B0")
                    else: tier_colors.append("#5BC0DE")

                fig_donut = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values,
                    hole=0.6, marker=dict(colors=tier_colors),
                    textinfo="label+percent", textfont=dict(family="Space Grotesk", size=11, color="#FFFFFF")
                )])
                fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=250)
                st.plotly_chart(fig_donut, use_container_width=True)

            with chart_col2:
                st.markdown('<div class="card-title">Forecast Grade Distribution (0-20)</div>', unsafe_allow_html=True)
                fig_hist = px.histogram(report_df, x="predicted_G3", nbins=15, color_discrete_sequence=["#5BC0DE"])
                fig_hist.add_vline(x=10.0, line_dash="dot", line_color="#F4D6B0", annotation_text="Pass Benchmark (10.0)", annotation_font_color="#F4D6B0")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=250,
                    xaxis=dict(title="Score (0-20)", gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
                    yaxis=dict(title="Student Frequency", gridcolor="#1C2541", tickfont=dict(family="JetBrains Mono", color="#94A3B8"))
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown('<div class="card-title">Stratified Student Roster</div>', unsafe_allow_html=True)
            st.dataframe(
                report_df[["age", "studytime", "failures", "absences", "G1", "G2", "predicted_G3", "risk_level"]],
                use_container_width=True
            )

            csv_export = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="EXPORT ANNOTATED COHORT REPORT (CSV)",
                data=csv_export,
                file_name="academiq_cohort_risk_report.csv",
                mime="text/csv"
            )
