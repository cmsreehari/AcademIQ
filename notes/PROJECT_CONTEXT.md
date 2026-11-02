# PROJECT CONTEXT & MASTER HANDOVER
**Project**: StudyFlow — Student Risk Report Web Application  
**Target Deadline**: 2-Day Sprint  
**Assessment**: Capstone Project (100 Marks Total — Evaluated on Python Engineering, Architecture, Testing, and Research)  
**Role Protocol**: The USER writes all code. The AI functions strictly as a senior mentor, code reviewer, and technical sounding board.

---

## 1. Executive Summary & Purpose
StudyFlow is a production-grade **Streamlit web application** and modular Python package (`src/studyflow/`) designed for educators. 
A teacher uploads a CSV of student records; the application validates schema and values, predicts each student's final mathematics grade ($G3$, scale $0–20$) using a trained Random Forest pipeline, classifies students as **"At Risk"** ($G3 < 10$) or **"On Track"** ($G3 \ge 10$), and exports an actionable diagnostic report.

---

## 2. Dataset & Historical Findings (From Phase 1)
* **Citation**: Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*.
* **Dataset**: UCI Machine Learning Repository — Portuguese Secondary Mathematics ([`student-mat.csv`](file:///E:/AiMl_mentoring/student-mat.csv)), semicolon-separated (`;`), 395 rows, 33 attributes.
* **Target Variable**: $G3$ (Final exam mark, $0–20$, passing mark $\ge 10$).
* **Critical Empirical Insights**:
  1. **Academic Momentum**: $G2$ ($r = 0.90$) and $G1$ ($r = 0.80$) are the strongest predictors ($G2$ accounts for $>70\%$ feature importance in decision trees).
  2. **Structural Zero ($G3 = 0$)**: 38 students scored exactly 0 on the final exam despite passing $G1$ and $G2$ ($8–14$). This reflects exam absenteeism, dropouts, or administrative issues rather than cognitive failure.
  3. **Parental Education Artifact**: Level 0 ("None") for `Medu` and `Fedu` showed misleadingly high median grades due to tiny sample sizes ($N=3$ for mother, $N=2$ for father), while the general population ($N=390$) shows a steady positive socioeconomic gradient.
  4. **Non-Linear Thresholds**: 1+ prior failures (`failures`, $r = -0.36$) causes a sharp drop in median score (from 11.5 down to 9.0).

---

## 3. Two-Day Scope & Boundaries
### ✅ What Will Be Built:
* **Data Processing & Defensive Validation**: Strict schema checking, range checks ($G1, G2 \in [0, 20]$, $\text{studytime} \in [1, 4]$), row/column error reporting.
* **Feature Engineering**: Domain feature builder (`grade_change = G2 - G1`).
* **Scikit-Learn Pipeline**: Integrated `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")` + `RandomForestRegressor`.
* **5 Cross-Validated Baselines (5-Fold CV)**:
  1. Mean Baseline
  2. $G3 = G2$ Baseline
  3. Linear Regression
  4. Random Forest Regressor
  5. Random Forest without $G1/G2$ (Honest Early-Warning model)
* **Dual Metric Reporting**: Metrics reported both **with** and **without** the $G3=0$ rows.
* **Thin Streamlit Web App (`app/streamlit_app.py`)**:
  - Page 1: Upload CSV, view validation errors, view risk table, download report.
  - Page 2: Single-student prediction form.
  - Page 3: Model evidence (comparison table & feature importance).
  - Page 4: About, ethical considerations, and limitations.
* **Automated Testing**: Comprehensive `pytest` suite for validator, features, predictor, risk labeling, and report builder.

### 🚫 Explicit Non-Goals (Documented as "Future Improvements"):
* No AI tutor / LLM chatbot.
* No SQLite / persistent database.
* No multi-grade time-series trend tracking (the dataset only has 1 row per student).
* No FastAPI backend (the web app is a thin UI calling the `studyflow` package).

---

## 4. Architectural Layout
```text
studyflow/
│
├── src/studyflow/
│   ├── __init__.py
│   ├── config.py              # Thresholds (PASSING_THRESHOLD=10), paths, seed=42
│   ├── exceptions.py          # StudyFlowError, DataValidationError, ModelInferenceError
│   ├── schemas.py             # Typed StudentRecord dataclass & validation rules
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # CSV loader (sep=";")
│   │   └── validator.py       # Validates columns, types, and ranges; returns structured errors
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py        # grade_change = G2 - G1 and interaction builders
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract BasePredictor (ABC)
│   │   ├── forest.py          # Concrete Random Forest implementation
│   │   └── train.py           # CLI training script: 5-fold CV, baselines, saves artifact
│   │
│   └── reporting/
│       ├── __init__.py
│       └── reporter.py        # Risk band assignment ("At Risk" vs "On Track"), CSV exporter
│
├── app/
│   └── streamlit_app.py       # Thin Streamlit frontend (no business logic)
│
├── tests/
│   ├── conftest.py            # Pytest fixtures (sample CSVs, mock rows)
│   ├── test_validator.py      # Tests valid/invalid CSV rows and edge cases
│   ├── test_features.py       # Tests grade_change calculation
│   ├── test_predictor.py      # Tests inference and model missing error
│   ├── test_reporting.py      # Tests risk tagging and report generation
│   └── test_end_to_end.py     # End-to-end integration test with miniature CSV
│
├── data/                      # Raw student-mat.csv
├── outputs/                   # Generated charts & reports
├── logs/                      # Application runtime logs
│
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Ignores models/*.joblib, logs/, data/*.db, .venv/
├── README.md                  # Comprehensive setup, run, test, and architecture docs
└── research_notes.md          # 5-question technical defense for all core decisions
```

---

## 5. Evaluation Rubric & Mark Distribution (100 Marks)
1. **Python Engineering & Code Quality (20 Marks)**: Type hints, dataclasses, custom exceptions, ABC, logging, `pathlib`.
2. **Project Architecture & Modularity (15 Marks)**: Clean separation of UI and library, single-responsibility modules.
3. **ML / Data Science Implementation (15 Marks)**: Pipeline training, 5-fold CV across 5 baselines, feature importance, metrics with/without zeros.
4. **Testing, Reliability & Reproducibility (10 Marks)**: `pytest` on validator, features, predictor, reporter; clean venv reproducibility.
5. **Technical Research & Reasoning (10 Marks)**: `research_notes.md` justifying One-Hot vs Label, Pipeline vs `get_dummies`, Medu sampling artifact, $G3=G2$ baseline importance.
6. **Problem Understanding & Solution Design (10 Marks)**: Clear scope, realistic teacher workflow.
7. **Documentation (5 Marks)**: Complete `README.md` with setup, usage, screenshots, limitations.
8. **Viva Defense / Explanation (5 Marks)**: Student's ability to explain and defend every line of code.

---

## 6. How to Resume in a New Chat
In the new chat, paste this prompt:
> **"Read `PROJECT_CONTEXT.md` to load the full history, dataset findings, and our finalized 2-day plan for StudyFlow. We are starting Day 1: Folder skeleton and Validator design. Remember: I will write the code; you are my mentor and reviewer."**
