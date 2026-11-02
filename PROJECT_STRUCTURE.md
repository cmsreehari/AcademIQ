# StudyFlow — Project Structure Reference (Current 2-Day Plan)

Use this reference as the source of truth for creating and navigating your project directories and files.

```text
AiMl_mentoring/
│
├── src/studyflow/                 # Core modular Python package
│   ├── __init__.py
│   ├── config.py                  # Thresholds (PASSING_THRESHOLD=10), paths, seed=42
│   ├── exceptions.py              # StudyFlowError, DataValidationError, ModelInferenceError
│   ├── schemas.py                 # Typed dataclasses & validation rule definitions
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Robust CSV loader (sep=";")
│   │   └── validator.py           # Validates columns, types, and ranges; returns structured errors
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py            # grade_change = G2 - G1 and domain feature builders
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract BasePredictor (ABC)
│   │   ├── forest.py              # Concrete Random Forest implementation
│   │   └── train.py               # CLI training script: 5-fold CV, 5 baselines, saves artifact
│   │
│   └── reporting/
│       ├── __init__.py
│       └── reporter.py            # Risk band assignment ("At Risk" vs "On Track"), CSV exporter
│
├── app/
│   └── streamlit_app.py           # Thin Streamlit frontend (no business logic)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures (sample CSVs, mock rows)
│   ├── test_validator.py          # Tests valid/invalid CSV rows and edge cases
│   ├── test_features.py           # Tests grade_change calculation
│   ├── test_predictor.py          # Tests inference and model missing error
│   ├── test_reporting.py          # Tests risk tagging and report generation
│   └── test_end_to_end.py         # End-to-end integration test with miniature CSV
│
├── data/                          # Raw datasets (student-mat.csv, student-por.csv, student.txt)
├── models/                        # Serialized .joblib model artifacts
├── outputs/                       # Generated charts & exported reports
├── logs/                          # Application runtime logs
│
├── pyproject.toml                 # Package configuration (enables pip install -e .)
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Ignores models/*.joblib, logs/, .venv/
├── README.md                      # Comprehensive setup, run, test, and architecture docs
├── research_notes.md              # 5-question technical defense for all core decisions
└── PROJECT_CONTEXT.md             # Master project context and sprint tracking
```

---

## Key Changes from Previous Draft
1. **Removed Out-of-Scope Components**:
   - `storage/` & `studyflow.db` (removed: project is stateless in-memory).
   - `quiz/` & `tutor/` (removed: no AI tutor or quiz generator in 2-day scope).
   - `analytics/` (removed: no multi-term time-series tracking; single-row tabular predictions).
2. **Aligned with 2-Day Scope**:
   - Added explicit `schemas.py` and `exceptions.py` under `src/studyflow/`.
   - Added `loader.py` and `validator.py` under `src/studyflow/data/`.
   - Added `train.py` under `src/studyflow/models/` for 5-fold CV and baseline benchmarking.
   - Standardized `app/streamlit_app.py` as a thin UI layer over the package.
