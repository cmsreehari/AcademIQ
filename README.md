# StudyFlow — Student Academic Performance & Early-Warning Platform

An end-to-end Machine Learning web application and modular Python system designed to analyze secondary school student academic performance, forecast final exam grades (`G3`), flag dropout and failure risks early, and provide actionable educational recommendations.

---

## 📌 Project Overview
- **Core Mission**: Move beyond post-exam grading by providing an **early-intervention advisory platform** for teachers, students, and parents.
- **Dataset**: UCI Machine Learning Repository — *Student Performance Dataset* (`student-mat.csv`).
- **Population**: 395 secondary school students across 33 academic, demographic, and socioeconomic features.
- **Target Variable**: `G3` (Final Mathematics Exam Grade, scale $0–20$, passing mark $\ge 10$).
- **Key Discovery**: Identified the **Structural Zero Dropout Phenomenon** (38 students scored $G3 = 0$ despite passing earlier terms, representing sudden withdrawal or exam absenteeism rather than gradual decline).

---

## 📁 Repository & Architecture Structure

```text
AiMl_mentoring/
│
├── pyproject.toml              # Build & package config (enables pip install -e .)
├── requirements.txt            # Dependency specification
├── README.md                   # Complete project documentation & guide
├── student_note.ipynb          # Original Jupyter Notebook (EDA, ML benchmarking)
│
├── data/                       # Benchmark data storage
│   ├── student-mat.csv         # UCI Mathematics benchmark dataset (semicolon-separated)
│   ├── student-por.csv         # Secondary Portuguese language dataset
│   └── student.txt             # Attribute definitions and codebook
│
├── models/                     # Serialized machine learning artifacts
│   ├── best_student_model.joblib # Trained Random Forest Regressor (R² = 0.815, MAE = 1.16)
│   └── model_features.joblib   # 56 aligned feature names for inference
│
├── src/studyflow/              # Core modular Python package
│   ├── __init__.py
│   ├── config.py               # OS-agnostic dynamic path resolution & domain constants
│   ├── exceptions.py           # Domain error hierarchy (StudyFlowError, DataValidationError)
│   ├── schemas.py              # Typed dataclass contracts (StudentInput, PredictionOutput)
│   │
│   ├── data/                   # Data ingestion and verification
│   │   ├── __init__.py
│   │   ├── loader.py           # Delimiter-aware CSV loader (sep=";")
│   │   └── validator.py        # Defensive schema and range validation (0-20 scale)
│   │
│   ├── features/               # Feature engineering & preprocessing
│   │   ├── __init__.py
│   │   └── engineer.py         # Academic momentum (grade_change = G2 - G1) & 56-feature alignment
│   │
│   ├── models/                 # Predictive inference engine
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract BasePredictor interface (Open-Closed Principle)
│   │   └── forest.py           # Concrete Random Forest wrapper, bounding & risk banding
│   │
│   └── reporting/              # Batch classroom analytics
│       ├── __init__.py
│       └── reporter.py         # Cohort health summarization & annotated CSV export
│
├── app/
│   └── streamlit_app.py        # Dual-Mode Interactive Web Application (Streamlit)
│
├── tests/                      # Automated unit and integration test suite
│   ├── conftest.py             # Pytest fixtures and mock student data
│   └── test_studyflow.py       # Comprehensive test cases for validator, features, ML & reports
│
└── notes/                      # Research deep-dive & evaluation documentation
    ├── personal_researchNotes.txt
    ├── research_notes.md
    └── PROJECT_CONTEXT.md
```

---

## ⚙️ Quick Start & Installation

### 1. Activate Virtual Environment
```powershell
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 2. Install Package in Editable Mode
```powershell
pip install -e .
```
*(This makes the `studyflow` library available everywhere in your environment without needing complex path hacks).*

### 3. Launch the Web Application
```powershell
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`** to interact with the dashboard!

### 4. Run Automated Tests
```powershell
pytest -v
```

---

## 🖥️ Application Features & User Modes

The Streamlit web application provides **two complementary modes**:

### Mode 1: 👤 Single Student Performance Forecast
- **Interactive Sliders**: Adjust Term 1 ($G1$), Term 2 ($G2$), weekly study hours, past failures, and absences.
- **Real-Time KPIs**: Displays predicted final grade ($G3$), passing status, risk category, and academic momentum.
- **Academic Trajectory Chart**: Visualizes progress across $G1 \to G2 \to \text{Predicted } G3$.
- **Early-Warning Advisory**: Generates personalized, empathetic guidance for educators and parents.

### Mode 2: 📁 Classroom Batch Upload & Analytics
- **Drag-and-Drop CSV Uploader**: Accepts any standard classroom CSV file (or 1-click loads the benchmark).
- **Automated Validation**: `validator.py` checks for column presence, data types, and value boundaries before processing.
- **Cohort Health Cards**: Instant class-wide metrics (cohort size, average predicted mark, pass rate, and high-risk count).
- **Risk Distribution Charts**: Donut and histogram charts breaking down students into *On Track*, *Moderate Risk*, and *High Risk*.
- **Annotated Export**: Download the enriched CSV report with predicted marks and risk bands for administrative records.

---

## 🧠 Machine Learning Insights & Model Evaluation

### Benchmark Comparison

$$\text{Random Forest } (R^2 = 0.815, \text{ MAE} = 1.16) \quad \text{vs.} \quad \text{Linear Regression } (R^2 = 0.724, \text{ MAE} = 1.65)$$

1. **Why Random Forest Outperformed Linear Regression**:
   - Academic risk is inherently **non-linear**: a student with $\ge 1$ failure experiences a sharp drop, whereas additional study hours plateau in benefit after 10 hours. Decision tree splits naturally capture these non-linear thresholds.
   - Captures **feature interactions** (e.g., *high absences + high study time* vs. *high absences + low study time*) without manual polynomial terms.
2. **Key Feature Importances**:
   - $G2$ (Second Term Mark) accounted for $>70\%$ of predictive power.
   - $G1$ (First Term Mark) provided early baseline trajectory.
   - `failures` (Previous class failures) was the strongest negative risk indicator ($r = -0.36$).
   - `studytime` and parental education (`Medu`, `Fedu`) provided positive protective momentum.
3. **Domain-Engineered Momentum**:
   - By engineering $\text{grade\_change} = G2 - G1$, the model distinguishes between a student improving from $10 \to 14$ versus a student collapsing from $14 \to 10$, even though both share the same average mark.

---

## 🛡️ Software Engineering Best Practices Applied

- **Abstract Base Classes (`BasePredictor`)**: Decouples the application from a specific ML algorithm, adhering to the **Open-Closed Principle**.
- **Defensive Data Validation**: Catches out-of-range marks, negative values, and missing fields before model inference.
- **Typed Data Contracts (`dataclasses`)**: Replaced fragile dictionary keys with type-safe `StudentInput` and `PredictionOutput` schemas.
- **OS-Agnostic Path Management**: Centralized in `config.py` using `pathlib.Path` to prevent broken paths across Windows, Mac, and Linux.
- **In-Memory Caching (`@st.cache_resource`)**: Loads ML weights into RAM once, guaranteeing sub-5ms UI responsiveness.
- **Automated Testing Suite**: 100% passing `pytest` test suite covering validation, feature transformation, inference, and batch reporting.

---

## 📜 Ethical Considerations
- **No Deficit Framing**: Predictions are intended solely to allocate **supportive resources** (remedial tutoring, study spaces, mentoring), never to punish or track students into lower educational tiers.
- **Human-in-the-Loop**: The model serves strictly as an **advisory decision-support tool** for educators, never an automated grading or disciplinary mechanism.
