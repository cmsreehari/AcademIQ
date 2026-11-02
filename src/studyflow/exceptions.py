''' 
This is a custom error handilig section for out application.
Provides a clear error hierarchy for ML inference, data validation, and database operations.
'''


class StudyFlowError(Exception):
    """Base class for all StudyFlow exceptions."""
    pass

# Machine Learning & Inference Errors
# -------------------------------------------------------------
class ModelLoadError(StudyFlowError):
    """Raised when the trained ML model or feature list cannot be loaded."""
    pass
class FeatureMismatchError(StudyFlowError):
    """Raised when the input features do not match what the ML model expects."""
    pass
class PredictionError(StudyFlowError):
    """Raised when model inference fails during runtime."""
    pass

# Data Validation & Storage Errors
# -------------------------------------------------------------
class InvalidGradeError(StudyFlowError):
    """Raised when a grade is outside the valid range [0.0, 20.0]."""
    pass
class StudentNotFoundError(StudyFlowError):
    """Raised when a requested student record does not exist in the database."""
    pass



'''(.venv) PS E:\AiMl_mentoring> pip install -e .
Obtaining file:///E:/AiMl_mentoring
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: pandas in .\.venv\Lib\site-packages (from studyflow==0.1.0) (3.0.5)
Requirement already satisfied: numpy in .\.venv\Lib\site-packages (from studyflow==0.1.0) (2.5.2)
Requirement already satisfied: scikit-learn in .\.venv\Lib\site-packages (from studyflow==0.1.0) (1.9.0)
Requirement already satisfied: joblib in .\.venv\Lib\site-packages (from studyflow==0.1.0) (1.5.3)
Requirement already satisfied: streamlit in .\.venv\Lib\site-packages (from studyflow==0.1.0) (1.66.1)
Requirement already satisfied: pytest in .\.venv\Lib\site-packages (from studyflow==0.1.0) (9.2.0)
Requirement already satisfied: python-dateutil>=2.8.2 in .\.venv\Lib\site-packages (from pandas->studyflow==0.1.0) (2.9.0.post0)
Requirement already satisfied: tzdata in .\.venv\Lib\site-packages (from pandas->studyflow==0.1.0) (2026.3)
Requirement already satisfied: six>=1.5 in .\.venv\Lib\site-packages (from python-dateutil>=2.8.2->pandas->studyflow==0.1.0) (1.17.0)
Requirement already satisfied: colorama>=0.4 in .\.venv\Lib\site-packages (from pytest->studyflow==0.1.0) (0.4.6)
Requirement already satisfied: iniconfig>=2 in .\.venv\Lib\site-packages (from pytest->studyflow==0.1.0) (2.3.1)
Requirement already satisfied: packaging>=24 in .\.venv\Lib\site-packages (from pytest->studyflow==0.1.0) (26.3)
Requirement already satisfied: pluggy<2,>=1.6 in .\.venv\Lib\site-packages (from pytest->studyflow==0.1.0) (1.7.0)
Requirement already satisfied: pygments>=2.15 in .\.venv\Lib\site-packages (from pytest->studyflow==0.1.0) (2.21.0)
Requirement already satisfied: scipy>=1.10.0 in .\.venv\Lib\site-packages (from scikit-learn->studyflow==0.1.0) (1.18.1)
Requirement already satisfied: narwhals>=2.0.1 in .\.venv\Lib\site-packages (from scikit-learn->studyflow==0.1.0) (2.25.0)
Requirement already satisfied: threadpoolctl>=3.5.0 in .\.venv\Lib\site-packages (from scikit-learn->studyflow==0.1.0) (3.6.0)
Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (6.3.0)
Requirement already satisfied: click<9,>=7.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (8.5.0)
Requirement already satisfied: pillow<14,>=9.2.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (12.3.0)
Requirement already satisfied: pydeck<1,>=0.8.0b4 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (0.9.3)
Requirement already satisfied: protobuf<8,>=5.26.1 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (7.36.2)
Requirement already satisfied: pyarrow!=25.0.0,<27,>=10.0.1 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (26.0.0)
Requirement already satisfied: requests<3,>=2.27 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (2.34.2)
Requirement already satisfied: typing-extensions<5,>=4.10.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (4.16.0)
Requirement already satisfied: starlette<2,>=0.46.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (1.10.0)
Requirement already satisfied: uvicorn<1,>=0.30.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (0.54.0)
Requirement already satisfied: anyio<5,>=4.0.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (4.16.0)
Requirement already satisfied: python-multipart<1,>=0.0.10 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (0.0.33)
Requirement already satisfied: websockets<18,>=12.0.0 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (17.4)
Requirement already satisfied: itsdangerous<3,>=2.1.2 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (2.2.0)
Requirement already satisfied: watchdog<7,>=2.1.5 in .\.venv\Lib\site-packages (from streamlit->studyflow==0.1.0) (6.0.0)
Requirement already satisfied: jinja2 in .\.venv\Lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (3.1.6)
Requirement already satisfied: jsonschema>=3.0 in .\.venv\Lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (4.26.0)
Requirement already satisfied: idna>=2.8 in .\.venv\Lib\site-packages (from anyio<5,>=4.0.0->streamlit->studyflow==0.1.0) (3.20)
Requirement already satisfied: charset_normalizer<4,>=2 in .\.venv\Lib\site-packages (from requests<3,>=2.27->streamlit->studyflow==0.1.0) (3.5.2)
Requirement already satisfied: urllib3<3,>=1.26 in .\.venv\Lib\site-packages (from requests<3,>=2.27->streamlit->studyflow==0.1.0) (2.8.0)
Requirement already satisfied: certifi>=2023.5.7 in .\.venv\Lib\site-packages (from requests<3,>=2.27->streamlit->studyflow==0.1.0) (2026.10.23)
Requirement already satisfied: h11>=0.8 in .\.venv\Lib\site-packages (from uvicorn<1,>=0.30.0->streamlit->studyflow==0.1.0) (0.16.0)
Requirement already satisfied: MarkupSafe>=2.0 in .\.venv\Lib\site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (3.0.4)
Requirement already satisfied: attrs>=22.2.0 in .\.venv\Lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (26.1.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in .\.venv\Lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in .\.venv\Lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in .\.venv\Lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit->studyflow==0.1.0) (2026.9.1)
Building wheels for collected packages: studyflow
  Building editable for studyflow (pyproject.toml) ... done
  Created wheel for studyflow: filename=studyflow-0.1.0-0.editable-py3-none-any.whl size=4640 sha256=be7806ec8779141c261701defb52779205ef185490e3b88499949c4c9df0e17a
  Stored in directory: C:\Users\Sreehari\AppData\Local\Temp\pip-ephem-wheel-cache-84eol4gy\wheels\32\f0\bd\b793cd3e56d673dd32a78f26d3654273025b287e4a8a5af793
Successfully built studyflow
Installing collected packages: studyflow
Successfully installed studyflow-0.1.0

[notice] A new release of pip is available: 26.2.1 -> 26.2.2
[notice] To update, run: python.exe -m pip install --upgrade pip
(.venv) PS E:\AiMl_mentoring> ^C
(.venv) PS E:\AiMl_mentoring> ^C
(.venv) PS E:\AiMl_mentoring> '''