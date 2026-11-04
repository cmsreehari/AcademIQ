"""
Pytest fixtures and mock datasets for StudyFlow test suite.
"""

import pytest
import pandas as pd
from studyflow.schemas import StudentInput


@pytest.fixture
def sample_valid_student() -> StudentInput:
    """Returns a valid StudentInput dataclass."""
    return StudentInput(
        age=17,
        studytime=3,
        failures=0,
        absences=2,
        G1=14.0,
        G2=15.0,
        higher="yes",
        Medu=3,
        Fedu=3,
        schoolsup="no"
    )


@pytest.fixture
def sample_student_dataframe() -> pd.DataFrame:
    """Returns a minimal valid DataFrame of students."""
    return pd.DataFrame([
        {
            "age": 17, "studytime": 3, "failures": 0, "absences": 2,
            "G1": 14.0, "G2": 15.0, "higher": "yes", "Medu": 3, "Fedu": 3, "schoolsup": "no"
        },
        {
            "age": 18, "studytime": 1, "failures": 2, "absences": 10,
            "G1": 6.0, "G2": 7.0, "higher": "no", "Medu": 1, "Fedu": 1, "schoolsup": "yes"
        }
    ])