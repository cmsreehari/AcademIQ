"""
Data Schemas and Typed Data Transfer Objects for StudyFlow.
Defines structured classes for student input data and ML prediction results.

This avoids the confusion of using untyped dictionaries and provides clear type hints for better code clarity and validation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StudentInput:
    """
    Structured representation of student attributes required for risk analysis and ML prediction.
    """
    age: int
    studytime: int          # Weekly study time: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
    failures: int           # Number of past class failures (0 to 3, else 4)
    absences: int           # Number of school absences (0 to 93)
    G1: float               # First term grade (0 to 20)
    G2: float               # Second term grade (0 to 20)
    higher: str = "yes"     # Wants to take higher education ("yes" or "no")
    Medu: int = 2           # Mother's education (0: none, 1: primary, 2: 5th-9th, 3: secondary, 4: higher)
    Fedu: int = 2           # Father's education (0: none, 1: primary, 2: 5th-9th, 3: secondary, 4: higher)
    schoolsup: str = "no"   # Extra educational support ("yes" or "no")


@dataclass
class PredictionOutput:
    """
    Structured result returned after running ML inference on a student.
    """
    predicted_grade: float  # Predicted G3 grade (0.0 to 20.0)
    is_at_risk: bool        # True if predicted_grade < DROPOUT_RISK_THRESHOLD
    risk_level: str         # "High Risk", "Moderate Risk", or "On Track"
    passed: bool            # True if predicted_grade >= PASSING_GRADE
    summary_message: str    # Human-readable advisory note for teachers/parents