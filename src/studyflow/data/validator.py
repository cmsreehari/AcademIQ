"""
Data validation rules and schema verification for StudyFlow.
Guarantees clean inputs before feature engineering and ML inference.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd

from studyflow import config
from studyflow.exceptions import DataValidationError

# Essential columns required for student analysis and prediction
REQUIRED_COLUMNS = [
    "age", "studytime", "failures", "absences",
    "G1", "G2", "higher", "Medu", "Fedu"
]


def validate_student_record(data: Dict[str, Any]) -> List[str]:
    """
    Validates a single student record dictionary.

    Returns:
    --------
    List[str]
        A list of error messages. If empty, the record is valid.
    """
    errors: List[str] = []

    # 1. Check required fields
    for col in REQUIRED_COLUMNS:
        if col not in data:
            errors.append(f"Missing required field: '{col}'")

    if errors:
        return errors  # Cannot validate values if fields are missing

    # 2. Check Grade Ranges (0.0 to 20.0)
    for grade_field in ["G1", "G2"]:
        val = data[grade_field]
        if not isinstance(val, (int, float)):
            errors.append(f"'{grade_field}' must be a number, got {type(val).__name__}")
        elif not (config.MIN_GRADE <= val <= config.MAX_GRADE):
            errors.append(f"'{grade_field}' ({val}) is outside valid range [{config.MIN_GRADE}, {config.MAX_GRADE}]")

    # 3. Check Absences (non-negative)
    absences = data.get("absences", 0)
    if not isinstance(absences, int) or absences < 0:
        errors.append(f"'absences' must be an integer >= 0, got {absences}")

    # 4. Check Study Time (1 to 4)
    studytime = data.get("studytime", 1)
    if studytime not in [1, 2, 3, 4]:
        errors.append(f"'studytime' must be 1, 2, 3, or 4, got {studytime}")

    # 5. Check Age
    age = data.get("age", 18)
    if not (15 <= age <= 22):
        errors.append(f"'age' ({age}) must be between 15 and 22")

    return errors


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates an entire pandas DataFrame of student records.

    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"DataFrame is missing required columns: {missing_cols}")
        return False, errors

    # Check for empty dataframe
    if df.empty:
        errors.append("DataFrame is empty (0 rows).")
        return False, errors

    # Check Grade ranges across whole columns
    for grade_col in ["G1", "G2"]:
        invalid_mask = (df[grade_col] < config.MIN_GRADE) | (df[grade_col] > config.MAX_GRADE)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            errors.append(f"Column '{grade_col}' has {invalid_count} values outside [{config.MIN_GRADE}, {config.MAX_GRADE}]")

    # Check Absences
    neg_absences = (df["absences"] < 0).sum()
    if neg_absences > 0:
        errors.append(f"Column 'absences' has {neg_absences} negative values.")

    is_valid = len(errors) == 0
    return is_valid, errors