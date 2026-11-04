"""
Automated unit and integration test suite for StudyFlow.
Verifies data validation, feature engineering, ML prediction, and batch reporting.
"""

import pytest
from studyflow import config
from studyflow.data.validator import validate_student_record, validate_dataframe
from studyflow.features.engineer import add_domain_features
from studyflow.models.forest import RandomForestStudentPredictor
from studyflow.reporting.reporter import generate_batch_report, summarize_class_health


def test_validator_detects_out_of_range_grade():
    """Ensures validator rejects grades outside 0-20 scale."""
    invalid_record = {
        "age": 17, "studytime": 2, "failures": 0, "absences": 4,
        "G1": 25.0,  # Invalid grade (> 20)
        "G2": 12.0, "higher": "yes", "Medu": 2, "Fedu": 2, "schoolsup": "no"
    }
    errors = validate_student_record(invalid_record)
    assert any("outside valid range" in err for err in errors)


def test_validator_detects_negative_absences():
    """Ensures validator rejects negative absences."""
    invalid_record = {
        "age": 17, "studytime": 2, "failures": 0,
        "absences": -5,  # Invalid negative
        "G1": 12.0, "G2": 12.0, "higher": "yes", "Medu": 2, "Fedu": 2, "schoolsup": "no"
    }
    errors = validate_student_record(invalid_record)
    assert any("must be an integer >= 0" in err for err in errors)


def test_feature_engineering_grade_change(sample_student_dataframe):
    """Verifies that grade_change = G2 - G1 is computed correctly."""
    df_feat = add_domain_features(sample_student_dataframe)
    assert "grade_change" in df_feat.columns
    # Student 0: G1=14, G2=15 -> grade_change = +1.0
    assert df_feat.loc[0, "grade_change"] == 1.0
    # Student 1: G1=6, G2=7 -> grade_change = +1.0
    assert df_feat.loc[1, "grade_change"] == 1.0


def test_model_inference_clamping_and_risk(sample_valid_student):
    """Verifies that the predictor produces grades within [0, 20] and valid risk bands."""
    predictor = RandomForestStudentPredictor()
    output = predictor.predict_student(sample_valid_student)

    assert config.MIN_GRADE <= output.predicted_grade <= config.MAX_GRADE
    assert output.risk_level in ["High Risk", "Moderate Risk", "On Track"]
    assert isinstance(output.passed, bool)


def test_batch_reporting_pipeline(sample_student_dataframe):
    """Verifies end-to-end batch prediction and classroom aggregation."""
    predictor = RandomForestStudentPredictor()
    report = generate_batch_report(sample_student_dataframe, predictor)

    assert "predicted_G3" in report.columns
    assert "risk_level" in report.columns

    summary = summarize_class_health(report)
    assert summary["total_students"] == 2
    assert "avg_predicted_grade" in summary
    assert "overall_pass_rate_pct" in summary