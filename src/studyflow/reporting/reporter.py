"""
Batch reporting and academic risk aggregation for StudyFlow.
Generates classroom-level risk summaries and exports annotated datasets.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from studyflow import config
from studyflow.exceptions import StudyFlowError
from studyflow.models.base import BasePredictor
from studyflow.features.engineer import prepare_features_for_inference


def generate_batch_report(
    df: pd.DataFrame, 
    predictor: BasePredictor
) -> pd.DataFrame:
    """
    Takes a raw or loaded student DataFrame, computes predictions for all rows,
    and returns an annotated DataFrame with risk levels and predicted grades.

    Parameters:
    -----------
    df : pd.DataFrame
        Classroom dataset (must contain required student features).
    predictor : BasePredictor
        An instantiated and loaded StudyFlow model predictor.

    Returns:
    --------
    pd.DataFrame
        Copy of the original DataFrame with predicted_G3 and risk_level added.
    """
    if df.empty:
        raise StudyFlowError("Cannot generate report on an empty DataFrame.")

    report_df = df.copy()

    # 1. Align features for the model
    features_df = prepare_features_for_inference(report_df, predictor.features)

    # 2. Batch predict
    predictions = predictor.predict(features_df)
    report_df["predicted_G3"] = np.round(predictions, 2)

    # 3. Assign Risk Categories
    def assign_risk(score: float) -> str:
        if score < config.DROPOUT_RISK_THRESHOLD:
            return "High Risk"
        elif score < 12.0:
            return "Moderate Risk"
        return "On Track"

    report_df["risk_level"] = report_df["predicted_G3"].apply(assign_risk)
    report_df["is_at_risk"] = report_df["predicted_G3"] < config.DROPOUT_RISK_THRESHOLD
    report_df["passed"] = report_df["predicted_G3"] >= config.PASSING_GRADE

    return report_df


def summarize_class_health(report_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level aggregated summary statistics for a processed classroom.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with total count, risk breakdown, and average predicted score.
    """
    if "predicted_G3" not in report_df.columns:
        raise StudyFlowError("DataFrame must be processed by generate_batch_report first.")

    total_students = len(report_df)
    high_risk_count = int((report_df["risk_level"] == "High Risk").sum())
    moderate_risk_count = int((report_df["risk_level"] == "Moderate Risk").sum())
    on_track_count = int((report_df["risk_level"] == "On Track").sum())

    avg_predicted = round(float(report_df["predicted_G3"].mean()), 2)
    pass_rate = round(float((report_df["passed"].sum() / total_students) * 100), 1)

    return {
        "total_students": total_students,
        "avg_predicted_grade": avg_predicted,
        "overall_pass_rate_pct": pass_rate,
        "high_risk_count": high_risk_count,
        "high_risk_pct": round((high_risk_count / total_students) * 100, 1),
        "moderate_risk_count": moderate_risk_count,
        "moderate_risk_pct": round((moderate_risk_count / total_students) * 100, 1),
        "on_track_count": on_track_count,
        "on_track_pct": round((on_track_count / total_students) * 100, 1),
    }


def export_report_to_csv(report_df: pd.DataFrame, output_path: Path) -> Path:
    """
    Saves the enriched report DataFrame to a CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    return output_path 