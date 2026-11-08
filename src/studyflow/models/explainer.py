"""
Explainable AI (XAI) engine for AcademIQ using TreeSHAP.
Computes local feature contributions to decompose individual predictions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import shap

from studyflow.schemas import StudentInput
from studyflow.models.forest import RandomForestStudentPredictor
from studyflow.features.engineer import prepare_features_for_inference


@dataclass
class ExplanationResult:
    """
    Structured feature attribution outcome for a student.
    """
    base_value: float                      # Average population score (~10.4)
    predicted_value: float                 # Model's output grade
    top_positive_factors: List[Tuple[str, float]] # Features pulling score UP
    top_negative_factors: List[Tuple[str, float]] # Features pulling score DOWN
    all_contributions: Dict[str, float]    # Full dictionary of feature impacts


class StudentModelExplainer:
    """
    TreeSHAP explainer wrapper for RandomForestStudentPredictor.
    """

    def __init__(self, predictor: Optional[RandomForestStudentPredictor] = None):
        self.predictor = predictor or RandomForestStudentPredictor()
        # Initialize TreeExplainer on trained Random Forest
        self.explainer = shap.TreeExplainer(self.predictor.model)
        self.expected_value = float(self.explainer.expected_value[0]) if isinstance(
            self.explainer.expected_value, (list, np.ndarray)
        ) else float(self.explainer.expected_value)

    def explain_student(self, student: StudentInput, top_n: int = 5) -> ExplanationResult:
        """
        Decomposes a single student's prediction into exact point-by-point contributions.

        Parameters:
        -----------
        student : StudentInput
            The student profile to explain.
        top_n : int
            Number of top positive and negative features to extract.

        Returns:
        --------
        ExplanationResult
        """
        # 1. Transform student into aligned 56-feature vector
        raw_df = pd.DataFrame([student.__dict__])
        features_df = prepare_features_for_inference(raw_df, self.predictor.features)

        # 2. Compute SHAP values
        shap_values = self.explainer.shap_values(features_df)
        shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Flatten if 2D array
        if len(shap_array.shape) > 1:
            shap_array = shap_array[0]

        # 3. Map contributions to human-friendly feature names
        contributions: Dict[str, float] = {}
        for col_name, val in zip(features_df.columns, shap_array):
            if abs(val) > 0.001:  # Filter out negligible noise
                contributions[col_name] = round(float(val), 2)

        # 4. Group positive and negative drivers
        positive_factors = sorted(
            [(k, v) for k, v in contributions.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        negative_factors = sorted(
            [(k, v) for k, v in contributions.items() if v < 0],
            key=lambda x: x[1]
        )[:top_n]

        predicted_grade = round(self.predictor.predict_student(student).predicted_grade, 2)

        return ExplanationResult(
            base_value=round(self.expected_value, 2),
            predicted_value=predicted_grade,
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors,
            all_contributions=contributions
        )