"""
Concrete Random Forest predictor implementation for StudyFlow.
Loads trained artifacts, aligns features, and evaluates student risk.
"""

from typing import List, Optional
from dataclasses import asdict
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from studyflow import config
from studyflow.exceptions import ModelLoadError, PredictionError
from studyflow.schemas import StudentInput, PredictionOutput
from studyflow.features.engineer import prepare_features_for_inference
from studyflow.models.base import BasePredictor


class RandomForestStudentPredictor(BasePredictor):
    """
    Random Forest Regressor wrapper for predicting student final marks (G3).
    """

    def __init__(
        self, 
        model_path: Optional[Path] = None, 
        features_path: Optional[Path] = None
    ):
        self.model_path = model_path or config.MODEL_PATH
        self.features_path = features_path or config.FEATURES_PATH
        self.model = None
        self.features: List[str] = []
        self.load()

    def load(self) -> None:
        """
        Loads the trained Random Forest model and the 56 feature names.

        Raises:
        -------
        ModelLoadError
            If model files are missing or corrupted.
        """
        if not self.model_path.exists():
            raise ModelLoadError(f"Model artifact not found at: {self.model_path}")

        if not self.features_path.exists():
            raise ModelLoadError(f"Feature list artifact not found at: {self.features_path}")

        try:
            self.model = joblib.load(self.model_path)
            self.features = joblib.load(self.features_path)
        except Exception as err:
            raise ModelLoadError(f"Failed to load model artifacts: {err}") from err

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Executes model prediction on a feature-aligned DataFrame.
        Clips predictions to valid scale [0.0, 20.0].
        """
        if self.model is None:
            raise PredictionError("Model is not loaded. Call load() first.")

        try:
            raw_predictions = self.model.predict(features_df)
            # Clip predictions so they never exceed domain boundaries [0, 20]
            clipped_predictions = np.clip(raw_predictions, config.MIN_GRADE, config.MAX_GRADE)
            return clipped_predictions
        except Exception as err:
            raise PredictionError(f"Error during model inference: {err}") from err

    def predict_student(self, student: StudentInput) -> PredictionOutput:
        """
        End-to-end inference for a single student.
        """
        # 1. Convert dataclass to a single-row DataFrame
        student_dict = asdict(student)
        raw_df = pd.DataFrame([student_dict])

        # 2. Align features to model's exact 56 input columns
        aligned_features = prepare_features_for_inference(raw_df, self.features)

        # 3. Predict grade
        pred_array = self.predict(aligned_features)
        predicted_grade = round(float(pred_array[0]), 2)

        # 4. Determine Status & Risk
        passed = predicted_grade >= config.PASSING_GRADE
        is_at_risk = predicted_grade < config.DROPOUT_RISK_THRESHOLD

        if is_at_risk:
            risk_level = "High Risk"
            summary_message = (
                f"Student predicted at {predicted_grade}/20. High risk of failing or dropping out. "
                "Urgent remedial tutoring and attendance review recommended."
            )
        elif predicted_grade < 12.0:
            risk_level = "Moderate Risk"
            summary_message = (
                f"Student predicted at {predicted_grade}/20. Borderline passing. "
                "Encourage consistent study habits and monitor term progress."
            )
        else:
            risk_level = "On Track"
            summary_message = (
                f"Student predicted at {predicted_grade}/20. Academic momentum is solid. "
                "Maintain current learning schedule."
            )

        return PredictionOutput(
            predicted_grade=predicted_grade,
            is_at_risk=is_at_risk,
            risk_level=risk_level,
            passed=passed,
            summary_message=summary_message
        )