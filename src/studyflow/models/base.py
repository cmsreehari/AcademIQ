"""
Abstract base class definition for all predictive models in StudyFlow.
Defines a standard interface for loading, inference, and student risk evaluation.
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np

from studyflow.schemas import StudentInput, PredictionOutput


class BasePredictor(ABC):
    """
    Abstract interface that all StudyFlow ML model wrappers must implement.
    """

    @abstractmethod
    def load(self) -> None:
        """Loads model artifacts and feature configurations from storage."""
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Executes raw prediction on preprocessed feature vectors.

        Returns:
        --------
        np.ndarray
            Predicted final exam grades (G3).
        """
        pass

    @abstractmethod
    def predict_student(self, student: StudentInput) -> PredictionOutput:
        """
        End-to-end inference for a single student.
        Validates input, engineers features, predicts grade, and assigns risk band.

        Returns:
        --------
        PredictionOutput
            Structured prediction result.
        """
        pass