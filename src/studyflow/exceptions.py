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

class DataValidationError(StudyFlowError):
    """Raised when input data fails schema validation or integrity checks."""
    pass
