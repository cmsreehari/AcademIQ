"""
Data loading utilities for StudyFlow.
Encapsulates CSV loading with delimiter handling and path resolution.

mostly it avoids the repetation of read _csv and path handling in multiple places and related error that could happen. 
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from studyflow import config
from studyflow.exceptions import StudyFlowError


def load_student_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads student performance data from a CSV file.

    Parameters:
    -----------
    filepath : Optional[Path]
        Custom path to the CSV file. If None, defaults to config.DATASET_PATH.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset with 33 columns.

    Raises:
    -------
    StudyFlowError
        If the file does not exist or cannot be parsed.
    """
    path_to_load = filepath or config.DATASET_PATH

    if not path_to_load.exists():
        raise StudyFlowError(f"Dataset file not found at: {path_to_load}")

    try:
        # UCI Portuguese student dataset uses semicolon (;) as separator
        df = pd.read_csv(path_to_load, sep=";")
        return df
    except Exception as err:
        raise StudyFlowError(f"Failed to load dataset: {err}") from err