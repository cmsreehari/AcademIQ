"""
Feature engineering and transformation pipeline for StudyFlow.
Extracts academic trajectory indicators and prepares feature vectors for ML inference.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific educational indicators.

    New Features Added:
    -------------------
    - grade_change : G2 - G1 (Academic momentum/trajectory)
    - avg_term_grade : Mean of G1 and G2
    - parent_edu_total : Medu + Fedu (Cumulative household educational capital)

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with domain features appended.
    """
    df_feat = df.copy()

    # 1. Academic Momentum (Trajectory)
    
    if "G1" in df_feat.columns and "G2" in df_feat.columns:
        df_feat["grade_change"] = df_feat["G2"] - df_feat["G1"]
        df_feat["avg_term_grade"] = (df_feat["G1"] + df_feat["G2"]) / 2.0

    # 2. Total Parental Education
    if "Medu" in df_feat.columns and "Fedu" in df_feat.columns:
        df_feat["parent_edu_total"] = df_feat["Medu"] + df_feat["Fedu"]

    return df_feat


def prepare_features_for_inference(
    df: pd.DataFrame, 
    required_feature_names: List[str]
) -> pd.DataFrame:
    
    """
    Converts raw student data into the exact 56-feature format expected
    by the trained Random Forest model.

    Steps:
    ------
    1. Computes domain features.
    2. Performs one-hot encoding for categorical variables (pd.get_dummies).
    3. Reindexes columns so they match required_feature_names perfectly,
       filling any missing dummy columns with 0.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw or preprocessed student input data.
    required_feature_names : List[str]
        The 56 feature names loaded from model_features.joblib.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with exactly the columns required by the model.
    """
    # 1. Add domain features
    df_transformed = add_domain_features(df)

    # 2. One-hot encode categorical string columns
    df_encoded = pd.get_dummies(df_transformed, drop_first=True)

    # 3. Align columns to match model's training features exactly
    # Missing columns become 0; extra columns are dropped
    df_aligned = df_encoded.reindex(columns=required_feature_names, fill_value=0)

    # Ensure all data is numeric
    return df_aligned.astype(float)