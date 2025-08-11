"""
Data preprocessing utilities for YugenAI project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def preprocess_housing_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Preprocess housing dataset for model training

    Args:
        input_path: Path to raw housing data CSV file
        output_path: Path to save preprocessed data (optional)

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading housing data from: {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df)} rows with missing values")

    # Target column
    target_col = "median_house_value"

    # Handle categorical columns (ocean_proximity)
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:
        logger.info(f"Applying one-hot encoding to: {categorical_columns}")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Get numerical columns (excluding target)
    numerical_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]

    # Scale numerical features
    if numerical_columns:
        logger.info(f"Scaling numerical features: {numerical_columns}")
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Ensure target column is the last column
    if target_col in df.columns:
        columns = [col for col in df.columns if col != target_col] + [target_col]
        df = df[columns]

    # Save preprocessed data if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to: {output_path}")

    return df


def preprocess_iris_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Preprocess iris dataset for model training

    Args:
        input_path: Path to raw iris data CSV file
        output_path: Path to save preprocessed data (optional)

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading iris data from: {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df)} rows with missing values")

    # Encode target variable (Species)
    if "Species" in df.columns:
        label_encoder = LabelEncoder()
        df["Species"] = label_encoder.fit_transform(df["Species"])
        logger.info(f"Encoded target variable with classes: {label_encoder.classes_}")

    # Scale features
    feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logger.info(f"Scaled feature columns: {feature_columns}")

    # Save preprocessed data if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to: {output_path}")

    return df


def validate_dataframe(df: pd.DataFrame, expected_columns: list = None) -> bool:
    """
    Validate DataFrame for training

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names

    Returns:
        True if validation passes, False otherwise
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False

    if df.isnull().any().any():
        logger.error("DataFrame contains missing values")
        return False

    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False

    logger.info("DataFrame validation passed")
    return True
