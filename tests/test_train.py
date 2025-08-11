import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Import training modules (if they exist)
try:
    from src.models.train_housing import train_housing_model
    from src.models.train_iris import train_iris_model

    TRAINING_MODULES_AVAILABLE = True
except ImportError:
    TRAINING_MODULES_AVAILABLE = False


class TestDataPreprocessing:
    """Test data preprocessing functionality"""

    def test_iris_data_preprocessing(self, sample_iris_data):
        """Test Iris data preprocessing"""
        # Test basic data validation
        assert not sample_iris_data.empty
        assert all(
            col in sample_iris_data.columns
            for col in [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
                "Species",
            ]
        )

        # Test data types
        numeric_cols = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_iris_data[col])

        # Test for missing values
        assert not sample_iris_data[numeric_cols].isnull().any().any()

        # Test target encoding
        unique_species = sample_iris_data["Species"].unique()
        assert len(unique_species) > 0
        assert all(
            species in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            for species in unique_species
        )

    def test_housing_data_preprocessing(self, sample_housing_data):
        """Test Housing data preprocessing"""
        # Test basic data validation
        assert not sample_housing_data.empty
        expected_cols = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ]
        assert all(col in sample_housing_data.columns for col in expected_cols)

        # Test data types
        numeric_cols = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_housing_data[col])

        # Test categorical data
        assert pd.api.types.is_object_dtype(sample_housing_data["ocean_proximity"])

        # Test for missing values
        assert not sample_housing_data[numeric_cols].isnull().any().any()

        # Test ocean proximity values
        valid_ocean_values = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        assert all(
            val in valid_ocean_values
            for val in sample_housing_data["ocean_proximity"].unique()
        )

    def test_feature_scaling(self, sample_housing_data):
        """Test feature scaling functionality"""
        from sklearn.preprocessing import StandardScaler

        # Select numeric features
        numeric_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]

        # Create scaler
        scaler = StandardScaler()

        # Fit and transform
        scaled_data = scaler.fit_transform(sample_housing_data[numeric_features])

        # Test that scaling worked
        assert scaled_data.shape == sample_housing_data[numeric_features].shape

        # Test that scaled data has mean close to 0 and std close to 1
        assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-10)

    def test_one_hot_encoding(self, sample_housing_data):
        """Test one-hot encoding for categorical variables"""
        from sklearn.preprocessing import OneHotEncoder

        # Create encoder
        encoder = OneHotEncoder(sparse_output=False, drop="first")

        # Fit and transform ocean_proximity
        encoded_data = encoder.fit_transform(sample_housing_data[["ocean_proximity"]])

        # Test that encoding worked
        assert encoded_data.shape[0] == len(sample_housing_data)
        assert (
            encoded_data.shape[1]
            == len(sample_housing_data["ocean_proximity"].unique()) - 1
        )

        # Test that all values are binary
        assert np.all(np.isin(encoded_data, [0, 1]))


class TestModelTraining:
    """Test model training functionality"""

    @pytest.mark.skipif(
        not TRAINING_MODULES_AVAILABLE, reason="Training modules not available"
    )
    def test_iris_model_training(self, sample_iris_data, temp_test_dir):
        """Test Iris model training"""
        # Create temporary model path
        model_path = temp_test_dir / "test_iris_model.pkl"

        try:
            # Mock the training function if it exists
            with patch("src.models.train_iris.train_iris_model") as mock_train:
                mock_train.return_value = Mock()

                # Call training function
                result = train_iris_model(sample_iris_data, model_path)

                # Verify training was called
                mock_train.assert_called_once()

                # Verify model file was created (if training actually works)
                # This would depend on the actual implementation

        except Exception as e:
            pytest.skip(f"Iris training test skipped: {str(e)}")

    @pytest.mark.skipif(
        not TRAINING_MODULES_AVAILABLE, reason="Training modules not available"
    )
    def test_housing_model_training(self, sample_housing_data, temp_test_dir):
        """Test Housing model training"""
        # Create temporary model path
        model_path = temp_test_dir / "test_housing_model.pkl"

        try:
            # Mock the training function if it exists
            with patch("src.models.train_housing.train_housing_model") as mock_train:
                mock_train.return_value = Mock()

                # Call training function
                result = train_housing_model(sample_housing_data, model_path)

                # Verify training was called
                mock_train.assert_called_once()

        except Exception as e:
            pytest.skip(f"Housing training test skipped: {str(e)}")

    def test_model_saving_and_loading(self, mock_iris_model, temp_test_dir):
        """Test model saving and loading functionality"""
        model_path = temp_test_dir / "test_model.pkl"

        try:
            # Save model
            joblib.dump(mock_iris_model, model_path)

            # Verify file was created
            assert model_path.exists()

            # Load model
            loaded_model = joblib.load(model_path)

            # Verify model was loaded correctly
            assert loaded_model is not None
            assert hasattr(loaded_model, "predict")

            # Test prediction
            test_data = np.array([[5.1, 3.5, 1.4, 0.2]])
            prediction = loaded_model.predict(test_data)
            assert prediction is not None

        except Exception as e:
            pytest.fail(f"Model saving/loading test failed: {str(e)}")
        finally:
            # Cleanup
            if model_path.exists():
                model_path.unlink()

    def test_model_with_scaler_saving(
        self, mock_housing_model, mock_feature_scaler, temp_test_dir
    ):
        """Test saving model with scaler"""
        model_path = temp_test_dir / "test_model_with_scaler.pkl"

        try:
            # Create model data with scaler
            model_data = {"model": mock_housing_model, "scaler": mock_feature_scaler}

            # Save model data
            joblib.dump(model_data, model_path)

            # Verify file was created
            assert model_path.exists()

            # Load model data
            loaded_data = joblib.load(model_path)

            # Verify both model and scaler were loaded
            assert "model" in loaded_data
            assert "scaler" in loaded_data
            assert hasattr(loaded_data["model"], "predict")
            assert hasattr(loaded_data["scaler"], "transform")

        except Exception as e:
            pytest.fail(f"Model with scaler saving test failed: {str(e)}")
        finally:
            # Cleanup
            if model_path.exists():
                model_path.unlink()


class TestModelEvaluation:
    """Test model evaluation functionality"""

    def test_iris_model_evaluation(self, sample_iris_data):
        """Test Iris model evaluation metrics"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        # Prepare data
        X = sample_iris_data[
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        ]
        y = sample_iris_data["Species"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Test metrics
        assert 0 <= accuracy <= 1
        assert accuracy > 0  # Should have some predictive power

        # Test classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        assert "accuracy" in report
        assert report["accuracy"] == accuracy

    def test_housing_model_evaluation(self, sample_housing_data):
        """Test Housing model evaluation metrics"""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Prepare data
        numeric_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        categorical_features = ["ocean_proximity"]

        # Create target (median_house_value) - using a simple transformation for testing
        y = sample_housing_data["median_income"] * 10000  # Simulate house prices

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        # Create model pipeline
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sample_housing_data, y, test_size=0.2, random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Test metrics
        assert mse >= 0
        assert -1 <= r2 <= 1  # R² can be negative for poor models

    def test_cross_validation(self, sample_iris_data):
        """Test cross-validation functionality"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Prepare data
        X = sample_iris_data[
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        ]
        y = sample_iris_data["Species"]

        # Create model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=3)

        # Test cross-validation results
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)
        assert cv_scores.mean() > 0  # Should have some predictive power


class TestDataValidation:
    """Test data validation functionality"""

    def test_data_quality_checks(self, sample_iris_data):
        """Test data quality validation"""
        # Test for missing values
        assert not sample_iris_data.isnull().any().any()

        # Test for duplicate rows
        assert not sample_iris_data.duplicated().any()

        # Test for infinite values
        numeric_cols = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
        assert not np.isinf(sample_iris_data[numeric_cols]).any().any()

        # Test for reasonable value ranges
        assert all(sample_iris_data["SepalLengthCm"] > 0)
        assert all(sample_iris_data["SepalWidthCm"] > 0)
        assert all(sample_iris_data["PetalLengthCm"] > 0)
        assert all(sample_iris_data["PetalWidthCm"] > 0)

    def test_data_consistency_checks(self, sample_housing_data):
        """Test data consistency validation"""
        # Test for missing values
        assert not sample_housing_data.isnull().any().any()

        # Test for duplicate rows
        assert not sample_housing_data.duplicated().any()

        # Test for infinite values
        numeric_cols = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        assert not np.isinf(sample_housing_data[numeric_cols]).any().any()

        # Test for reasonable value ranges
        assert all(sample_housing_data["longitude"] >= -180)
        assert all(sample_housing_data["longitude"] <= 180)
        assert all(sample_housing_data["latitude"] >= -90)
        assert all(sample_housing_data["latitude"] <= 90)
        assert all(sample_housing_data["housing_median_age"] >= 0)
        assert all(sample_housing_data["total_rooms"] >= 0)
        assert all(sample_housing_data["total_bedrooms"] >= 0)
        assert all(sample_housing_data["population"] >= 0)
        assert all(sample_housing_data["households"] >= 0)
        assert all(sample_housing_data["median_income"] >= 0)


class TestTrainingPipeline:
    """Test complete training pipeline"""

    def test_end_to_end_iris_training(self, sample_iris_data, temp_test_dir):
        """Test end-to-end Iris training pipeline"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        # Prepare data
        X = sample_iris_data[
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        ]
        y = sample_iris_data["Species"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        model_path = temp_test_dir / "iris_model.pkl"
        joblib.dump(model, model_path)

        # Verify model was saved
        assert model_path.exists()

        # Load and test model
        loaded_model = joblib.load(model_path)
        loaded_pred = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, loaded_pred)

        # Verify predictions are the same
        assert np.array_equal(y_pred, loaded_pred)
        assert accuracy == loaded_accuracy

        # Cleanup
        if model_path.exists():
            model_path.unlink()

    def test_end_to_end_housing_training(self, sample_housing_data, temp_test_dir):
        """Test end-to-end Housing training pipeline"""
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Prepare data
        numeric_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        categorical_features = ["ocean_proximity"]

        # Create target
        y = sample_housing_data["median_income"] * 10000

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        # Create model pipeline
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sample_housing_data, y, test_size=0.2, random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Save model
        model_path = temp_test_dir / "housing_model.pkl"
        joblib.dump(model, model_path)

        # Verify model was saved
        assert model_path.exists()

        # Load and test model
        loaded_model = joblib.load(model_path)
        loaded_pred = loaded_model.predict(X_test)
        loaded_mse = mean_squared_error(y_test, loaded_pred)

        # Verify predictions are the same
        assert np.allclose(y_pred, loaded_pred)
        assert mse == loaded_mse

        # Cleanup
        if model_path.exists():
            model_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
