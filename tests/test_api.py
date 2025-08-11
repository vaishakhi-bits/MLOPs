import json
import shutil
import tempfile
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schema import HousingFeatures, IrisFeatures

# Create test client
client = TestClient(app)


# Test data fixtures
@pytest.fixture
def sample_iris_features():
    return {
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2,
    }


@pytest.fixture
def sample_housing_features():
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }


@pytest.fixture
def mock_iris_model():
    """Mock iris model that returns predictable results"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0])  # Iris-setosa
    return mock_model


@pytest.fixture
def mock_housing_model():
    """Mock housing model that returns predictable results"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([250000.0])
    return mock_model


@pytest.fixture
def temp_logs_dir():
    """Create temporary logs directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir()
    yield logs_dir
    shutil.rmtree(temp_dir)


class TestRootEndpoint:
    """Test the root endpoint"""

    def test_read_root(self):
        """Test root endpoint returns correct message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome to YugenAI API" in data["message"]


class TestMetricsEndpoint:
    """Test the Prometheus metrics endpoint"""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Check for some expected Prometheus metrics
        content = response.text
        assert "http_requests_total" in content
        assert "http_request_duration_seconds" in content


class TestLogsEndpoint:
    """Test the logs viewing endpoint"""

    @patch("src.api.main.Path")
    @patch("builtins.open", create=True)
    def test_view_logs_api(self, mock_open, mock_path):
        """Test viewing API logs"""
        # Mock logs directory and files
        mock_logs_dir = Mock()
        mock_path.return_value = mock_logs_dir
        mock_logs_dir.exists.return_value = True

        mock_api_log = Mock()
        mock_api_log.exists.return_value = True

        # Mock the division operation to handle nested paths
        def mock_division(self, other):
            if other == "api":
                mock_api_dir = Mock()
                def mock_api_division(self, x):
                    return mock_api_log
                mock_api_dir.__truediv__ = mock_api_division
                return mock_api_dir
            return mock_api_log
        mock_logs_dir.__truediv__ = mock_division

        # Mock the open function to return a file-like object
        mock_file = Mock()
        mock_file.readlines.return_value = ["2024-01-01 10:00:00 - api - INFO - Test log\n"]
        mock_open.return_value.__enter__.return_value = mock_file

        response = client.get("/logs?log_type=api&lines=5")
        assert response.status_code == 200
        data = response.json()
        assert "log_type" in data
        assert data["log_type"] == "api"
        assert "logs" in data

    @patch("src.api.main.Path")
    @patch("builtins.open", create=True)
    def test_view_logs_prediction(self, mock_open, mock_path):
        """Test viewing prediction logs"""
        # Mock logs directory and files
        mock_logs_dir = Mock()
        mock_path.return_value = mock_logs_dir
        mock_logs_dir.exists.return_value = True

        mock_pred_log = Mock()
        mock_pred_log.exists.return_value = True

        # Mock the division operation to handle nested paths
        def mock_division(self, other):
            if other == "predictions":
                mock_pred_dir = Mock()
                def mock_pred_division(self, x):
                    return mock_pred_log
                mock_pred_dir.__truediv__ = mock_pred_division
                return mock_pred_dir
            return mock_pred_log
        mock_logs_dir.__truediv__ = mock_division

        # Mock the open function to return a file-like object
        mock_file = Mock()
        mock_file.readlines.return_value = ['2024-01-01 10:00:00 - REQUEST: {"test": "data"}\n']
        mock_open.return_value.__enter__.return_value = mock_file

        response = client.get("/logs?log_type=prediction&lines=5")
        assert response.status_code == 200
        data = response.json()
        assert "log_type" in data
        assert data["log_type"] == "prediction"

    def test_view_logs_invalid_type(self):
        """Test viewing logs with invalid log type"""
        response = client.get("/logs?log_type=invalid&lines=5")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert len(data["logs"]) == 0

    @patch("src.api.main.Path")
    def test_view_logs_directory_not_found(self, mock_path):
        """Test viewing logs when logs directory doesn't exist"""
        mock_logs_dir = Mock()
        mock_path.return_value = mock_logs_dir
        mock_logs_dir.exists.return_value = False

        response = client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Logs directory not found" in data["error"]


class TestIrisPrediction:
    """Test Iris prediction endpoint"""

    def test_predict_iris_success(self, sample_iris_features, mock_iris_model):
        """Test successful Iris prediction"""
        with patch("src.api.main.iris_model", mock_iris_model):
            response = client.post("/predict_iris", json=sample_iris_features)
            assert response.status_code == 200
            data = response.json()
            assert "predicted_class" in data
            assert data["predicted_class"] == "Iris-setosa"

    def test_predict_iris_all_classes(self, sample_iris_features):
        """Test Iris prediction for all possible classes"""
        class_mappings = [
            (0, "Iris-setosa"),
            (1, "Iris-versicolor"),
            (2, "Iris-virginica"),
        ]

        for class_id, expected_class in class_mappings:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([class_id])

            with patch("src.api.main.iris_model", mock_model):
                response = client.post("/predict_iris", json=sample_iris_features)
                assert response.status_code == 200
                data = response.json()
                assert data["predicted_class"] == expected_class

    def test_predict_iris_no_model(self, sample_iris_features):
        """Test Iris prediction when model is not loaded"""
        with patch("src.api.main.iris_model", None):
            response = client.post("/predict_iris", json=sample_iris_features)
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "Iris model is not loaded" in data["detail"]

    def test_predict_iris_invalid_features(self):
        """Test Iris prediction with invalid features"""
        invalid_features = {
            "SepalLengthCm": "invalid",  # Should be float
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2,
        }

        response = client.post("/predict_iris", json=invalid_features)
        assert response.status_code == 422  # Validation error

    def test_predict_iris_missing_features(self):
        """Test Iris prediction with missing features"""
        incomplete_features = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            # Missing PetalLengthCm and PetalWidthCm
        }

        response = client.post("/predict_iris", json=incomplete_features)
        assert response.status_code == 422  # Validation error

    def test_predict_iris_model_error(self, sample_iris_features):
        """Test Iris prediction when model raises an error"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")

        with patch("src.api.main.iris_model", mock_model):
            response = client.post("/predict_iris", json=sample_iris_features)
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Model error" in data["detail"]


class TestHousingPrediction:
    """Test Housing prediction endpoint"""

    def test_predict_housing_success(self, sample_housing_features, mock_housing_model):
        """Test successful Housing prediction"""
        with patch("src.api.main.housing_model", mock_housing_model):
            with patch("src.api.main.feature_scaler", None):  # No scaling
                response = client.post("/predict_housing", json=sample_housing_features)
                assert response.status_code == 200
                data = response.json()
                assert "predicted_price" in data
                assert isinstance(data["predicted_price"], float)
                assert data["predicted_price"] == 250000.0

    def test_predict_housing_with_scaler(
        self, sample_housing_features, mock_housing_model
    ):
        """Test Housing prediction with feature scaling"""
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        )

        with patch("src.api.main.housing_model", mock_housing_model):
            with patch("src.api.main.feature_scaler", mock_scaler):
                response = client.post("/predict_housing", json=sample_housing_features)
                assert response.status_code == 200
                data = response.json()
                assert "predicted_price" in data
                assert isinstance(data["predicted_price"], float)

    def test_predict_housing_log_transformed(self, sample_housing_features):
        """Test Housing prediction with log-transformed target"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([12.0])  # log(price)
        mock_model.target_transform_ = "log"

        with patch("src.api.main.housing_model", mock_model):
            with patch("src.api.main.feature_scaler", None):
                response = client.post("/predict_housing", json=sample_housing_features)
                assert response.status_code == 200
                data = response.json()
                assert "predicted_price" in data
                # Should be exp(12.0) - 1
                expected_price = np.expm1(12.0)
                assert abs(data["predicted_price"] - expected_price) < 0.01

    def test_predict_housing_no_model(self, sample_housing_features):
        """Test Housing prediction when model is not loaded"""
        with patch("src.api.main.housing_model", None):
            response = client.post("/predict_housing", json=sample_housing_features)
            assert response.status_code == 200
            data = response.json()
            assert "predicted_price" in data
            assert data["predicted_price"] == 0.0

    def test_predict_housing_invalid_features(self):
        """Test Housing prediction with invalid features"""
        invalid_features = {
            "longitude": "invalid",  # Should be float
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        }

        response = client.post("/predict_housing", json=invalid_features)
        assert response.status_code == 422  # Validation error

    def test_predict_housing_missing_features(self):
        """Test Housing prediction with missing features"""
        incomplete_features = {
            "longitude": -122.23,
            "latitude": 37.88,
            # Missing other required features
        }

        response = client.post("/predict_housing", json=incomplete_features)
        assert response.status_code == 422  # Validation error

    def test_predict_housing_invalid_ocean_proximity(
        self, sample_housing_features, mock_housing_model
    ):
        """Test Housing prediction with invalid ocean proximity"""
        features_with_invalid_ocean = sample_housing_features.copy()
        features_with_invalid_ocean["ocean_proximity"] = "INVALID_LOCATION"

        with patch("src.api.main.housing_model", mock_housing_model):
            with patch("src.api.main.feature_scaler", None):
                response = client.post(
                    "/predict_housing", json=features_with_invalid_ocean
                )
                assert response.status_code == 422  # Should be 422 for validation error
                data = response.json()
                assert "detail" in data  # Should have validation error details
                # Should still work but with warning logged

    def test_predict_housing_all_ocean_proximities(self, mock_housing_model):
        """Test Housing prediction with all valid ocean proximity values"""
        ocean_values = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

        for ocean_value in ocean_values:
            features = {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": ocean_value,
            }

            with patch("src.api.main.housing_model", mock_housing_model):
                with patch("src.api.main.feature_scaler", None):
                    response = client.post("/predict_housing", json=features)
                    assert response.status_code == 200
                    data = response.json()
                    assert "predicted_price" in data
                    assert isinstance(data["predicted_price"], float)

    def test_predict_housing_model_error(self, sample_housing_features):
        """Test Housing prediction when model raises an error"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")

        with patch("src.api.main.housing_model", mock_model):
            with patch("src.api.main.feature_scaler", None):
                response = client.post("/predict_housing", json=sample_housing_features)
                assert response.status_code == 200
                data = response.json()
                assert "predicted_price" in data
                assert data["predicted_price"] == 0.0  # Default value on error


class TestSchemaValidation:
    """Test Pydantic schema validation"""

    def test_iris_features_validation(self):
        """Test IrisFeatures schema validation"""
        # Valid features
        valid_features = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2,
        }
        iris_features = IrisFeatures(**valid_features)
        assert iris_features.SepalLengthCm == 5.1

        # Invalid features (missing required field)
        invalid_features = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            # Missing PetalWidthCm
        }
        with pytest.raises(ValueError):
            IrisFeatures(**invalid_features)

    def test_housing_features_validation(self):
        """Test HousingFeatures schema validation"""
        # Valid features
        valid_features = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        }
        housing_features = HousingFeatures(**valid_features)
        assert housing_features.longitude == -122.23

        # Invalid features (wrong type)
        invalid_features = {
            "longitude": "invalid",
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY",
        }
        with pytest.raises(ValueError):
            HousingFeatures(**invalid_features)


class TestMiddleware:
    """Test FastAPI middleware functionality"""

    def test_cors_middleware(self):
        """Test CORS middleware is configured"""
        # This is a basic test to ensure the app has CORS middleware
        # In a real scenario, you'd test actual CORS headers
        response = client.get("/")
        assert response.status_code == 200

    def test_metrics_middleware(self):
        """Test that metrics are being collected"""
        # Make a request to trigger metrics collection
        response = client.get("/")
        assert response.status_code == 200

        # Check metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_content = metrics_response.text

        # Should contain metrics for our request
        assert "http_requests_total" in metrics_content


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_404_error(self):
        """Test 404 error handling"""
        response = client.get("/nonexistent_endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test method not allowed error"""
        response = client.put("/predict_iris", json={})
        assert response.status_code == 405

    def test_invalid_json(self):
        """Test invalid JSON handling"""
        response = client.post("/predict_iris", data="invalid json")
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__])
