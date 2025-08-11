"""
Pytest configuration and shared fixtures for YugenAI tests
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for all tests"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_iris_data():
    """Sample Iris dataset for testing"""
    return pd.DataFrame(
        {
            "SepalLengthCm": [5.1, 4.9, 4.7, 4.6, 5.0],
            "SepalWidthCm": [3.5, 3.0, 3.2, 3.1, 3.6],
            "PetalLengthCm": [1.4, 1.4, 1.3, 1.5, 1.4],
            "PetalWidthCm": [0.2, 0.2, 0.2, 0.2, 0.2],
            "Species": [
                "Iris-setosa",
                "Iris-setosa",
                "Iris-setosa",
                "Iris-setosa",
                "Iris-setosa",
            ],
        }
    )


@pytest.fixture(scope="session")
def sample_housing_data():
    """Sample Housing dataset for testing"""
    return pd.DataFrame(
        {
            "longitude": [-122.23, -122.22, -122.24, -122.25, -122.21],
            "latitude": [37.88, 37.86, 37.85, 37.87, 37.89],
            "housing_median_age": [41.0, 21.0, 52.0, 52.0, 52.0],
            "total_rooms": [880.0, 7099.0, 1467.0, 1274.0, 1627.0],
            "total_bedrooms": [129.0, 1106.0, 190.0, 235.0, 280.0],
            "population": [322.0, 2401.0, 496.0, 558.0, 565.0],
            "households": [126.0, 1138.0, 177.0, 219.0, 259.0],
            "median_income": [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
            "ocean_proximity": [
                "NEAR BAY",
                "NEAR BAY",
                "NEAR BAY",
                "NEAR BAY",
                "NEAR BAY",
            ],
        }
    )


@pytest.fixture
def mock_iris_model():
    """Mock iris model for testing"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0])  # Iris-setosa
    return mock_model


@pytest.fixture
def mock_housing_model():
    """Mock housing model for testing"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([250000.0])
    return mock_model


@pytest.fixture
def mock_feature_scaler():
    """Mock feature scaler for testing"""
    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    )
    return mock_scaler


@pytest.fixture
def sample_iris_features():
    """Sample Iris features for testing"""
    return {
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2,
    }


@pytest.fixture
def sample_housing_features():
    """Sample Housing features for testing"""
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
def mock_logs_directory():
    """Mock logs directory for testing"""
    with pytest.MonkeyPatch().context() as m:
        temp_dir = Path(tempfile.mkdtemp())
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir()

        # Mock the Path to return our temp directory
        m.setattr(
            "src.utils.logger.Path", lambda x: logs_dir if x == "logs" else Path(x)
        )

        yield logs_dir

        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_prediction_logger():
    """Mock prediction logger for testing"""
    mock_logger = Mock()
    mock_logger.info = Mock()
    return mock_logger


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # This runs before each test
    pass


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Cleanup test environment after each test"""
    yield
    # This runs after each test
    pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on their location
        if "test_api.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_logging.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_schema.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        else:
            item.add_marker(pytest.mark.unit)


# Test data generators
class TestDataGenerator:
    """Utility class for generating test data"""

    @staticmethod
    def generate_iris_features(count=10):
        """Generate random Iris features for testing"""
        np.random.seed(42)  # For reproducible results
        return [
            {
                "SepalLengthCm": float(np.random.uniform(4.0, 8.0)),
                "SepalWidthCm": float(np.random.uniform(2.0, 5.0)),
                "PetalLengthCm": float(np.random.uniform(1.0, 7.0)),
                "PetalWidthCm": float(np.random.uniform(0.1, 2.5)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def generate_housing_features(count=10):
        """Generate random Housing features for testing"""
        np.random.seed(42)  # For reproducible results
        ocean_proximities = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

        return [
            {
                "longitude": float(np.random.uniform(-180.0, 180.0)),
                "latitude": float(np.random.uniform(-90.0, 90.0)),
                "housing_median_age": float(np.random.uniform(0.0, 100.0)),
                "total_rooms": float(np.random.uniform(0.0, 10000.0)),
                "total_bedrooms": float(np.random.uniform(0.0, 5000.0)),
                "population": float(np.random.uniform(0.0, 10000.0)),
                "households": float(np.random.uniform(0.0, 5000.0)),
                "median_income": float(np.random.uniform(0.0, 20.0)),
                "ocean_proximity": np.random.choice(ocean_proximities),
            }
            for _ in range(count)
        ]


@pytest.fixture
def test_data_generator():
    """Fixture providing test data generator"""
    return TestDataGenerator
