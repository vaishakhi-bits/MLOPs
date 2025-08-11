import pytest
from pydantic import ValidationError
from src.api.schema import (
    IrisFeatures, 
    HousingFeatures, 
    IrisPredictionResponse, 
    HousingPredictionResponse,
    ModelStatus,
    HealthCheck,
    ErrorResponse
)

class TestIrisFeatures:
    """Test IrisFeatures schema validation"""
    
    def test_valid_iris_features(self):
        """Test valid Iris features"""
        valid_data = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
        
        iris_features = IrisFeatures(**valid_data)
        assert iris_features.SepalLengthCm == 5.1
        assert iris_features.SepalWidthCm == 3.5
        assert iris_features.PetalLengthCm == 1.4
        assert iris_features.PetalWidthCm == 0.2
    
    def test_iris_features_with_float_strings(self):
        """Test Iris features with string representations of floats"""
        valid_data = {
            "SepalLengthCm": "5.1",
            "SepalWidthCm": "3.5",
            "PetalLengthCm": "1.4",
            "PetalWidthCm": "0.2"
        }
        
        iris_features = IrisFeatures(**valid_data)
        assert iris_features.SepalLengthCm == 5.1
        assert iris_features.SepalWidthCm == 3.5
        assert iris_features.PetalLengthCm == 1.4
        assert iris_features.PetalWidthCm == 0.2
    
    def test_iris_features_missing_field(self):
        """Test Iris features with missing required field"""
        invalid_data = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4
            # Missing PetalWidthCm
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "PetalWidthCm" in errors[0]["loc"]
    
    def test_iris_features_invalid_type(self):
        """Test Iris features with invalid data type"""
        invalid_data = {
            "SepalLengthCm": "invalid",
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "float_parsing"
        assert "SepalLengthCm" in errors[0]["loc"]
    
    def test_iris_features_negative_values(self):
        """Test Iris features with negative values (should fail validation)"""
        invalid_data = {
            "SepalLengthCm": -1.0,
            "SepalWidthCm": -2.0,
            "PetalLengthCm": -3.0,
            "PetalWidthCm": -4.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1  # At least one validation error
    
    def test_iris_features_zero_values(self):
        """Test Iris features with zero values (should fail validation)"""
        invalid_data = {
            "SepalLengthCm": 0.0,
            "SepalWidthCm": 0.0,
            "PetalLengthCm": 0.0,
            "PetalWidthCm": 0.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1  # At least one validation error
    
    def test_iris_features_out_of_range_values(self):
        """Test Iris features with values outside allowed ranges"""
        invalid_data = {
            "SepalLengthCm": 25.0,  # > 20.0
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_iris_features_sepal_length_less_than_width(self):
        """Test validation when sepal length is less than width"""
        invalid_data = {
            "SepalLengthCm": 3.0,  # Less than width
            "SepalWidthCm": 4.0,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_iris_features_petal_length_less_than_width(self):
        """Test validation when petal length is less than width"""
        invalid_data = {
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 0.1,  # Less than width
            "PetalWidthCm": 0.5
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1

class TestHousingFeatures:
    """Test HousingFeatures schema validation"""
    
    def test_valid_housing_features(self):
        """Test valid Housing features"""
        valid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        housing_features = HousingFeatures(**valid_data)
        assert housing_features.longitude == -122.23
        assert housing_features.latitude == 37.88
        assert housing_features.housing_median_age == 41.0
        assert housing_features.total_rooms == 880.0
        assert housing_features.total_bedrooms == 129.0
        assert housing_features.population == 322.0
        assert housing_features.households == 126.0
        assert housing_features.median_income == 8.3252
        assert housing_features.ocean_proximity == "NEAR BAY"
    
    def test_housing_features_with_strings(self):
        """Test Housing features with string representations of numbers"""
        valid_data = {
            "longitude": "-122.23",
            "latitude": "37.88",
            "housing_median_age": "41.0",
            "total_rooms": "880.0",
            "total_bedrooms": "129.0",
            "population": "322.0",
            "households": "126.0",
            "median_income": "8.3252",
            "ocean_proximity": "NEAR BAY"
        }
        
        housing_features = HousingFeatures(**valid_data)
        assert housing_features.longitude == -122.23
        assert housing_features.latitude == 37.88
        assert housing_features.housing_median_age == 41.0
        assert housing_features.total_rooms == 880.0
        assert housing_features.total_bedrooms == 129.0
        assert housing_features.population == 322.0
        assert housing_features.households == 126.0
        assert housing_features.median_income == 8.3252
        assert housing_features.ocean_proximity == "NEAR BAY"
    
    def test_housing_features_missing_field(self):
        """Test Housing features with missing required field"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252
            # Missing ocean_proximity
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "ocean_proximity" in errors[0]["loc"]
    
    def test_housing_features_invalid_numeric_type(self):
        """Test Housing features with invalid numeric data type"""
        invalid_data = {
            "longitude": "invalid",
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "float_parsing"
        assert "longitude" in errors[0]["loc"]
    
    def test_housing_features_all_ocean_proximities(self):
        """Test Housing features with all valid ocean proximity values"""
        ocean_values = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        
        for ocean_value in ocean_values:
            valid_data = {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": ocean_value
            }
            
            housing_features = HousingFeatures(**valid_data)
            assert housing_features.ocean_proximity == ocean_value
    
    def test_housing_features_invalid_ocean_proximity(self):
        """Test Housing features with invalid ocean proximity"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "INVALID_LOCATION"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "literal_error"
        assert "ocean_proximity" in errors[0]["loc"]
    
    def test_housing_features_case_sensitive_ocean_proximity(self):
        """Test Housing features with case-sensitive ocean proximity"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "near bay"  # lowercase
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "literal_error"
        assert "ocean_proximity" in errors[0]["loc"]
    
    def test_housing_features_out_of_range_values(self):
        """Test Housing features with values outside allowed ranges"""
        invalid_data = {
            "longitude": 200.0,  # > 180.0
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_housing_features_bedrooms_exceed_rooms(self):
        """Test validation when bedrooms exceed total rooms"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 10.0,
            "total_bedrooms": 15.0,  # More bedrooms than rooms
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_housing_features_households_exceed_population(self):
        """Test validation when households exceed population"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 100.0,
            "households": 150.0,  # More households than population
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_housing_features_high_median_income(self):
        """Test validation for unusually high median income"""
        invalid_data = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 25.0,  # > 20.0
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_housing_features_outside_california_bounds(self):
        """Test validation for coordinates outside California"""
        invalid_data = {
            "longitude": -100.0,  # Outside California bounds
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingFeatures(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1

class TestIrisPredictionResponse:
    """Test IrisPredictionResponse schema validation"""
    
    def test_valid_iris_prediction_response(self):
        """Test valid Iris prediction response"""
        valid_data = {
            "predicted_class": "Iris-setosa"
        }
        
        response = IrisPredictionResponse(**valid_data)
        assert response.predicted_class == "Iris-setosa"
        assert response.confidence_score is None
        assert response.class_probabilities is None
    
    def test_iris_prediction_response_with_confidence(self):
        """Test Iris prediction response with confidence score"""
        valid_data = {
            "predicted_class": "Iris-setosa",
            "confidence_score": 0.95
        }
        
        response = IrisPredictionResponse(**valid_data)
        assert response.predicted_class == "Iris-setosa"
        assert response.confidence_score == 0.95
        assert response.class_probabilities is None
    
    def test_iris_prediction_response_with_probabilities(self):
        """Test Iris prediction response with class probabilities"""
        valid_data = {
            "predicted_class": "Iris-setosa",
            "confidence_score": 0.95,
            "class_probabilities": {
                "Iris-setosa": 0.95,
                "Iris-versicolor": 0.03,
                "Iris-virginica": 0.02
            }
        }
        
        response = IrisPredictionResponse(**valid_data)
        assert response.predicted_class == "Iris-setosa"
        assert response.confidence_score == 0.95
        assert response.class_probabilities == {
            "Iris-setosa": 0.95,
            "Iris-versicolor": 0.03,
            "Iris-virginica": 0.02
        }
    
    def test_iris_prediction_response_all_classes(self):
        """Test Iris prediction response with all possible classes"""
        valid_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica", "Unknown"]
        
        for predicted_class in valid_classes:
            valid_data = {
                "predicted_class": predicted_class
            }
            
            response = IrisPredictionResponse(**valid_data)
            assert response.predicted_class == predicted_class
    
    def test_iris_prediction_response_missing_field(self):
        """Test Iris prediction response with missing field"""
        invalid_data = {}
        
        with pytest.raises(ValidationError) as exc_info:
            IrisPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "predicted_class" in errors[0]["loc"]
    
    def test_iris_prediction_response_invalid_class(self):
        """Test Iris prediction response with invalid class"""
        invalid_data = {
            "predicted_class": "Invalid-Class"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_iris_prediction_response_invalid_confidence_score(self):
        """Test Iris prediction response with invalid confidence score"""
        invalid_data = {
            "predicted_class": "Iris-setosa",
            "confidence_score": 1.5  # > 1.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_iris_prediction_response_invalid_probabilities(self):
        """Test Iris prediction response with invalid probabilities"""
        invalid_data = {
            "predicted_class": "Iris-setosa",
            "class_probabilities": {
                "Iris-setosa": 0.5,
                "Iris-versicolor": 0.3,
                "Iris-virginica": 0.1  # Sum < 1.0
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IrisPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1

class TestHousingPredictionResponse:
    """Test HousingPredictionResponse schema validation"""
    
    def test_valid_housing_prediction_response(self):
        """Test valid Housing prediction response"""
        valid_data = {
            "predicted_price": 250000.0
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 250000.0
        assert response.confidence_score is None
    
    def test_housing_prediction_response_with_confidence(self):
        """Test Housing prediction response with confidence score"""
        valid_data = {
            "predicted_price": 250000.0,
            "confidence_score": 0.85
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 250000.0
        assert response.confidence_score == 0.85
    
    def test_housing_prediction_response_with_string(self):
        """Test Housing prediction response with string representation of float"""
        valid_data = {
            "predicted_price": "250000.0"
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 250000.0
    
    def test_housing_prediction_response_zero_price(self):
        """Test Housing prediction response with zero price"""
        valid_data = {
            "predicted_price": 0.0
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 0.0
    
    def test_housing_prediction_response_negative_price(self):
        """Test Housing prediction response with negative price (should fail)"""
        invalid_data = {
            "predicted_price": -1000.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_housing_prediction_response_very_large_price(self):
        """Test Housing prediction response with very large price"""
        valid_data = {
            "predicted_price": 999999999.99
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 999999999.99
    
    def test_housing_prediction_response_missing_field(self):
        """Test Housing prediction response with missing field"""
        invalid_data = {}
        
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "predicted_price" in errors[0]["loc"]
    
    def test_housing_prediction_response_invalid_type(self):
        """Test Housing prediction response with invalid data type"""
        invalid_data = {
            "predicted_price": "invalid"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "float_parsing"
        assert "predicted_price" in errors[0]["loc"]
    
    def test_housing_prediction_response_invalid_confidence_score(self):
        """Test Housing prediction response with invalid confidence score"""
        invalid_data = {
            "predicted_price": 250000.0,
            "confidence_score": -0.1  # < 0.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HousingPredictionResponse(**invalid_data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1

class TestModelStatus:
    """Test ModelStatus schema validation"""
    
    def test_valid_model_status(self):
        """Test valid ModelStatus"""
        valid_data = {
            "model_name": "iris",
            "status": "loaded",
            "loaded": True,
            "last_updated": "2024-01-01 12:00:00"
        }
        
        model_status = ModelStatus(**valid_data)
        assert model_status.model_name == "iris"
        assert model_status.status == "loaded"
        assert model_status.loaded is True
        assert model_status.last_updated == "2024-01-01 12:00:00"
    
    def test_model_status_without_last_updated(self):
        """Test ModelStatus without last_updated field"""
        valid_data = {
            "model_name": "housing",
            "status": "not_loaded",
            "loaded": False
        }
        
        model_status = ModelStatus(**valid_data)
        assert model_status.model_name == "housing"
        assert model_status.status == "not_loaded"
        assert model_status.loaded is False
        assert model_status.last_updated is None

class TestHealthCheck:
    """Test HealthCheck schema validation"""
    
    def test_valid_health_check(self):
        """Test valid HealthCheck"""
        valid_data = {
            "status": "healthy",
            "models": [
                {
                    "model_name": "iris",
                    "status": "loaded",
                    "loaded": True,
                    "last_updated": "2024-01-01 12:00:00"
                },
                {
                    "model_name": "housing",
                    "status": "loaded",
                    "loaded": True,
                    "last_updated": "2024-01-01 12:00:00"
                }
            ],
            "timestamp": "2024-01-01 12:00:00",
            "version": "1.0.0"
        }
        
        health_check = HealthCheck(**valid_data)
        assert health_check.status == "healthy"
        assert len(health_check.models) == 2
        assert health_check.timestamp == "2024-01-01 12:00:00"
        assert health_check.version == "1.0.0"

class TestErrorResponse:
    """Test ErrorResponse schema validation"""
    
    def test_valid_error_response(self):
        """Test valid ErrorResponse"""
        valid_data = {
            "error": "Validation Error",
            "detail": "Invalid input data",
            "request_id": "12345",
            "timestamp": "2024-01-01 12:00:00"
        }
        
        error_response = ErrorResponse(**valid_data)
        assert error_response.error == "Validation Error"
        assert error_response.detail == "Invalid input data"
        assert error_response.request_id == "12345"
        assert error_response.timestamp == "2024-01-01 12:00:00"
    
    def test_error_response_without_optional_fields(self):
        """Test ErrorResponse without optional fields"""
        valid_data = {
            "error": "Internal Server Error",
            "timestamp": "2024-01-01 12:00:00"
        }
        
        error_response = ErrorResponse(**valid_data)
        assert error_response.error == "Internal Server Error"
        assert error_response.detail is None
        assert error_response.request_id is None
        assert error_response.timestamp == "2024-01-01 12:00:00"

class TestSchemaSerialization:
    """Test schema serialization and deserialization"""
    
    def test_iris_features_serialization(self):
        """Test IrisFeatures serialization to dict"""
        iris_features = IrisFeatures(
            SepalLengthCm=5.1,
            SepalWidthCm=3.5,
            PetalLengthCm=1.4,
            PetalWidthCm=0.2
        )
        
        data = iris_features.model_dump()
        assert data["SepalLengthCm"] == 5.1
        assert data["SepalWidthCm"] == 3.5
        assert data["PetalLengthCm"] == 1.4
        assert data["PetalWidthCm"] == 0.2
    
    def test_housing_features_serialization(self):
        """Test HousingFeatures serialization to dict"""
        housing_features = HousingFeatures(
            longitude=-122.23,
            latitude=37.88,
            housing_median_age=41.0,
            total_rooms=880.0,
            total_bedrooms=129.0,
            population=322.0,
            households=126.0,
            median_income=8.3252,
            ocean_proximity="NEAR BAY"
        )
        
        data = housing_features.model_dump()
        assert data["longitude"] == -122.23
        assert data["latitude"] == 37.88
        assert data["housing_median_age"] == 41.0
        assert data["total_rooms"] == 880.0
        assert data["total_bedrooms"] == 129.0
        assert data["population"] == 322.0
        assert data["households"] == 126.0
        assert data["median_income"] == 8.3252
        assert data["ocean_proximity"] == "NEAR BAY"
    
    def test_iris_prediction_response_serialization(self):
        """Test IrisPredictionResponse serialization to dict"""
        response = IrisPredictionResponse(
            predicted_class="Iris-setosa",
            confidence_score=0.95,
            class_probabilities={
                "Iris-setosa": 0.95,
                "Iris-versicolor": 0.03,
                "Iris-virginica": 0.02
            }
        )
        
        data = response.model_dump()
        assert data["predicted_class"] == "Iris-setosa"
        assert data["confidence_score"] == 0.95
        assert data["class_probabilities"] == {
            "Iris-setosa": 0.95,
            "Iris-versicolor": 0.03,
            "Iris-virginica": 0.02
        }
    
    def test_housing_prediction_response_serialization(self):
        """Test HousingPredictionResponse serialization to dict"""
        response = HousingPredictionResponse(
            predicted_price=250000.0,
            confidence_score=0.85
        )
        
        data = response.model_dump()
        assert data["predicted_price"] == 250000.0
        assert data["confidence_score"] == 0.85

if __name__ == "__main__":
    pytest.main([__file__]) 