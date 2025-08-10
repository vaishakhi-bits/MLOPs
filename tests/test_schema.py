import pytest
from pydantic import ValidationError
from src.api.schema import IrisFeatures, HousingFeatures, IrisPredictionResponse, HousingPredictionResponse

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
        """Test Iris features with negative values (should be valid)"""
        valid_data = {
            "SepalLengthCm": -1.0,
            "SepalWidthCm": -2.0,
            "PetalLengthCm": -3.0,
            "PetalWidthCm": -4.0
        }
        
        iris_features = IrisFeatures(**valid_data)
        assert iris_features.SepalLengthCm == -1.0
        assert iris_features.SepalWidthCm == -2.0
        assert iris_features.PetalLengthCm == -3.0
        assert iris_features.PetalWidthCm == -4.0
    
    def test_iris_features_zero_values(self):
        """Test Iris features with zero values"""
        valid_data = {
            "SepalLengthCm": 0.0,
            "SepalWidthCm": 0.0,
            "PetalLengthCm": 0.0,
            "PetalWidthCm": 0.0
        }
        
        iris_features = IrisFeatures(**valid_data)
        assert iris_features.SepalLengthCm == 0.0
        assert iris_features.SepalWidthCm == 0.0
        assert iris_features.PetalLengthCm == 0.0
        assert iris_features.PetalWidthCm == 0.0
    
    def test_iris_features_large_values(self):
        """Test Iris features with large values"""
        valid_data = {
            "SepalLengthCm": 1000.0,
            "SepalWidthCm": 1000.0,
            "PetalLengthCm": 1000.0,
            "PetalWidthCm": 1000.0
        }
        
        iris_features = IrisFeatures(**valid_data)
        assert iris_features.SepalLengthCm == 1000.0
        assert iris_features.SepalWidthCm == 1000.0
        assert iris_features.PetalLengthCm == 1000.0
        assert iris_features.PetalWidthCm == 1000.0

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
        assert errors[0]["type"] == "enum"
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
        assert errors[0]["type"] == "enum"
        assert "ocean_proximity" in errors[0]["loc"]
    
    def test_housing_features_extreme_values(self):
        """Test Housing features with extreme values"""
        valid_data = {
            "longitude": -180.0,  # Minimum longitude
            "latitude": -90.0,    # Minimum latitude
            "housing_median_age": 0.0,
            "total_rooms": 0.0,
            "total_bedrooms": 0.0,
            "population": 0.0,
            "households": 0.0,
            "median_income": 0.0,
            "ocean_proximity": "INLAND"
        }
        
        housing_features = HousingFeatures(**valid_data)
        assert housing_features.longitude == -180.0
        assert housing_features.latitude == -90.0
        assert housing_features.housing_median_age == 0.0
        assert housing_features.total_rooms == 0.0
        assert housing_features.total_bedrooms == 0.0
        assert housing_features.population == 0.0
        assert housing_features.households == 0.0
        assert housing_features.median_income == 0.0
        assert housing_features.ocean_proximity == "INLAND"
    
    def test_housing_features_very_large_values(self):
        """Test Housing features with very large values"""
        valid_data = {
            "longitude": 180.0,   # Maximum longitude
            "latitude": 90.0,     # Maximum latitude
            "housing_median_age": 999999.0,
            "total_rooms": 999999.0,
            "total_bedrooms": 999999.0,
            "population": 999999.0,
            "households": 999999.0,
            "median_income": 999999.0,
            "ocean_proximity": "NEAR OCEAN"
        }
        
        housing_features = HousingFeatures(**valid_data)
        assert housing_features.longitude == 180.0
        assert housing_features.latitude == 90.0
        assert housing_features.housing_median_age == 999999.0
        assert housing_features.total_rooms == 999999.0
        assert housing_features.total_bedrooms == 999999.0
        assert housing_features.population == 999999.0
        assert housing_features.households == 999999.0
        assert housing_features.median_income == 999999.0
        assert housing_features.ocean_proximity == "NEAR OCEAN"

class TestIrisPredictionResponse:
    """Test IrisPredictionResponse schema validation"""
    
    def test_valid_iris_prediction_response(self):
        """Test valid Iris prediction response"""
        valid_data = {
            "predicted_class": "Iris-setosa"
        }
        
        response = IrisPredictionResponse(**valid_data)
        assert response.predicted_class == "Iris-setosa"
    
    def test_iris_prediction_response_all_classes(self):
        """Test Iris prediction response with all possible classes"""
        valid_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
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
    
    def test_iris_prediction_response_empty_string(self):
        """Test Iris prediction response with empty string"""
        valid_data = {
            "predicted_class": ""
        }
        
        response = IrisPredictionResponse(**valid_data)
        assert response.predicted_class == ""

class TestHousingPredictionResponse:
    """Test HousingPredictionResponse schema validation"""
    
    def test_valid_housing_prediction_response(self):
        """Test valid Housing prediction response"""
        valid_data = {
            "predicted_price": 250000.0
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == 250000.0
    
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
        """Test Housing prediction response with negative price"""
        valid_data = {
            "predicted_price": -1000.0
        }
        
        response = HousingPredictionResponse(**valid_data)
        assert response.predicted_price == -1000.0
    
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
        
        data = iris_features.dict()
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
        
        data = housing_features.dict()
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
        response = IrisPredictionResponse(predicted_class="Iris-setosa")
        
        data = response.dict()
        assert data["predicted_class"] == "Iris-setosa"
    
    def test_housing_prediction_response_serialization(self):
        """Test HousingPredictionResponse serialization to dict"""
        response = HousingPredictionResponse(predicted_price=250000.0)
        
        data = response.dict()
        assert data["predicted_price"] == 250000.0

if __name__ == "__main__":
    pytest.main([__file__]) 