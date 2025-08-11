from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
import numpy as np

# ===== Housing Schemas =====


class HousingFeatures(BaseModel):
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude coordinate (-180 to 180)",
        json_schema_extra={"example": -122.23},
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude coordinate (-90 to 90)",
        json_schema_extra={"example": 37.88},
    )
    housing_median_age: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Median age of housing units (0 to 100 years)",
        json_schema_extra={"example": 41.0},
    )
    total_rooms: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Total number of rooms (0 to 10,000)",
        json_schema_extra={"example": 880.0},
    )
    total_bedrooms: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Total number of bedrooms (0 to 5,000)",
        json_schema_extra={"example": 129.0},
    )
    population: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Population count (0 to 100,000)",
        json_schema_extra={"example": 322.0},
    )
    households: float = Field(
        ...,
        ge=0.0,
        le=5000.0,
        description="Number of households (0 to 5,000)",
        json_schema_extra={"example": 126.0},
    )
    median_income: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Median income in tens of thousands (0 to 50)",
        json_schema_extra={"example": 8.3252},
    )
    ocean_proximity: Literal[
        "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"
    ] = Field(
        ...,
        description="Proximity to the ocean",
        json_schema_extra={"example": "NEAR BAY"},
    )

    @field_validator("total_bedrooms")
    @classmethod
    def validate_bedrooms_vs_rooms(cls, v, info):
        """Validate that bedrooms don't exceed total rooms"""
        if "total_rooms" in info.data and v > info.data["total_rooms"]:
            raise ValueError("Total bedrooms cannot exceed total rooms")
        return v

    @field_validator("households")
    @classmethod
    def validate_households_vs_population(cls, v, info):
        """Validate that households don't exceed population"""
        if "population" in info.data and v > info.data["population"]:
            raise ValueError("Number of households cannot exceed population")
        return v

    @field_validator("median_income")
    @classmethod
    def validate_median_income(cls, v):
        """Validate median income is reasonable"""
        if v > 20.0:
            raise ValueError("Median income seems unusually high (>20.0)")
        return v

    @model_validator(mode="after")
    def validate_coordinate_consistency(self):
        """Validate coordinate consistency for California housing data"""
        # California roughly spans -124 to -114 longitude and 32 to 42 latitude
        if not (-124 <= self.longitude <= -114) or not (32 <= self.latitude <= 42):
            raise ValueError("Coordinates appear to be outside California bounds")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class HousingPredictionResponse(BaseModel):
    predicted_price: float = Field(
        ...,
        ge=0.0,
        description="Predicted housing price in USD",
        json_schema_extra={"example": 250000.0},
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0 to 1)",
        json_schema_extra={"example": 0.85},
    )

    model_config = {
        "json_schema_extra": {
            "example": {"predicted_price": 250000.0, "confidence_score": 0.85}
        }
    }


# ===== Iris Schemas =====


class IrisFeatures(BaseModel):
    SepalLengthCm: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Sepal length in centimeters (0 to 20 cm)",
        json_schema_extra={"example": 5.1},
    )
    SepalWidthCm: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Sepal width in centimeters (0 to 10 cm)",
        json_schema_extra={"example": 3.5},
    )
    PetalLengthCm: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Petal length in centimeters (0 to 10 cm)",
        json_schema_extra={"example": 1.4},
    )
    PetalWidthCm: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="Petal width in centimeters (0 to 5 cm)",
        json_schema_extra={"example": 0.2},
    )

    @field_validator("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
    @classmethod
    def validate_positive_measurements(cls, v):
        """Validate that all measurements are positive"""
        if v <= 0:
            raise ValueError("All measurements must be positive")
        return v

    @field_validator("SepalLengthCm")
    @classmethod
    def validate_sepal_length_vs_width(cls, v, info):
        """Validate sepal length is reasonable compared to width"""
        if "SepalWidthCm" in info.data and v < info.data["SepalWidthCm"]:
            raise ValueError(
                "Sepal length should typically be greater than sepal width"
            )
        return v

    @field_validator("PetalLengthCm")
    @classmethod
    def validate_petal_length_vs_width(cls, v, info):
        """Validate petal length is reasonable compared to width"""
        if "PetalWidthCm" in info.data and v < info.data["PetalWidthCm"]:
            raise ValueError(
                "Petal length should typically be greater than petal width"
            )
        return v

    @model_validator(mode="after")
    def validate_measurement_consistency(self):
        """Validate overall measurement consistency"""
        # Check if measurements are within typical Iris ranges
        if not (4.0 <= self.SepalLengthCm <= 8.0):
            raise ValueError("Sepal length is outside typical Iris range (4-8 cm)")
        if not (2.0 <= self.SepalWidthCm <= 5.0):
            raise ValueError("Sepal width is outside typical Iris range (2-5 cm)")
        if not (1.0 <= self.PetalLengthCm <= 7.0):
            raise ValueError("Petal length is outside typical Iris range (1-7 cm)")
        if not (0.1 <= self.PetalWidthCm <= 2.5):
            raise ValueError("Petal width is outside typical Iris range (0.1-2.5 cm)")

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            }
        }
    }


class IrisPredictionResponse(BaseModel):
    predicted_class: str = Field(
        ...,
        description="Predicted Iris species",
        json_schema_extra={"example": "Iris-setosa"},
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0 to 1)",
        json_schema_extra={"example": 0.95},
    )
    class_probabilities: Optional[dict] = Field(
        None,
        description="Probability scores for each class",
        json_schema_extra={
            "example": {
                "Iris-setosa": 0.95,
                "Iris-versicolor": 0.03,
                "Iris-virginica": 0.02,
            }
        },
    )

    @field_validator("predicted_class")
    @classmethod
    def validate_predicted_class(cls, v):
        """Validate predicted class is one of the expected Iris species"""
        valid_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica", "Unknown"]
        if v not in valid_classes:
            raise ValueError(
                f'Predicted class must be one of: {", ".join(valid_classes)}'
            )
        return v

    @field_validator("class_probabilities")
    @classmethod
    def validate_class_probabilities(cls, v):
        """Validate class probabilities if provided"""
        if v is not None:
            valid_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            for class_name, prob in v.items():
                if class_name not in valid_classes:
                    raise ValueError(
                        f"Invalid class name in probabilities: {class_name}"
                    )
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(
                        f"Probability for {class_name} must be between 0 and 1"
                    )

            # Check if probabilities sum to approximately 1
            total_prob = sum(v.values())
            if not (0.99 <= total_prob <= 1.01):
                raise ValueError("Class probabilities must sum to approximately 1.0")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "Iris-setosa",
                "confidence_score": 0.95,
                "class_probabilities": {
                    "Iris-setosa": 0.95,
                    "Iris-versicolor": 0.03,
                    "Iris-virginica": 0.02,
                },
            }
        }
    }


# ===== Additional Utility Schemas =====


class ModelStatus(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    status: str = Field(..., description="Current status of the model")
    loaded: bool = Field(..., description="Whether the model is loaded")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class HealthCheck(BaseModel):
    status: str = Field(..., description="Overall health status")
    models: list[ModelStatus] = Field(..., description="Status of all models")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: str = Field(..., description="Error timestamp")


# ===== Retraining Schemas =====


class RetrainingConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    data_path: str = Field(..., description="Path to the data file")
    check_interval: int = Field(300, description="Check interval in seconds")
    min_data_size: int = Field(100, description="Minimum data size required")
    max_model_age_hours: int = Field(24, description="Maximum model age in hours")
    performance_threshold: float = Field(
        0.05, description="Performance degradation threshold"
    )
    enable_auto_retraining: bool = Field(
        True, description="Enable automatic retraining"
    )


class RetrainingStatus(BaseModel):
    is_running: bool = Field(
        ..., description="Whether retraining monitoring is running"
    )
    models: dict = Field(..., description="Status of each model")


class RetrainingHistory(BaseModel):
    history: list = Field(..., description="List of retraining events")
    total: int = Field(..., description="Total number of events")
