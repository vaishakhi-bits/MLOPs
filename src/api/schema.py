from pydantic import BaseModel
from typing import Optional
from typing import Literal
 
# ===== Housing Schemas =====
 
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
 
class HousingPredictionResponse(BaseModel):
    predicted_price: float
 
 
# ===== Iris Schemas =====
 
class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float
 
class IrisPredictionResponse(BaseModel):
    predicted_class: str