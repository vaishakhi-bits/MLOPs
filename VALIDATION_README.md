# Input Validation with Pydantic

This document describes the comprehensive input validation system implemented using Pydantic V2 for the MLOPs project.

## Overview

The project now includes robust input validation for both the Housing and Iris prediction models, ensuring data quality and preventing invalid requests from reaching the machine learning models.

## Features

### 1. Field-Level Validation

#### Housing Features Validation
- **Coordinates**: Longitude (-180 to 180), Latitude (-90 to 90)
- **Housing Age**: 0 to 100 years
- **Room Counts**: Total rooms (0-10,000), Bedrooms (0-5,000)
- **Population Data**: Population (0-100,000), Households (0-5,000)
- **Income**: Median income (0-50, in tens of thousands)
- **Ocean Proximity**: Strict enum validation for 5 valid values

#### Iris Features Validation
- **Sepal Measurements**: Length (0-20 cm), Width (0-10 cm)
- **Petal Measurements**: Length (0-10 cm), Width (0-5 cm)
- **Positive Values**: All measurements must be positive
- **Typical Ranges**: Validation against typical Iris measurement ranges

### 2. Cross-Field Validation

#### Housing Data Consistency
- Bedrooms cannot exceed total rooms
- Households cannot exceed population
- Median income sanity check (>20.0 flagged as unusual)
- California coordinate bounds validation

#### Iris Data Consistency
- Sepal length should typically be greater than sepal width
- Petal length should typically be greater than petal width
- Measurements within typical Iris species ranges

### 3. Response Validation

#### Prediction Responses
- **Housing**: Predicted price (≥0), optional confidence score (0-1)
- **Iris**: Valid class names, confidence score (0-1), class probabilities validation

### 4. Error Handling

#### Validation Error Responses
- Structured error messages with field names
- Request ID tracking for debugging
- Timestamp information
- Detailed error descriptions

#### API Error Handlers
- Automatic validation error handling
- HTTP 422 for validation errors
- HTTP 500 for internal server errors
- Consistent error response format

## Usage Examples

### Valid Housing Request
```json
{
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
```

### Valid Iris Request
```json
{
  "SepalLengthCm": 5.1,
  "SepalWidthCm": 3.5,
  "PetalLengthCm": 1.4,
  "PetalWidthCm": 0.2
}
```

### Example Validation Error Response
```json
{
  "error": "Validation Error",
  "detail": "Invalid input data: 2 validation errors",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01 12:00:00"
}
```

## New API Endpoints

### Health Check Endpoint
- **GET** `/health`
- Returns model status and API health information
- Useful for monitoring and debugging

### Enhanced Error Responses
- All endpoints now return structured error responses
- Consistent error format across the API
- Request ID tracking for debugging

## Testing

The validation system includes comprehensive tests covering:

- Valid input scenarios
- Invalid input scenarios
- Edge cases and boundary conditions
- Cross-field validation rules
- Error message accuracy
- Serialization/deserialization

Run tests with:
```bash
python -m pytest tests/test_schema.py -v
```

## Benefits

1. **Data Quality**: Prevents invalid data from reaching ML models
2. **Error Prevention**: Catches issues early in the request pipeline
3. **Debugging**: Clear error messages with request tracking
4. **Documentation**: Self-documenting API with field descriptions
5. **Type Safety**: Strong typing with Pydantic models
6. **Consistency**: Uniform validation across all endpoints

## Migration Notes

### From Pydantic V1 to V2
- Updated validators to use `@field_validator` and `@model_validator`
- Replaced `@root_validator` with `@model_validator(mode='after')`
- Updated field configuration to use `json_schema_extra`
- Changed serialization from `.dict()` to `.model_dump()`

### Backward Compatibility
- All existing API endpoints remain functional
- Response formats are enhanced but backward compatible
- New optional fields (confidence scores) don't break existing clients

## Future Enhancements

Potential improvements for the validation system:

1. **Custom Validators**: Domain-specific validation rules
2. **Async Validation**: For complex validation scenarios
3. **Caching**: Validation result caching for performance
4. **Metrics**: Validation error tracking and analytics
5. **Rate Limiting**: Based on validation error frequency

## Configuration

Validation rules can be adjusted in `src/api/schema.py`:

- Field constraints (min/max values)
- Custom validation logic
- Error messages
- Schema examples

The validation system is designed to be easily configurable and extensible for future requirements.
