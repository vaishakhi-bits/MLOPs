# YugenAI API Logging System

This document describes the comprehensive logging system implemented for the YugenAI API, which tracks incoming prediction requests, model outputs, and system events.

## Overview

The logging system provides:
- **Request/Response Logging**: Tracks all prediction requests with unique IDs, timing, and results
- **Model Status Logging**: Monitors model loading and status changes
- **Error Tracking**: Captures and logs all errors with detailed context
- **Performance Monitoring**: Tracks processing times for each request
- **Client Information**: Logs client IP addresses and user agents

## Log Files

The system creates the following log files in the `logs/` directory:

### 1. `api.log`
General API logs including:
- Application startup/shutdown events
- Model loading status
- General API operations
- Error messages

### 2. `predictions.log`
Structured prediction logs in JSON format including:
- Incoming prediction requests with features
- Prediction responses with results
- Processing times
- Error details
- Client information

## Log Structure

### Prediction Request Log Entry
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "event_type": "prediction_request",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_type": "iris",
  "features": {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
  },
  "client_ip": "127.0.0.1",
  "user_agent": "python-requests/2.28.1"
}
```

### Prediction Response Log Entry
```json
{
  "timestamp": "2024-01-15T10:30:45.234567",
  "event_type": "prediction_response",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_type": "iris",
  "prediction": "Iris-setosa",
  "processing_time_seconds": 0.0234,
  "success": true,
  "error": null
}
```

### Model Status Log Entry
```json
{
  "timestamp": "2024-01-15T10:30:45.345678",
  "event_type": "model_status",
  "model_type": "iris",
  "status": "ready",
  "details": "Model loaded and ready for predictions"
}
```

## API Endpoints

### 1. Prediction Endpoints
- `POST /predict_iris` - Iris flower classification
- `POST /predict_housing` - Housing price prediction

Both endpoints now include comprehensive logging with:
- Unique request IDs
- Request timing
- Feature logging
- Response logging
- Error handling

### 2. Log Viewing Endpoint
- `GET /logs?log_type={type}&lines={count}` - View recent logs

Parameters:
- `log_type`: "api", "prediction", or "all" (default: "api")
- `lines`: Number of recent lines to return (default: 50)

Example:
```bash
# View last 20 API logs
curl "http://localhost:8000/logs?log_type=api&lines=20"

# View last 10 prediction logs
curl "http://localhost:8000/logs?log_type=prediction&lines=10"

# View all recent logs
curl "http://localhost:8000/logs?log_type=all&lines=30"
```

## Usage Examples

### 1. Making a Prediction Request
```python
import requests

# Iris prediction
iris_features = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}

response = requests.post("http://localhost:8000/predict_iris", json=iris_features)
print(response.json())
```

### 2. Viewing Logs Programmatically
```python
import requests

# Get recent prediction logs
response = requests.get("http://localhost:8000/logs?log_type=prediction&lines=10")
logs = response.json()
for log in logs['logs']:
    print(log)
```

### 3. Testing the Logging System
Run the provided test script:
```bash
python test_logging.py
```

This script will:
- Make sample prediction requests
- Display the responses
- Show recent logs from both API and prediction log files

## Log Analysis

### Finding Specific Requests
Each request has a unique ID that appears in both request and response logs. You can search for specific requests:

```bash
# Find all logs for a specific request ID
grep "550e8400-e29b-41d4-a716-446655440000" logs/predictions.log
```

### Performance Analysis
Processing times are logged for each request. You can analyze performance:

```bash
# Extract processing times from prediction logs
grep "processing_time_seconds" logs/predictions.log | jq '.processing_time_seconds'
```

### Error Analysis
Failed requests are logged with error details:

```bash
# Find all failed requests
grep '"success": false' logs/predictions.log
```

## Configuration

### Log Levels
The logging system uses the following log levels:
- `INFO`: Normal operations, requests, responses
- `WARNING`: Non-critical issues (missing features, etc.)
- `ERROR`: Critical errors (model loading failures, prediction errors)

### Log Rotation
Log files are not automatically rotated. For production use, consider:
- Using logrotate or similar tools
- Implementing log file size limits
- Setting up log archival

## Monitoring and Alerting

The structured JSON logs can be easily integrated with monitoring systems:

### Prometheus Integration
The API already includes Prometheus metrics for:
- Request counts
- Request latency
- Status codes

### Log Aggregation
Consider using tools like:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd
- Splunk
- Cloud logging services (AWS CloudWatch, GCP Logging)

## Security Considerations

### Sensitive Data
- Client IP addresses are logged
- User agents are logged
- Feature values are logged in full

### Data Retention
- Implement log retention policies
- Consider data privacy regulations
- Anonymize sensitive data if needed

## Troubleshooting

### Common Issues

1. **Logs not appearing**
   - Check if the `logs/` directory exists
   - Verify file permissions
   - Check disk space

2. **High log volume**
   - Consider reducing log verbosity
   - Implement log filtering
   - Use log rotation

3. **Performance impact**
   - Logging is asynchronous and should have minimal impact
   - Monitor disk I/O if logs are large
   - Consider log buffering for high-traffic scenarios

### Debug Mode
To enable more verbose logging, modify the log level in the logger setup:

```python
logger = setup_logger("api", log_level=logging.DEBUG)
```

## Future Enhancements

Potential improvements to consider:
- Log compression
- Structured logging with correlation IDs
- Integration with distributed tracing
- Real-time log streaming
- Log analytics dashboard
- Automated log analysis and alerting 