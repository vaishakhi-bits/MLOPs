# Model Retraining System

This document describes the automated model retraining system that monitors data changes and triggers retraining when new data is available.

## Overview

The retraining system provides automatic model retraining based on various triggers:
- **Data Changes**: File system monitoring for data file modifications
- **Time-based**: Retraining when models exceed maximum age
- **Performance-based**: Retraining when performance degrades below threshold
- **Manual**: On-demand retraining via API or CLI

## Features

### **Automatic Triggers**
- **File System Monitoring**: Watches data directories for changes
- **Time-based Triggers**: Retrains models older than configured threshold
- **Performance Monitoring**: Tracks model performance degradation
- **Data Sufficiency Checks**: Ensures sufficient data before retraining

### **Monitoring & Metrics**
- **Prometheus Integration**: Comprehensive metrics for retraining events
- **Retraining History**: Complete audit trail of all retraining events
- **Real-time Status**: Current status of all models and retraining processes
- **Performance Tracking**: Model age and performance metrics

### **Management Tools**
- **CLI Interface**: Full command-line management
- **REST API**: Programmatic control via HTTP endpoints
- **Configuration Management**: Flexible configuration per model
- **Notification System**: Callback support for retraining events

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Files    │    │ Retraining      │    │   ML Models     │
│   (CSV, etc.)   │───▶│ Manager         │───▶│   (Updated)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ File System     │    │ Background      │    │ MLflow          │
│ Watcher         │    │ Monitoring      │    │ Tracking        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install watchdog>=3.0.0
```

### 2. Start Retraining System

```bash
# Using CLI
python scripts/manage_retraining.py start

# Using API
curl -X POST http://localhost:8000/retraining/start
```

### 3. Monitor Status

```bash
# Check status
python scripts/manage_retraining.py status

# View history
python scripts/manage_retraining.py history

# API endpoint
curl http://localhost:8000/retraining/status
```

## Configuration

### Default Configuration

```python
{
    "iris": {
        "model_name": "iris",
        "data_path": "data/raw/iris.csv",
        "check_interval": 300,  # 5 minutes
        "min_data_size": 50,
        "max_model_age_hours": 24,
        "performance_threshold": 0.05,
        "enable_auto_retraining": True
    },
    "housing": {
        "model_name": "housing",
        "data_path": "data/raw/housing.csv",
        "check_interval": 600,  # 10 minutes
        "min_data_size": 100,
        "max_model_age_hours": 48,
        "performance_threshold": 0.03,
        "enable_auto_retraining": True
    }
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `check_interval` | How often to check for triggers (seconds) | 300 |
| `min_data_size` | Minimum number of records required | 100 |
| `max_model_age_hours` | Maximum model age before retraining | 24 |
| `performance_threshold` | Performance degradation threshold | 0.05 |
| `enable_auto_retraining` | Enable automatic retraining | True |

## CLI Commands

### Start/Stop System

```bash
# Start retraining monitoring
python scripts/manage_retraining.py start

# Stop retraining monitoring
python scripts/manage_retraining.py stop

# Start with custom data paths
python scripts/manage_retraining.py start --iris-data /path/to/iris.csv --housing-data /path/to/housing.csv
```

### Status & Monitoring

```bash
# Show system status
python scripts/manage_retraining.py status

# View retraining history
python scripts/manage_retraining.py history

# View history for specific model
python scripts/manage_retraining.py history --model iris

# Limit history results
python scripts/manage_retraining.py history --limit 10
```

### Manual Control

```bash
# Manually trigger retraining
python scripts/manage_retraining.py trigger iris
python scripts/manage_retraining.py trigger housing

# Configure retraining settings
python scripts/manage_retraining.py config --model iris
```

## API Endpoints

### Start/Stop System

```bash
# Start retraining system
POST /retraining/start

# Stop retraining system
POST /retraining/stop
```

### Status & Monitoring

```bash
# Get system status
GET /retraining/status

# Get retraining history
GET /retraining/history?model_name=iris&limit=20
```

### Manual Control

```bash
# Trigger manual retraining
POST /retraining/trigger/{model_name}
```

### Example API Usage

```python
import requests

# Start retraining system
response = requests.post("http://localhost:8000/retraining/start")
print(response.json())

# Check status
response = requests.get("http://localhost:8000/retraining/status")
status = response.json()
print(f"System running: {status['is_running']}")

# Trigger manual retraining
response = requests.post("http://localhost:8000/retraining/trigger/iris")
print(response.json())

# Get history
response = requests.get("http://localhost:8000/retraining/history?limit=5")
history = response.json()
for event in history['history']:
    print(f"{event['model_name']}: {event['trigger_type']} - {event['status']}")
```

## Triggers

### 1. Data Change Triggers

**How it works:**
- File system watcher monitors data directories
- Detects when data files are modified
- Triggers retraining for affected models
- Includes cooldown period to prevent duplicate triggers

**Example:**
```bash
# Modify data file to trigger retraining
echo "new_data" >> data/raw/iris.csv
# Retraining will be triggered automatically
```

### 2. Time-based Triggers

**How it works:**
- Monitors model age based on file modification time
- Triggers retraining when model exceeds maximum age
- Configurable per model

**Configuration:**
```python
config = RetrainingConfig(
    model_name="iris",
    max_model_age_hours=24,  # Retrain every 24 hours
    check_interval=300       # Check every 5 minutes
)
```

### 3. Performance-based Triggers

**How it works:**
- Monitors model performance metrics
- Compares recent performance with baseline
- Triggers retraining when degradation exceeds threshold

**Configuration:**
```python
config = RetrainingConfig(
    model_name="iris",
    performance_threshold=0.05,  # 5% degradation threshold
    check_interval=300
)
```

### 4. Manual Triggers

**Via CLI:**
```bash
python scripts/manage_retraining.py trigger iris
```

**Via API:**
```bash
curl -X POST http://localhost:8000/retraining/trigger/iris
```

## Monitoring & Metrics

### Prometheus Metrics

The retraining system exposes comprehensive Prometheus metrics:

```python
# Retraining triggers
model_retraining_triggered_total{model_name="iris", trigger_type="data_change"}

# Retraining duration
model_retraining_duration_seconds{model_name="iris", status="success"}

# Success/failure counts
model_retraining_success_total{model_name="iris"}
model_retraining_failure_total{model_name="iris", error_type="ValueError"}

# Model age
model_age_seconds{model_name="iris"}
```

### Retraining History

All retraining events are logged with detailed information:

```json
{
    "model_name": "iris",
    "trigger_type": "data_change",
    "trigger_time": "2024-01-01T12:00:00",
    "status": "success",
    "duration": 45.2,
    "data_hash": "abc123...",
    "description": "Data file modified: data/raw/iris.csv"
}
```

## Integration with MLflow

The retraining system integrates with MLflow for experiment tracking:

```python
# Each retraining creates a new MLflow experiment
experiment_name = f"{model_name}_retraining"

# Training results are logged
mlflow.log_metric("accuracy", accuracy_score)
mlflow.log_metric("training_time", duration)

# Model is saved and registered
mlflow.sklearn.log_model(model, "model")
```

## Error Handling

### Retraining Failures

When retraining fails:
1. **Error is logged** with full details
2. **Metrics are updated** (failure count, duration)
3. **Notification is sent** (if callback configured)
4. **System continues** monitoring other models

### Common Issues

**Insufficient Data:**
```bash
# Check data file
python scripts/manage_retraining.py status
# Verify data file exists and has sufficient records
```

**Model Loading Errors:**
```bash
# Check model files
ls -la src/models/saved/
# Verify model files are valid
```

**Permission Issues:**
```bash
# Check file permissions
ls -la data/raw/
# Ensure read/write access
```

## Production Considerations

### Security

1. **File Permissions**: Restrict access to data directories
2. **API Authentication**: Add authentication to retraining endpoints
3. **Network Security**: Use HTTPS for API communication
4. **Audit Logging**: Monitor all retraining activities

### Performance

1. **Resource Limits**: Set memory and CPU limits for retraining
2. **Concurrent Retraining**: Prevent multiple retraining of same model
3. **Background Processing**: Use separate processes for retraining
4. **Resource Monitoring**: Monitor system resources during retraining

### Reliability

1. **Error Recovery**: Automatic retry on failures
2. **Data Validation**: Validate data before retraining
3. **Model Validation**: Validate new models before deployment
4. **Rollback Capability**: Ability to revert to previous model

### Scaling

1. **Distributed Processing**: Use multiple workers for retraining
2. **Queue Management**: Queue retraining requests
3. **Load Balancing**: Distribute retraining across nodes
4. **Resource Pooling**: Share resources across retraining jobs

## Troubleshooting

### System Won't Start

```bash
# Check dependencies
pip list | grep watchdog

# Check configuration
python scripts/manage_retraining.py status

# Check logs
tail -f logs/retraining.log
```

### No Retraining Triggers

```bash
# Check if monitoring is running
python scripts/manage_retraining.py status

# Verify data files exist
ls -la data/raw/

# Check file permissions
ls -la data/raw/iris.csv
```

### Retraining Fails

```bash
# Check retraining history
python scripts/manage_retraining.py history

# Check model files
ls -la src/models/saved/

# Check MLflow logs
mlflow ui
```

### Performance Issues

```bash
# Monitor system resources
htop

# Check retraining duration
python scripts/manage_retraining.py history --limit 10

# Optimize configuration
python scripts/manage_retraining.py config --model iris
```

## Examples

### Custom Configuration

```python
from src.models.retraining import RetrainingConfig, start_retraining_monitoring

# Custom configuration
configs = {
    "custom_model": RetrainingConfig(
        model_name="custom_model",
        data_path="data/custom/data.csv",
        check_interval=600,  # 10 minutes
        min_data_size=200,
        max_model_age_hours=12,
        performance_threshold=0.03,
        enable_auto_retraining=True
    )
}

# Start monitoring
manager = start_retraining_monitoring(configs)
```

### Notification Callback

```python
def retraining_notification(model_name, status, duration, **kwargs):
    if status == "success":
        print(f"{model_name} retraining completed in {duration:.2f}s")
    else:
        print(f"{model_name} retraining failed: {kwargs.get('error')}")

# Add callback to configuration
config.notification_callback = retraining_notification
```

### Integration with CI/CD

```yaml
# GitHub Actions example
- name: Test Retraining System
  run: |
    python scripts/manage_retraining.py start
    sleep 30
    python scripts/manage_retraining.py status
    python scripts/manage_retraining.py trigger iris
    sleep 60
    python scripts/manage_retraining.py history
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/retraining.log`
3. Check retraining history with CLI
4. Verify configuration and data files
5. Monitor system resources and permissions
