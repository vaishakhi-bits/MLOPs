# ML API Monitoring with Prometheus and Grafana

This document describes the comprehensive monitoring setup for the ML API using Prometheus and Grafana.

## Overview

The monitoring stack provides real-time visibility into:
- **API Performance**: Request rates, response times, error rates
- **ML Model Metrics**: Prediction latency, confidence scores, model health
- **System Health**: Model load status, active requests, validation errors
- **Business Metrics**: Prediction distributions, class predictions

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ML API    │    │ Prometheus  │    │   Grafana   │
│  (Port 8000)│───▶│ (Port 9090) │───▶│ (Port 3000) │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Metrics Collected

### HTTP Request Metrics
- `http_requests_total`: Total request count by method, endpoint, status code
- `http_request_duration_seconds`: Request latency histograms
- `active_requests`: Number of currently active requests

### ML Model Metrics
- `ml_predictions_total`: Prediction count by model and status
- `ml_prediction_duration_seconds`: Prediction latency histograms
- `ml_prediction_confidence`: Confidence score distributions
- `model_load_status`: Model availability (1=loaded, 0=not_loaded)
- `model_last_update_timestamp`: Last model update time

### Validation & Error Metrics
- `validation_errors_total`: Validation error count by model and type
- `errors_total`: General error count by type and endpoint

### Business Metrics
- `housing_price_predictions`: Housing price prediction statistics
- `iris_class_predictions_total`: Iris class prediction counts

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- ML API running on port 8000

### 1. Start the Monitoring Stack

```bash
# Using the provided script
python scripts/start_monitoring.py start

# Or manually with Docker Compose
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access the Dashboards

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 3. View the ML API Dashboard

The ML API Dashboard will be automatically loaded in Grafana with the following panels:

#### Performance Panels
- **Request Rate**: HTTP requests per second by endpoint
- **Response Time**: 95th percentile response times
- **Prediction Rate**: ML predictions per second by model
- **Prediction Latency**: 95th percentile prediction times

#### Health Panels
- **Model Load Status**: Current model availability
- **Active Requests**: Number of concurrent requests
- **Validation Error Rate**: Input validation errors
- **Error Rate**: General error rates by type

#### Business Panels
- **Prediction Confidence**: Model confidence scores
- **Iris Class Predictions**: Distribution of predicted classes

## Dashboard Features

### Real-time Monitoring
- Auto-refresh every 5 seconds
- Configurable time ranges (1h, 6h, 24h, 7d)
- Interactive graphs with zoom and pan

### Alerting Ready
- Threshold-based alerting support
- Integration with notification systems
- Custom alert rules

### Multi-dimensional Analysis
- Filter by model, endpoint, error type
- Drill-down capabilities
- Comparative analysis

## Configuration

### Prometheus Configuration (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 10s
    metrics_path: '/metrics'
```

### Grafana Configuration

- **Datasource**: Automatically configured to connect to Prometheus
- **Dashboards**: Auto-provisioned from `grafana/dashboards/`
- **Authentication**: Default admin/admin (change in production)

## Advanced Usage

### Custom Queries

#### Request Rate by Status Code
```promql
rate(http_requests_total[5m])
```

#### Error Rate Percentage
```promql
(rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100
```

#### Model Success Rate
```promql
(rate(ml_predictions_total{status="success"}[5m]) / rate(ml_predictions_total[5m])) * 100
```

#### Average Prediction Confidence
```promql
histogram_quantile(0.5, rate(ml_prediction_confidence_bucket[5m]))
```

### Adding Custom Metrics

To add new metrics to the API:

1. **Define the metric** in `src/api/main.py`:
```python
CUSTOM_METRIC = Counter(
    'custom_metric_total',
    'Description of custom metric',
    ['label1', 'label2']
)
```

2. **Update the metric** in your code:
```python
CUSTOM_METRIC.labels(label1="value1", label2="value2").inc()
```

3. **Add to dashboard** in Grafana with appropriate PromQL query

### Alerting Rules

Create alerting rules in Prometheus:

```yaml
groups:
  - name: ml-api-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
```

## Troubleshooting

### Common Issues

#### Prometheus Can't Scrape Metrics
- Check if ML API is running on port 8000
- Verify `/metrics` endpoint is accessible
- Check firewall settings

#### Grafana Can't Connect to Prometheus
- Ensure Prometheus is running on port 9090
- Check network connectivity between containers
- Verify datasource configuration

#### No Data in Dashboard
- Check if metrics are being generated
- Verify Prometheus is scraping successfully
- Check time range settings in Grafana

### Debug Commands

```bash
# Check container status
docker-compose -f docker-compose.monitoring.yml ps

# View Prometheus logs
docker-compose -f docker-compose.monitoring.yml logs prometheus

# View Grafana logs
docker-compose -f docker-compose.monitoring.yml logs grafana

# Test metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## Production Considerations

### Security
- Change default Grafana credentials
- Use HTTPS for all endpoints
- Implement authentication for Prometheus
- Restrict network access

### Performance
- Adjust scrape intervals based on load
- Configure retention policies
- Use persistent volumes for data
- Monitor Prometheus resource usage

### Scaling
- Use Prometheus federation for multiple instances
- Implement service discovery
- Consider using Thanos or Cortex for long-term storage
- Set up alerting with Alertmanager

## Integration with CI/CD

### Automated Testing
```yaml
# Example GitHub Actions step
- name: Test Monitoring Setup
  run: |
    python scripts/start_monitoring.py start
    sleep 30
    curl -f http://localhost:9090/api/v1/targets
    curl -f http://localhost:3000/api/health
```

### Deployment
```yaml
# Example deployment script
- name: Deploy Monitoring
  run: |
    docker-compose -f docker-compose.monitoring.yml up -d
    # Wait for services to be ready
    # Verify dashboard is accessible
```

## Metrics Glossary

| Metric | Type | Description | Labels |
|--------|------|-------------|---------|
| `http_requests_total` | Counter | Total HTTP requests | method, endpoint, status_code |
| `http_request_duration_seconds` | Histogram | Request duration | method, endpoint |
| `ml_predictions_total` | Counter | Total ML predictions | model_name, status |
| `ml_prediction_duration_seconds` | Histogram | Prediction duration | model_name |
| `ml_prediction_confidence` | Histogram | Prediction confidence | model_name |
| `model_load_status` | Gauge | Model availability | model_name |
| `validation_errors_total` | Counter | Validation errors | model_name, error_type |
| `errors_total` | Counter | General errors | error_type, endpoint |

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Prometheus and Grafana documentation
3. Check container logs for detailed error messages
4. Verify network connectivity between services
