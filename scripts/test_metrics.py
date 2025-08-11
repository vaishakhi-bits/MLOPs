#!/usr/bin/env python3
"""
Test script to verify Prometheus metrics are working correctly.
"""

import requests
import time
import json
from pathlib import Path

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    print("Testing Prometheus Metrics Endpoint...")
    
    try:
        # Test metrics endpoint
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        
        if response.status_code == 200:
            print("Metrics endpoint is accessible")
            
            # Check for key metrics
            metrics_content = response.text
            key_metrics = [
                "http_requests_total",
                "http_request_duration_seconds",
                "ml_predictions_total",
                "ml_prediction_duration_seconds",
                "ml_prediction_confidence",
                "model_load_status",
                "validation_errors_total",
                "errors_total",
                "active_requests"
            ]
            
            found_metrics = []
            for metric in key_metrics:
                if metric in metrics_content:
                    found_metrics.append(metric)
                    print(f"Found metric: {metric}")
                else:
                    print(f"Missing metric: {metric}")
            
            print(f"\nFound {len(found_metrics)}/{len(key_metrics)} expected metrics")
            
            return len(found_metrics) >= len(key_metrics) * 0.8  # 80% threshold
            
        else:
            print(f"Metrics endpoint returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to metrics endpoint. Is the API running?")
        return False
    except Exception as e:
        print(f"Error testing metrics endpoint: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    print("\nTesting Health Endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print("Health endpoint is accessible")
            
            health_data = response.json()
            print(f"API Status: {health_data.get('status', 'unknown')}")
            print(f"API Version: {health_data.get('version', 'unknown')}")
            
            models = health_data.get('models', [])
            for model in models:
                print(f"Model: {model.get('model_name')} - Status: {model.get('status')} - Loaded: {model.get('loaded')}")
            
            return True
        else:
            print(f"Health endpoint returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to health endpoint. Is the API running?")
        return False
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_prediction_endpoints():
    """Test prediction endpoints to generate metrics"""
    print("\nTesting Prediction Endpoints...")
    
    # Test Iris prediction
    iris_data = {
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_iris",
            json=iris_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("Iris prediction endpoint working")
            result = response.json()
            print(f"Predicted class: {result.get('predicted_class')}")
        else:
            print(f"Iris prediction returned status: {response.status_code}")
            
    except Exception as e:
        print(f"Error testing Iris prediction: {e}")
    
    # Test Housing prediction
    housing_data = {
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
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_housing",
            json=housing_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("Housing prediction endpoint working")
            result = response.json()
            print(f"Predicted price: ${result.get('predicted_price'):,.2f}")
        else:
            print(f"Housing prediction returned status: {response.status_code}")
            
    except Exception as e:
        print(f"Error testing Housing prediction: {e}")

def main():
    """Main test function"""
    print("ML API Metrics Test Suite")
    print("=" * 40)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test prediction endpoints to generate metrics
    test_prediction_endpoints()
    
    # Wait a moment for metrics to be updated
    print("\n⏳ Waiting for metrics to update...")
    time.sleep(2)
    
    # Test metrics endpoint
    metrics_ok = test_metrics_endpoint()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"   Health Endpoint: {'PASS' if health_ok else 'FAIL'}")
    print(f"   Metrics Endpoint: {'PASS' if metrics_ok else 'FAIL'}")
    
    if health_ok and metrics_ok:
        print("\nAll tests passed! Monitoring setup is ready.")
        print("\nNext steps:")
        print("   1. Start Prometheus and Grafana:")
        print("      python scripts/start_monitoring.py start")
        print("   2. Access Grafana at http://localhost:3000")
        print("   3. View the ML API Dashboard")
    else:
        print("\nSome tests failed. Please check the API is running correctly.")
        print("\nTroubleshooting:")
        print("   1. Ensure the API is running on port 8000")
        print("   2. Check if models are loaded correctly")
        print("   3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()
