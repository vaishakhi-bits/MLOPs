#!/usr/bin/env python3
"""
Example script demonstrating the monitoring setup for the ML API.
This script generates sample traffic to populate the metrics and dashboard.
"""

import requests
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def make_iris_prediction():
    """Make a sample Iris prediction"""
    iris_data = {
        "SepalLengthCm": random.uniform(4.0, 8.0),
        "SepalWidthCm": random.uniform(2.0, 5.0),
        "PetalLengthCm": random.uniform(1.0, 7.0),
        "PetalWidthCm": random.uniform(0.1, 2.5)
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_iris",
            json=iris_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"Iris: {result.get('predicted_class')} (confidence: {result.get('confidence_score', 'N/A')})"
        else:
            return f"Iris: Error {response.status_code}"
            
    except Exception as e:
        return f"Iris: Exception - {str(e)}"

def make_housing_prediction():
    """Make a sample Housing prediction"""
    housing_data = {
        "longitude": random.uniform(-124, -114),
        "latitude": random.uniform(32, 42),
        "housing_median_age": random.uniform(0, 100),
        "total_rooms": random.uniform(0, 10000),
        "total_bedrooms": random.uniform(0, 5000),
        "population": random.uniform(0, 100000),
        "households": random.uniform(0, 5000),
        "median_income": random.uniform(0, 20),
        "ocean_proximity": random.choice(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_housing",
            json=housing_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"Housing: ${result.get('predicted_price', 0):,.2f} (confidence: {result.get('confidence_score', 'N/A')})"
        else:
            return f"Housing: Error {response.status_code}"
            
    except Exception as e:
        return f"Housing: Exception - {str(e)}"

def make_invalid_request():
    """Make an invalid request to test validation error metrics"""
    invalid_data = {
        "SepalLengthCm": -1.0,  # Invalid negative value
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_iris",
            json=invalid_data,
            timeout=5
        )
        
        return f"Invalid: Status {response.status_code}"
        
    except Exception as e:
        return f"Invalid: Exception - {str(e)}"

def check_health():
    """Check API health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"🏥 API Status: {health_data.get('status')}")
            print(f"🏥 API Version: {health_data.get('version')}")
            
            models = health_data.get('models', [])
            for model in models:
                status_icon = "✅" if model.get('loaded') else "❌"
                print(f"🤖 {status_icon} {model.get('model_name')}: {model.get('status')}")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def generate_traffic(duration=60, requests_per_second=2):
    """Generate traffic for the specified duration"""
    print(f"🚀 Generating traffic for {duration} seconds at {requests_per_second} req/s...")
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration:
        batch_start = time.time()
        
        # Make a batch of requests
        requests_this_batch = []
        
        # 70% valid Iris predictions
        for _ in range(int(requests_per_second * 0.7)):
            requests_this_batch.append(("iris", make_iris_prediction))
        
        # 20% valid Housing predictions
        for _ in range(int(requests_per_second * 0.2)):
            requests_this_batch.append(("housing", make_housing_prediction))
        
        # 10% invalid requests
        for _ in range(int(requests_per_second * 0.1)):
            requests_this_batch.append(("invalid", make_invalid_request))
        
        # Execute requests in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(func) for _, func in requests_this_batch]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    request_count += 1
                    print(f"📊 Request {request_count}: {result}")
                except Exception as e:
                    print(f"❌ Request failed: {e}")
        
        # Sleep to maintain rate
        elapsed = time.time() - batch_start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    
    print(f"✅ Generated {request_count} requests in {duration} seconds")

def show_metrics_summary():
    """Show a summary of current metrics"""
    print("\n📊 Current Metrics Summary:")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        
        if response.status_code == 200:
            metrics = response.text
            
            # Count different metric types
            metric_counts = {}
            for line in metrics.split('\n'):
                if line and not line.startswith('#'):
                    metric_name = line.split('{')[0] if '{' in line else line.split(' ')[0]
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
            
            print("📈 Available Metrics:")
            for metric, count in sorted(metric_counts.items()):
                print(f"   • {metric}: {count} series")
            
            print(f"\n📊 Total metric series: {sum(metric_counts.values())}")
            
        else:
            print(f"❌ Could not fetch metrics: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")

def main():
    """Main function"""
    print("🎯 ML API Monitoring Example")
    print("=" * 40)
    
    # Check if API is running
    print("🔍 Checking API health...")
    if not check_health():
        print("❌ API is not running or not accessible")
        print("💡 Please start the API first:")
        print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        return
    
    print("\n✅ API is healthy and ready!")
    
    # Show initial metrics
    show_metrics_summary()
    
    # Generate traffic
    print("\n" + "=" * 40)
    generate_traffic(duration=30, requests_per_second=3)
    
    # Show final metrics
    print("\n" + "=" * 40)
    show_metrics_summary()
    
    print("\n🎉 Traffic generation complete!")
    print("\n📊 Next steps:")
    print("   1. Start Prometheus and Grafana:")
    print("      python scripts/start_monitoring.py start")
    print("   2. Access Grafana at http://localhost:3000")
    print("   3. View the ML API Dashboard")
    print("   4. Explore the metrics in Prometheus at http://localhost:9090")

if __name__ == "__main__":
    main()
