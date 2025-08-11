#!/usr/bin/env python3
"""
Example script demonstrating the model retraining system.
This script shows how to set up and use automatic retraining triggers.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.retraining import (
    RetrainingConfig, 
    ModelRetrainingManager, 
    create_default_configs
)

def create_sample_data():
    """Create sample data files for demonstration"""
    print("📊 Creating sample data files...")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample Iris data
    iris_data = pd.DataFrame({
        'SepalLengthCm': np.random.uniform(4.0, 8.0, 150),
        'SepalWidthCm': np.random.uniform(2.0, 5.0, 150),
        'PetalLengthCm': np.random.uniform(1.0, 7.0, 150),
        'PetalWidthCm': np.random.uniform(0.1, 2.5, 150),
        'Species': np.random.choice(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 150)
    })
    
    iris_data.to_csv(data_dir / "iris.csv", index=False)
    print(f"Created {len(iris_data)} Iris records")
    
    # Create sample Housing data
    housing_data = pd.DataFrame({
        'longitude': np.random.uniform(-124, -114, 200),
        'latitude': np.random.uniform(32, 42, 200),
        'housing_median_age': np.random.uniform(0, 100, 200),
        'total_rooms': np.random.uniform(0, 10000, 200),
        'total_bedrooms': np.random.uniform(0, 5000, 200),
        'population': np.random.uniform(0, 100000, 200),
        'households': np.random.uniform(0, 5000, 200),
        'median_income': np.random.uniform(0, 20, 200),
        'ocean_proximity': np.random.choice(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], 200),
        'median_house_value': np.random.uniform(50000, 500000, 200)
    })
    
    housing_data.to_csv(data_dir / "housing.csv", index=False)
    print(f"Created {len(housing_data)} Housing records")

def notification_callback(model_name, status, duration, **kwargs):
    """Example notification callback for retraining events"""
    if status == "success":
        print(f"🎉 {model_name.upper()} retraining completed successfully in {duration:.2f}s")
        if 'results' in kwargs:
            print(f"   Results: {kwargs['results']}")
    else:
        print(f"{model_name.upper()} retraining failed after {duration:.2f}s")
        if 'error' in kwargs:
            print(f"   Error: {kwargs['error']}")

def demonstrate_data_change_trigger(manager):
    """Demonstrate data change trigger by modifying data files"""
    print("\nDemonstrating Data Change Trigger...")
    
    # Wait a moment for system to settle
    time.sleep(2)
    
    # Modify Iris data file
    iris_file = Path("data/raw/iris.csv")
    if iris_file.exists():
        # Add a new record to trigger retraining
        new_record = pd.DataFrame({
            'SepalLengthCm': [5.5],
            'SepalWidthCm': [3.8],
            'PetalLengthCm': [1.6],
            'PetalWidthCm': [0.3],
            'Species': ['Iris-setosa']
        })
        
        # Append to existing file
        new_record.to_csv(iris_file, mode='a', header=False, index=False)
        print("Modified Iris data file - retraining should be triggered")
    
    # Wait for retraining to complete
    print("Waiting for retraining to complete...")
    time.sleep(10)

def demonstrate_manual_trigger(manager):
    """Demonstrate manual retraining trigger"""
    print("\nDemonstrating Manual Trigger...")
    
    # Trigger manual retraining for Iris
    manager.trigger_retraining(
        "iris",
        "manual",
        "Manual trigger for demonstration"
    )
    
    print("Manual retraining triggered for Iris")
    print("Waiting for retraining to complete...")
    time.sleep(10)

def demonstrate_api_endpoints():
    """Demonstrate API endpoints for retraining management"""
    print("\nDemonstrating API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Start retraining system via API
        print("Starting retraining system via API...")
        response = requests.post(f"{base_url}/retraining/start")
        if response.status_code == 200:
            print("Retraining system started via API")
        else:
            print(f"Failed to start retraining system: {response.status_code}")
            return
        
        # Check status
        print("Checking retraining status...")
        response = requests.get(f"{base_url}/retraining/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   System running: {status['is_running']}")
            print(f"   Models: {list(status['models'].keys())}")
        
        # Trigger manual retraining via API
        print("Triggering manual retraining via API...")
        response = requests.post(f"{base_url}/retraining/trigger/iris")
        if response.status_code == 200:
            result = response.json()
            print(f"{result['message']}")
        
        # Get retraining history
        print("Getting retraining history...")
        response = requests.get(f"{base_url}/retraining/history?limit=5")
        if response.status_code == 200:
            history = response.json()
            print(f"   Found {history['total']} retraining events")
            for event in history['history'][:3]:
                print(f"   • {event['model_name']}: {event['trigger_type']} - {event['status']}")
        
        # Stop retraining system
        print("Stopping retraining system...")
        response = requests.post(f"{base_url}/retraining/stop")
        if response.status_code == 200:
            print("Retraining system stopped")
    
    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"API demonstration failed: {e}")

def show_retraining_history(manager):
    """Show retraining history"""
    print("\nRetraining History:")
    history = manager.get_retraining_history(limit=10)
    
    if not history:
        print("   No retraining events recorded yet")
        return
    
    for i, record in enumerate(history, 1):
        status_icon = "✅" if record["status"] == "success" else "❌"
        duration = f"{record['duration']:.2f}s" if "duration" in record else "N/A"
        
        print(f"   {i:2d}. {status_icon} {record['model_name'].upper()}")
        print(f"       Trigger: {record['trigger_type']}")
        print(f"       Time: {record['trigger_time']}")
        print(f"       Duration: {duration}")
        print(f"       Description: {record.get('description', 'N/A')}")
        print()

def main():
    """Main demonstration function"""
    print("Model Retraining System Demonstration")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Create custom configuration with notification callback
    print("\nSetting up retraining configuration...")
    configs = create_default_configs()
    
    # Add notification callback to both models
    for config in configs.values():
        config.notification_callback = notification_callback
        config.check_interval = 30  # Check every 30 seconds for demo
        config.max_model_age_hours = 1  # Retrain every hour for demo
    
    # Start retraining manager
    print("Starting retraining manager...")
    manager = ModelRetrainingManager(configs)
    manager.start_monitoring()
    
    print("Retraining system started!")
    print(f"Monitoring {len(configs)} models:")
    for model_name, config in configs.items():
        print(f"   • {model_name}: {config.data_path}")
    
    try:
        # Demonstrate different triggers
        demonstrate_data_change_trigger(manager)
        demonstrate_manual_trigger(manager)
        
        # Show history
        show_retraining_history(manager)
        
        # Demonstrate API endpoints (if server is running)
        demonstrate_api_endpoints()
        
        print("\nDemonstration completed!")
        print("\nNext steps:")
        print("   1. Check retraining history: python scripts/manage_retraining.py history")
        print("   2. Monitor system status: python scripts/manage_retraining.py status")
        print("   3. Configure settings: python scripts/manage_retraining.py config --model iris")
        print("   4. View Prometheus metrics: http://localhost:9090")
        print("   5. Check MLflow experiments: mlflow ui")
        
        # Keep running for a while to show monitoring
        print("\nKeeping system running for 30 seconds to demonstrate monitoring...")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    
    finally:
        # Stop the retraining system
        print("\nStopping retraining system...")
        manager.stop_monitoring()
        print("Retraining system stopped")

if __name__ == "__main__":
    main()
