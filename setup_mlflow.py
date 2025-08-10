#!/usr/bin/env python3
"""
Setup MLflow with correct directory structure for YugenAI project
"""

import os
import sys
from pathlib import Path

def setup_mlflow_directory():
    """Ensure MLflow uses the correct directory structure"""
    
    # Create the standard mlruns directory
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    
    # Remove any duplicate directories
    duplicate_dirs = [".mlruns", "mlflow_logs"]
    for duplicate_dir in duplicate_dirs:
        if os.path.exists(duplicate_dir):
            import shutil
            try:
                shutil.rmtree(duplicate_dir)
                print(f"Removed duplicate directory: {duplicate_dir}")
            except Exception as e:
                print(f"Error removing {duplicate_dir}: {e}")
    
    # Set environment variable for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    
    print("MLflow directory setup complete:")
    print(f"   Tracking URI: file:./mlruns")
    print(f"   Directory: {mlruns_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    setup_mlflow_directory() 