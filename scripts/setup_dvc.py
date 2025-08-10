#!/usr/bin/env python3
"""
DVC setup script for YugenAI project
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_dvc_installed():
    """Check if DVC is installed"""
    try:
        result = subprocess.run(['dvc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ DVC is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ DVC is not installed or not accessible")
            return False
    except FileNotFoundError:
        print("❌ DVC is not installed")
        return False

def setup_dvc():
    """Setup DVC for the project"""
    print("🚀 Setting up DVC for YugenAI project")
    print("=" * 50)
    
    # Check if DVC is installed
    if not check_dvc_installed():
        print("\n📦 Installing DVC...")
        install_result = run_command("pip install dvc[gcs]", "Installing DVC with GCS support")
        if not install_result:
            print("❌ Failed to install DVC. Please install manually: pip install dvc[gcs]")
            return False
    
    # Initialize DVC if not already initialized
    if not Path('.dvc').exists():
        print("\n🔧 Initializing DVC...")
        init_result = run_command("dvc init", "Initializing DVC")
        if not init_result:
            return False
    
    # Add data files to DVC
    print("\n📁 Adding data files to DVC...")
    
    # Add raw data
    data_files = [
        'data/raw/housing.csv',
        'data/raw/iris.csv',
        'data/processed/housing_preprocessed.csv',
        'data/processed/iris_preprocessed.csv'
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            add_result = run_command(f"dvc add {data_file}", f"Adding {data_file} to DVC")
            if not add_result:
                print(f"⚠️ Warning: Failed to add {data_file}")
    
    # Add model files if they exist
    model_files = [
        'src/models/saved/housing_model.pkl',
        'src/models/saved/iris_model.pkl'
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            add_result = run_command(f"dvc add {model_file}", f"Adding {model_file} to DVC")
            if not add_result:
                print(f"⚠️ Warning: Failed to add {model_file}")
    
    # Setup remote storage
    print("\n☁️ Setting up remote storage...")
    
    # Check if GCS credentials are available
    gcs_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gcs_creds and Path(gcs_creds).exists():
        print(f"✅ GCS credentials found: {gcs_creds}")
        remote_result = run_command("dvc remote add gcs gs://rapid_care/mlops-dvc", "Adding GCS remote")
        if remote_result:
            run_command("dvc remote default gcs", "Setting GCS as default remote")
    else:
        print("⚠️ GCS credentials not found. Using local storage.")
        print("To use GCS, set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    # Create local storage directory
    local_storage = Path("dvc-storage")
    local_storage.mkdir(exist_ok=True)
    print(f"✅ Local storage directory created: {local_storage}")
    
    # Push to remote if available
    print("\n📤 Pushing to remote storage...")
    push_result = run_command("dvc push", "Pushing data to remote")
    if not push_result:
        print("⚠️ Warning: Failed to push to remote. Data is stored locally.")
    
    # Show DVC status
    print("\n📊 DVC Status:")
    status_result = run_command("dvc status", "Checking DVC status")
    
    print("\n🎉 DVC setup completed!")
    print("\n📋 Next steps:")
    print("1. Commit .dvc files to git: git add *.dvc")
    print("2. Run DVC pipelines: dvc repro")
    print("3. Push data changes: dvc push")
    print("4. Pull data: dvc pull")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("DVC Setup Script for YugenAI")
        print("Usage: python scripts/setup_dvc.py")
        print("\nThis script will:")
        print("- Check if DVC is installed")
        print("- Initialize DVC if needed")
        print("- Add data files to DVC tracking")
        print("- Setup remote storage")
        print("- Push data to remote")
        return
    
    success = setup_dvc()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 