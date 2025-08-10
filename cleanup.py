#!/usr/bin/env python3
"""
Directory cleanup script for YugenAI project
"""

import os
import shutil
import glob
from pathlib import Path
import argparse

def cleanup_python_cache():
    """Remove Python cache directories and files"""
    print("Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"  Removed: {cache_path}")
                except Exception as e:
                    print(f"  Error removing {cache_path}: {e}")
    
    # Remove .pyc files
    for pyc_file in glob.glob('**/*.pyc', recursive=True):
        try:
            os.remove(pyc_file)
            print(f"  Removed: {pyc_file}")
        except Exception as e:
            print(f"  Error removing {pyc_file}: {e}")

def cleanup_test_artifacts():
    """Remove test artifacts and cache"""
    print("Cleaning test artifacts...")
    
    test_artifacts = [
        '.pytest_cache',
        '.coverage',
        'coverage.xml',
        'htmlcov',
        'test-results.xml',
        '.hypothesis'
    ]
    
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            try:
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                else:
                    os.remove(artifact)
                print(f"  Removed: {artifact}")
            except Exception as e:
                print(f"  Error removing {artifact}: {e}")

def cleanup_build_artifacts():
    """Remove build and distribution artifacts"""
    print("Cleaning build artifacts...")
    
    build_dirs = [
        'build',
        'dist',
        '*.egg-info',
        '*.egg'
    ]
    
    for pattern in build_dirs:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"  Removed: {path}")
            except Exception as e:
                print(f"  Error removing {path}: {e}")

def cleanup_temp_files():
    """Remove temporary files"""
    print("Cleaning temporary files...")
    
    temp_patterns = [
        '*.tmp',
        '*.temp',
        '*.swp',
        '*.swo',
        '*~',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            try:
                os.remove(temp_file)
                print(f"  Removed: {temp_file}")
            except Exception as e:
                print(f"  Error removing {temp_file}: {e}")

def organize_artifacts():
    """Organize artifacts into proper directories"""
    print("Organizing artifacts...")
    
    artifacts_dir = Path('artifacts')
    if not artifacts_dir.exists():
        print("  No artifacts directory found")
        return
    
    # Create subdirectories if they don't exist
    experiments_dir = artifacts_dir / 'experiments'
    models_dir = artifacts_dir / 'models'
    reports_dir = artifacts_dir / 'reports'
    
    experiments_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Move files to appropriate directories
    for file_path in artifacts_dir.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix in ['.png', '.jpg', '.jpeg', '.svg']:
                    # Move image files to experiments
                    shutil.move(str(file_path), str(experiments_dir / file_path.name))
                    print(f"  Moved {file_path.name} to experiments/")
                elif file_path.suffix in ['.csv', '.txt', '.json']:
                    # Move data files to reports
                    shutil.move(str(file_path), str(reports_dir / file_path.name))
                    print(f"  Moved {file_path.name} to reports/")
                elif file_path.suffix in ['.pkl', '.joblib', '.h5', '.hdf5', '.onnx']:
                    # Move model files to models
                    shutil.move(str(file_path), str(models_dir / file_path.name))
                    print(f"  Moved {file_path.name} to models/")
            except Exception as e:
                print(f"  Error moving {file_path.name}: {e}")

def organize_logs():
    """Organize logs into proper directories"""
    print("Organizing logs...")
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("  No logs directory found")
        return
    
    # Create subdirectories if they don't exist
    api_logs_dir = logs_dir / 'api'
    predictions_logs_dir = logs_dir / 'predictions'
    
    api_logs_dir.mkdir(exist_ok=True)
    predictions_logs_dir.mkdir(exist_ok=True)
    
    # Move log files to appropriate directories
    for file_path in logs_dir.iterdir():
        if file_path.is_file():
            try:
                if 'api' in file_path.name.lower():
                    # Move API logs
                    shutil.move(str(file_path), str(api_logs_dir / file_path.name))
                    print(f"  Moved {file_path.name} to api/")
                elif any(keyword in file_path.name.lower() for keyword in ['prediction', 'request', 'response']):
                    # Move prediction logs
                    shutil.move(str(file_path), str(predictions_logs_dir / file_path.name))
                    print(f"  Moved {file_path.name} to predictions/")
            except Exception as e:
                print(f"  Error moving {file_path.name}: {e}")

def cleanup_mlflow():
    """Clean up MLflow artifacts"""
    print("Cleaning MLflow artifacts...")
    
    mlflow_dirs = [
        '.mlruns',
        'mlruns',
        'mlflow_logs'
    ]
    
    for mlflow_dir in mlflow_dirs:
        if os.path.exists(mlflow_dir):
            try:
                shutil.rmtree(mlflow_dir)
                print(f"  Removed: {mlflow_dir}")
            except Exception as e:
                print(f"  Error removing {mlflow_dir}: {e}")

def show_directory_structure():
    """Show current directory structure"""
    print("Current directory structure:")
    print()
    
    def print_tree(path, prefix="", is_last=True):
        if not os.path.exists(path):
            return
        
        if os.path.isdir(path):
            print(f"{prefix}{'└── ' if is_last else '├── '}{os.path.basename(path)}/")
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                item_path = os.path.join(path, item)
                is_last_item = i == len(items) - 1
                print_tree(item_path, prefix + ("    " if is_last else "│   "), is_last_item)
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}{os.path.basename(path)}")
    
    print_tree(".")

def main():
    parser = argparse.ArgumentParser(description="Clean up YugenAI project directory")
    parser.add_argument("--all", action="store_true", help="Run all cleanup operations")
    parser.add_argument("--cache", action="store_true", help="Clean Python cache files")
    parser.add_argument("--tests", action="store_true", help="Clean test artifacts")
    parser.add_argument("--build", action="store_true", help="Clean build artifacts")
    parser.add_argument("--temp", action="store_true", help="Clean temporary files")
    parser.add_argument("--artifacts", action="store_true", help="Organize artifacts")
    parser.add_argument("--logs", action="store_true", help="Organize logs")
    parser.add_argument("--mlflow", action="store_true", help="Clean MLflow artifacts")
    parser.add_argument("--show", action="store_true", help="Show directory structure")
    
    args = parser.parse_args()
    
    print("YugenAI Project Cleanup")
    print("=" * 50)
    
    if args.show:
        show_directory_structure()
        return
    
    if args.all or args.cache:
        cleanup_python_cache()
    
    if args.all or args.tests:
        cleanup_test_artifacts()
    
    if args.all or args.build:
        cleanup_build_artifacts()
    
    if args.all or args.temp:
        cleanup_temp_files()
    
    if args.all or args.artifacts:
        organize_artifacts()
    
    if args.all or args.logs:
        organize_logs()
    
    if args.all or args.mlflow:
        cleanup_mlflow()
    
    if not any([args.all, args.cache, args.tests, args.build, args.temp, 
                args.artifacts, args.logs, args.mlflow, args.show]):
        print("No cleanup operations specified. Use --help for options.")
        print("Use --all to run all cleanup operations.")
        return
    
    print("\nCleanup completed!")
    print("\nTips:")
    print("  - Run this script regularly to keep your project clean")
    print("  - Use --show to view current directory structure")
    print("  - Use --all for comprehensive cleanup")

if __name__ == "__main__":
    main() 