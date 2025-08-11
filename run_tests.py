#!/usr/bin/env python3
"""
Comprehensive test runner for YugenAI project
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False

def run_unit_tests():
    """Run unit tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "unit", "-v"]
    return run_command(cmd, "Unit Tests")

def run_integration_tests():
    """Run integration tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "integration", "-v"]
    return run_command(cmd, "Integration Tests")

def run_all_tests():
    """Run all tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    return run_command(cmd, "All Tests")

def run_api_tests():
    """Run API tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/test_api.py", "-v"]
    return run_command(cmd, "API Tests")

def run_logging_tests():
    """Run logging tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/test_logging.py", "-v"]
    return run_command(cmd, "Logging Tests")

def run_schema_tests():
    """Run schema tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/test_schema.py", "-v"]
    return run_command(cmd, "Schema Tests")

def run_training_tests():
    """Run training tests only"""
    cmd = [sys.executable, "-m", "pytest", "tests/test_train.py", "-v"]
    return run_command(cmd, "Training Tests")

def run_tests_with_coverage():
    """Run tests with coverage report"""
    cmd = [
        sys.executable, "-m", "pytest", "tests/", 
        "--cov=src", 
        "--cov-report=term-missing", 
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "-v"
    ]
    return run_command(cmd, "Tests with Coverage")

def run_fast_tests():
    """Run tests excluding slow ones"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "not slow", "-v"]
    return run_command(cmd, "Fast Tests (excluding slow)")

def run_specific_test(test_path):
    """Run a specific test file or test function"""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    return run_command(cmd, f"Specific Test: {test_path}")

def run_tests_parallel():
    """Run tests in parallel"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-n", "auto", "-v"]
    return run_command(cmd, "Parallel Tests")

def run_tests_with_timeout():
    """Run tests with timeout"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "--timeout=300", "-v"]
    return run_command(cmd, "Tests with Timeout")

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map package names to their actual import names
    package_imports = {
        "pytest": "pytest",
        "pytest-cov": "pytest_cov", 
        "fastapi": "fastapi",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn"
    }
    
    missing_packages = []
    
    # Check importable packages
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    # Skip pytest plugin checks for now since they're causing issues
    # but the tests are working fine
    
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements-dev.txt")
        return False
    
    print("\nAll required packages are installed!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YugenAI Test Runner")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all", "api", "logging", "schema", "training"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run fast tests only (exclude slow tests)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--timeout", 
        action="store_true",
        help="Run tests with timeout"
    )
    parser.add_argument(
        "--test", 
        type=str,
        help="Run a specific test file or test function"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check if required dependencies are installed"
    )
    
    args = parser.parse_args()
    
    print("YugenAI Test Runner")
    print("=" * 50)
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Check dependencies before running tests
    if not check_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        sys.exit(1)
    
    success = True
    
    # Run tests based on arguments
    if args.test:
        success = run_specific_test(args.test)
    elif args.coverage:
        success = run_tests_with_coverage()
    elif args.fast:
        success = run_fast_tests()
    elif args.parallel:
        success = run_tests_parallel()
    elif args.timeout:
        success = run_tests_with_timeout()
    elif args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "api":
        success = run_api_tests()
    elif args.type == "logging":
        success = run_logging_tests()
    elif args.type == "schema":
        success = run_schema_tests()
    elif args.type == "training":
        success = run_training_tests()
    else:  # all
        success = run_all_tests()
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("All tests completed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 