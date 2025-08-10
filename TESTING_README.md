# YugenAI Testing Guide

This document provides a comprehensive guide to the testing system for the YugenAI project, including how to run tests, write new tests, and understand test coverage.

## Overview

The testing system is designed to ensure code quality, reliability, and maintainability through:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and API endpoints
- **Schema Validation Tests**: Test data validation and serialization
- **Logging Tests**: Test logging functionality and output
- **Training Tests**: Test model training and evaluation pipelines

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_api.py              # API endpoint tests
├── test_logging.py          # Logging functionality tests
├── test_schema.py           # Schema validation tests
├── test_train.py            # Model training tests
└── __pycache__/
```

## Quick Start

### 1. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install specific testing packages
pip install pytest pytest-cov pytest-xdist pytest-timeout
```

### 2. Run All Tests

```bash
# Using the test runner script
python run_tests.py

# Or using pytest directly
pytest tests/ -v
```

### 3. Run Specific Test Types

```bash
# Unit tests only
python run_tests.py --type unit

# Integration tests only
python run_tests.py --type integration

# API tests only
python run_tests.py --type api

# Logging tests only
python run_tests.py --type logging

# Schema tests only
python run_tests.py --type schema

# Training tests only
python run_tests.py --type training
```

## Test Runner Options

The `run_tests.py` script provides several options for running tests:

### Basic Usage

```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run fast tests only (exclude slow tests)
python run_tests.py --fast

# Run tests in parallel
python run_tests.py --parallel

# Run tests with timeout
python run_tests.py --timeout

# Run a specific test file
python run_tests.py --test tests/test_api.py

# Run a specific test function
python run_tests.py --test tests/test_api.py::TestIrisPrediction::test_predict_iris_success

# Check dependencies
python run_tests.py --check-deps
```

### Advanced Usage

```bash
# Run tests with detailed output
pytest tests/ -v -s

# Run tests with coverage and generate HTML report
pytest tests/ --cov=src --cov-report=html:htmlcov

# Run tests in parallel with 4 workers
pytest tests/ -n 4

# Run tests with timeout of 5 minutes
pytest tests/ --timeout=300

# Run only tests marked as "slow"
pytest tests/ -m slow

# Run tests excluding "slow" tests
pytest tests/ -m "not slow"
```

## Test Categories

### 1. Unit Tests (`test_logging.py`, `test_schema.py`)

Unit tests focus on testing individual functions and classes in isolation:

- **Logger Setup Tests**: Test logger configuration and initialization
- **Prediction Logger Tests**: Test prediction request/response logging
- **Schema Validation Tests**: Test Pydantic model validation
- **Data Processing Tests**: Test data transformation and validation

### 2. Integration Tests (`test_api.py`)

Integration tests verify that components work together correctly:

- **API Endpoint Tests**: Test FastAPI endpoints with mocked models
- **Request/Response Tests**: Test complete request-response cycles
- **Error Handling Tests**: Test error scenarios and edge cases
- **Middleware Tests**: Test CORS, metrics, and logging middleware

### 3. Training Tests (`test_train.py`)

Training tests verify model training and evaluation functionality:

- **Data Preprocessing Tests**: Test data cleaning and transformation
- **Model Training Tests**: Test model training pipelines
- **Model Evaluation Tests**: Test metrics calculation and validation
- **End-to-End Tests**: Test complete training workflows

## Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import patch, Mock

class TestFeatureName:
    """Test class for a specific feature"""
    
    def test_specific_functionality(self, fixture_name):
        """Test a specific piece of functionality"""
        # Arrange
        expected = "expected_value"
        
        # Act
        result = function_under_test()
        
        # Assert
        assert result == expected
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a long time to run"""
        # Test implementation
        pass
```

### Using Fixtures

```python
def test_with_fixtures(sample_iris_features, mock_iris_model):
    """Test using shared fixtures"""
    # Use the fixtures directly
    assert sample_iris_features["SepalLengthCm"] == 5.1
    assert mock_iris_model is not None
```

### Mocking External Dependencies

```python
@patch('src.api.main.iris_model')
def test_with_mocked_model(mock_model):
    """Test with mocked external dependency"""
    # Configure mock
    mock_model.predict.return_value = [0]
    
    # Test implementation
    # ...
```

### Testing API Endpoints

```python
def test_api_endpoint(client, sample_iris_features, mock_iris_model):
    """Test API endpoint with FastAPI TestClient"""
    with patch('src.api.main.iris_model', mock_iris_model):
        response = client.post("/predict_iris", json=sample_iris_features)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
```

## Test Coverage

### Running Coverage Reports

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html:htmlcov

# Generate XML coverage report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml:coverage.xml
```

### Coverage Targets

- **Overall Coverage**: Aim for >80%
- **Critical Paths**: Aim for >90%
- **API Endpoints**: Aim for >95%

### Viewing Coverage Reports

After running coverage, you can view the HTML report:

```bash
# Open HTML coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

The `pytest.ini` file configures:

- **Test Discovery**: Where to find tests
- **Markers**: Custom test markers for categorization
- **Execution Options**: Verbosity, warnings, etc.
- **Coverage Settings**: Coverage reporting options

### Shared Fixtures (`conftest.py`)

The `conftest.py` file provides:

- **Test Data**: Sample datasets for testing
- **Mock Objects**: Pre-configured mocks
- **Temporary Directories**: Clean test environments
- **Test Data Generators**: Utilities for creating test data

## Test Markers

Use markers to categorize and filter tests:

```python
@pytest.mark.unit
def test_unit_function():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration_function():
    """Integration test"""
    pass

@pytest.mark.slow
def test_slow_function():
    """Slow running test"""
    pass

@pytest.mark.api
def test_api_endpoint():
    """API test"""
    pass
```

### Available Markers

- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow running tests
- `api`: API endpoint tests
- `logging`: Logging functionality tests
- `schema`: Schema validation tests
- `training`: Model training tests

## Best Practices

### 1. Test Organization

- **One test per function**: Each test should verify one specific behavior
- **Descriptive names**: Use clear, descriptive test names
- **Arrange-Act-Assert**: Structure tests with clear sections
- **Group related tests**: Use test classes to group related functionality

### 2. Test Data

- **Use fixtures**: Leverage shared fixtures for common test data
- **Minimal test data**: Use the smallest dataset that tests the functionality
- **Realistic data**: Use realistic but simple test data
- **Edge cases**: Include tests for edge cases and error conditions

### 3. Mocking

- **Mock external dependencies**: Don't rely on external services in tests
- **Mock at the right level**: Mock at the boundary of your system
- **Verify mock calls**: Ensure mocks are called as expected
- **Use realistic mocks**: Configure mocks to return realistic data

### 4. Assertions

- **Specific assertions**: Use specific assertions rather than generic ones
- **Meaningful messages**: Include helpful error messages in assertions
- **Test one thing**: Each assertion should test one specific aspect
- **Use appropriate matchers**: Use the right assertion for the data type

### 5. Performance

- **Fast tests**: Keep tests fast to encourage frequent running
- **Parallel execution**: Use parallel execution for large test suites
- **Timeout protection**: Add timeouts to prevent hanging tests
- **Resource cleanup**: Clean up resources after tests

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with maximum verbosity
pytest tests/ -v -s

# Run a specific test with debug output
pytest tests/test_api.py::test_specific_function -v -s

# Run with print statements visible
pytest tests/ -s
```

### Using Pytest Debugger

```python
def test_with_debugger():
    """Test with debugger breakpoint"""
    import pdb; pdb.set_trace()  # Debugger will stop here
    # Test implementation
```

### Common Issues and Solutions

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Issues**: Check that mocks are configured correctly
3. **Fixture Errors**: Verify fixture names and dependencies
4. **Timeout Issues**: Increase timeout or optimize slow tests

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          python run_tests.py --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/, -v]
```

## Additional Resources

### Useful Commands

```bash
# List all available tests
pytest --collect-only

# Show test coverage for specific files
pytest --cov=src.api --cov-report=term-missing

# Run tests and generate JUnit XML report
pytest tests/ --junitxml=test-results.xml

# Run tests with custom markers
pytest tests/ -m "unit and not slow"
```

### Testing Tools

- **pytest**: Main testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-timeout**: Test timeout management
- **pytest-mock**: Enhanced mocking capabilities

### Documentation

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic Validation](https://pydantic-docs.helpmanual.io/usage/validation_decorator/)

## Contributing

When adding new tests:

1. **Follow existing patterns**: Use the same structure and style
2. **Add appropriate markers**: Mark tests with relevant categories
3. **Update documentation**: Document new test functionality
4. **Maintain coverage**: Ensure new code is adequately tested
5. **Run all tests**: Verify that new tests don't break existing ones

## Support

If you encounter issues with the testing system:

1. Check the [Common Issues](#common-issues-and-solutions) section
2. Review the test logs for specific error messages
3. Ensure all dependencies are installed correctly
4. Verify that the test environment is properly configured
5. Consult the pytest documentation for framework-specific issues 