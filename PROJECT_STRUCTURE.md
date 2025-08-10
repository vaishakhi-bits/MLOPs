# YugenAI Project Structure

This document outlines the organized directory structure of the YugenAI project.

## Root Directory

```
yugenai/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       └── tests.yml          # Comprehensive test workflow
├── .dvc/                       # DVC configuration
│   ├── config                 # DVC settings
│   └── .gitignore             # DVC ignore rules
├── dvc.yaml                   # DVC pipeline definition
├── dvc-storage/               # Local DVC storage (gitignored)
├── src/                       # Source code
│   ├── api/                   # FastAPI application
│   │   ├── main.py           # Main API endpoints
│   │   └── schema.py         # Pydantic models
│   ├── models/               # Model-related code
│   │   ├── saved/            # Saved model files
│   │   ├── train_iris.py     # Iris model training
│   │   └── train_housing.py  # Housing model training
│   └── utils/                # Utility functions
│       └── logger.py         # Logging configuration
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest configuration
│   ├── test_api.py           # API endpoint tests
│   ├── test_logging.py       # Logging tests
│   ├── test_schema.py        # Schema validation tests
│   └── test_train.py         # Training tests
├── data/                     # Data files
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── artifacts/                # Experiment artifacts
│   ├── experiments/          # Experiment outputs (plots, etc.)
│   ├── models/               # Trained models
│   └── reports/              # Analysis reports
├── logs/                     # Application logs
│   ├── api/                  # API logs
│   └── predictions/          # Prediction logs
├── mlruns/                   # MLflow tracking directory (single location)
├── docs/                     # Documentation
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
├── pytest.ini               # Pytest configuration
├── run_tests.py             # Test runner script
├── cleanup.py               # Directory cleanup script
├── setup_mlflow.py          # MLflow setup script
├── mlflow_config.py         # MLflow configuration
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker services
├── .gitignore               # Git ignore rules
├── README.md                # Project overview
├── TESTING_README.md        # Testing documentation
├── LOGGING_README.md        # Logging documentation
└── PROJECT_STRUCTURE.md     # This file
```

### Configuration Files
- **`requirements.txt`**: Production Python dependencies
- **`requirements-dev.txt`**: Development and testing dependencies
- **`pytest.ini`**: Pytest configuration and markers
- **`setup.py`**: Package installation configuration
- **`mlflow_config.py`**: Centralized MLflow configuration
- **`dvc.yaml`**: DVC pipeline definition
- **`.dvc/config`**: DVC configuration settings

## Directory Purposes

### Source Code (`src/`)
- **`api/`**: FastAPI application with endpoints and schemas
- **`models/`**: Machine learning model training and management
- **`utils/`**: Shared utility functions and configurations

### Testing (`tests/`)
- **`conftest.py`**: Shared pytest fixtures and configuration
- **`test_*.py`**: Comprehensive test suites for each component
- Organized by functionality (API, logging, schema, training)

### Data Management (`data/`)
- **`raw/`**: Original, unprocessed data files
- **`processed/`**: Cleaned and preprocessed data files
- Version controlled with DVC

### DVC Configuration (`.dvc/`)
- **`config`**: DVC settings and remote configurations
- **`.gitignore`**: DVC-specific ignore patterns
- **`dvc.yaml`**: Pipeline definitions for data processing and training
- **`dvc-storage/`**: Local storage for DVC artifacts

### Artifacts (`artifacts/`)
- **`experiments/`**: Experiment outputs, plots, and visualizations
- **`models/`**: Trained model files and checkpoints
- **`reports/`**: Analysis reports and summaries

### Logging (`logs/`)
- **`api/`**: API request/response logs
- **`predictions/`**: Prediction request/response logs and databases
- Structured logging with JSON format

### MLflow (`mlruns/`)
- **Single tracking directory**: All MLflow experiments and runs
- **Consistent location**: Always uses `file:./mlruns` as tracking URI
- **No duplicates**: Removes `.mlruns/` and `mlflow_logs/` directories
- **Centralized config**: Uses `mlflow_config.py` for consistent setup

## File Naming Conventions

### Python Files
- Use snake_case for file names: `train_iris.py`, `test_api.py`
- Use descriptive names that indicate purpose
- Group related functionality in modules

### Test Files
- Prefix with `test_`: `test_api.py`, `test_logging.py`
- Match the structure of the source code being tested
- Use descriptive test class and method names

### Configuration Files
- Use lowercase with appropriate extensions: `pytest.ini`, `requirements.txt`
- Use descriptive names that indicate purpose

## Best Practices

### Directory Organization
1. **Separation of Concerns**: Keep different types of files in appropriate directories
2. **Logical Grouping**: Group related files together
3. **Scalability**: Structure should accommodate future growth
4. **Consistency**: Follow established patterns throughout the project

### File Management
1. **Version Control**: Use `.gitignore` to exclude generated files
2. **Cleanup**: Regularly remove temporary and cache files
3. **Documentation**: Keep documentation up to date with structure changes
4. **Naming**: Use clear, descriptive names for all files and directories

### Testing Structure
1. **Comprehensive Coverage**: Test all major components
2. **Organized Tests**: Group tests by functionality
3. **Fixtures**: Use shared fixtures for common test data
4. **Markers**: Use pytest markers for test categorization

### Logging Structure
1. **Organized Logs**: Separate logs by type and purpose
2. **Structured Format**: Use JSON for machine-readable logs
3. **Rotation**: Implement log rotation for production
4. **Monitoring**: Integrate with monitoring systems

### MLflow Management
1. **Single Directory**: Use only `mlruns/` for all MLflow tracking
2. **Consistent Configuration**: Use centralized `mlflow_config.py`
3. **No Duplicates**: Remove `.mlruns/` and `mlflow_logs/` directories
4. **Environment Variables**: Set `MLFLOW_TRACKING_URI=file:./mlruns`

### DVC Management
1. **Data Versioning**: Track all data files with DVC
2. **Pipeline Automation**: Use DVC pipelines for reproducible workflows
3. **Remote Storage**: Use local storage for development, cloud for production
4. **Artifact Management**: Track models and experiment outputs

## Maintenance

### Regular Cleanup Tasks
1. Remove `__pycache__/` directories
2. Clean up temporary files
3. Archive old experiment artifacts
4. Rotate log files
5. Update documentation
6. Ensure single MLflow directory (`mlruns/`)
7. Clean DVC cache: `dvc gc`

### Adding New Components
1. Follow established naming conventions
2. Update this documentation
3. Add appropriate tests
4. Update `.gitignore` if needed
5. Consider impact on existing structure
6. Add DVC tracking for new data files

## Tools and Scripts

### Test Runner (`run_tests.py`)
- Comprehensive test execution
- Multiple test types and configurations
- Coverage reporting
- Parallel execution support

### Cleanup Script (`cleanup.py`)
- Remove Python cache files
- Clean test artifacts
- Organize artifacts and logs
- Remove duplicate MLflow directories
- Show directory structure

### MLflow Setup (`setup_mlflow.py`)
- Ensure single MLflow directory
- Remove duplicate directories
- Set environment variables
- Configure tracking URI

### DVC Setup (`scripts/setup_dvc.py`)
- Initialize DVC configuration
- Setup remote storage
- Add data files to tracking
- Configure pipelines

### GitHub Actions
- **Single Workflow**: `tests.yml` handles all testing scenarios
- **Smart Execution**: Quick tests for PRs, comprehensive tests for main/develop
- **Multiple Python versions**: Matrix testing for compatibility
- **Coverage reporting**: Integrated with Codecov
- **Code quality**: SonarQube analysis for main branch
- **Performance testing**: Dedicated performance test suite

### Docker Support
- Containerized development environment
- Production deployment ready
- Service orchestration with docker-compose

This structure promotes maintainability, scalability, and collaboration while keeping the project organized and easy to navigate.