# YugenAI - Machine Learning API

A comprehensive machine learning API built with FastAPI, featuring housing price prediction and iris classification models.

## Features

- **FastAPI API** with automatic documentation
- **Machine Learning Models**: Housing price prediction and Iris classification
- **MLflow Integration** for experiment tracking
- **DVC Integration** for data version control
- **Comprehensive Testing** with pytest
- **Docker Support** for containerized deployment
- **GitHub Actions** for CI/CD
- **Structured Logging** with JSON format
- **Prometheus Metrics** for monitoring

## Quick Start

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yugenai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Setup DVC (Data Version Control)**
   ```bash
   python scripts/setup_dvc.py
   ```

5. **Run the API**
   ```bash
   uvicorn src.api.main:app --reload
   ```

6. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/
   - Metrics: http://localhost:8000/metrics

## Data Management with DVC

This project uses DVC for data version control. See [DVC_README.md](DVC_README.md) for detailed instructions.

### Quick DVC Commands

```bash
# Setup DVC
python scripts/setup_dvc.py

# Add data files
dvc add data/raw/housing.csv
dvc add data/processed/housing_preprocessed.csv

# Run data pipeline
dvc repro

# Push data to remote
dvc push
```

## API Endpoints

### Housing Price Prediction
- **POST** `/predict_housing` - Predict housing prices
- **Input**: Housing features (longitude, latitude, etc.)
- **Output**: Predicted price

### Iris Classification
- **POST** `/predict_iris` - Classify iris flowers
- **Input**: Iris features (sepal/petal dimensions)
- **Output**: Predicted species

### Monitoring
- **GET** `/metrics` - Prometheus metrics
- **GET** `/logs` - View application logs

## Development

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --training
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
pylint src/
```

### Training Models

```bash
# Train specific model
python -m src.models.train --model housing
python -m src.models.train --model iris

# Train all models
python -m src.models.train --model all
```

## Docker

### Build and Run

```bash
# Build image
docker build -t yugenai .

# Run container
docker run -p 8000:8000 yugenai

# Or use docker-compose
docker-compose up
```

## Project Structure

```
yugenai/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── data/              # Data processing
│   ├── models/            # Model training
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── data/                  # Data files (DVC tracked)
├── artifacts/             # Experiment artifacts
├── logs/                  # Application logs
├── mlruns/                # MLflow tracking
├── scripts/               # Utility scripts
├── .github/workflows/     # GitHub Actions
└── docs/                  # Documentation
```

## Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [DVC Guide](DVC_README.md) - Data version control
- [Testing Guide](TESTING_README.md) - Testing documentation
- [Logging Guide](LOGGING_README.md) - Logging system
- [Code Organization](CODE_ORGANIZATION.md) - Code structure
- [Project Structure](PROJECT_STRUCTURE.md) - Directory organization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the BITS License - see the LICENSE file for details.
