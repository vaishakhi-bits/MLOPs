# DVC (Data Version Control) Setup Guide

This document explains how to use DVC for data version control in the YugenAI project.

## What is DVC?

DVC (Data Version Control) is a tool that helps you version control large files, data sets, machine learning models, and experiments. It's designed to work alongside Git.

## Project Structure

```
yugenai/
├── .dvc/                    # DVC configuration
│   ├── config              # DVC settings
│   └── .gitignore          # DVC ignore rules
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC lock file (version control)
├── dvc-storage/            # Local DVC storage (gitignored)
├── data/                   # Data files
│   ├── raw/                # Raw data (tracked by DVC)
│   └── processed/          # Processed data (tracked by DVC)
├── src/models/saved/       # Saved models (tracked by DVC)
└── scripts/setup_dvc.py    # DVC setup script
```

## Quick Start

### 1. Install DVC

```bash
# Install DVC with Google Cloud Storage support
pip install dvc[gcs]

# Or install from requirements
pip install -r requirements.txt
```

### 2. Setup DVC

```bash
# Run the setup script
python scripts/setup_dvc.py

# Or setup manually
dvc init
dvc remote add default ./dvc-storage
dvc remote add gcs gs://rapid_care/mlops-dvc
```

### 3. Add Data Files

```bash
# Add raw data
dvc add data/raw/housing.csv
dvc add data/raw/iris.csv

# Add processed data
dvc add data/processed/housing_preprocessed.csv
dvc add data/processed/iris_preprocessed.csv

# Add models
dvc add src/models/saved/housing_model.pkl
dvc add src/models/saved/iris_model.pkl
```

### 4. Commit DVC Files

```bash
# Add .dvc files to git
git add *.dvc
git commit -m "Add data files to DVC tracking"
```

## DVC Pipelines

The project uses DVC pipelines to automate data processing and model training workflows.

### Available Pipelines

#### 1. Data Preprocessing

```bash
# Preprocess housing data
dvc repro preprocess_housing

# Preprocess iris data
dvc repro preprocess_iris
```

#### 2. Model Training

```bash
# Train housing model
dvc repro train_housing

# Train iris model
dvc repro train_iris

# Train all models
dvc repro train_all
```

#### 3. Full Pipeline

```bash
# Run complete pipeline (preprocessing + training)
dvc repro
```

### Pipeline Stages

The `dvc.yaml` file defines the following stages:

1. **preprocess_housing**: Preprocess housing dataset
2. **preprocess_iris**: Preprocess iris dataset
3. **train_housing**: Train housing prediction models
4. **train_iris**: Train iris classification models
5. **train_all**: Train all models

## Remote Storage

### Local Storage (Default)

- **URL**: `./dvc-storage`
- **Purpose**: Local development and testing
- **Usage**: Default remote for all operations

### Google Cloud Storage

- **URL**: `gs://rapid_care/mlops-dvc`
- **Purpose**: Production data storage
- **Requirements**: GCS credentials

### Setup GCS Remote

1. **Install Google Cloud SDK**
   ```bash
   # Install gcloud CLI
   # Follow: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project rapid_care
   ```

3. **Set Credentials**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   ```

4. **Add Remote**
   ```bash
   dvc remote add gcs gs://rapid_care/mlops-dvc
   ```

## Common Commands

### Data Management

```bash
# Add file to DVC tracking
dvc add <file_path>

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Check status
dvc status
```

### Pipeline Management

```bash
# Run pipeline
dvc repro

# Run specific stage
dvc repro <stage_name>

# Show pipeline graph
dvc dag

# Show pipeline status
dvc status --show-json
```

### Remote Management

```bash
# List remotes
dvc remote list

# Add remote
dvc remote add <name> <url>

# Set default remote
dvc remote default <name>

# Remove remote
dvc remote remove <name>
```

## Best Practices

### 1. File Organization

- **Raw data**: Store in `data/raw/`
- **Processed data**: Store in `data/processed/`
- **Models**: Store in `src/models/saved/`
- **Artifacts**: Store in `artifacts/`

### 2. Version Control

- **Always commit `.dvc` files** to git
- **Never commit large files** directly to git
- **Use descriptive commit messages** for data changes

### 3. Pipeline Design

- **Separate preprocessing** from training
- **Use dependencies** to ensure correct execution order
- **Include metrics** for tracking performance

### 4. Remote Storage

- **Use local storage** for development
- **Use cloud storage** for production
- **Backup important data** to multiple remotes

## Troubleshooting

### Common Issues

#### 1. DVC Not Found

```bash
# Install DVC
pip install dvc[gcs]

# Check installation
dvc --version
```

#### 2. Remote Connection Issues

```bash
# Check remote configuration
dvc remote list

# Test connection
dvc push --remote <remote_name>
```

#### 3. Pipeline Failures

```bash
# Check pipeline status
dvc status

# Run with verbose output
dvc repro --verbose

# Clean and rerun
dvc repro --force
```

#### 4. Large File Issues

```bash
# Check file size
ls -lh <file_path>

# Use .dvcignore for large directories
echo "large_directory/" >> .dvcignore
```

### Debug Commands

```bash
# Show DVC configuration
dvc config --list

# Show cache information
dvc cache dir

# Show remote information
dvc remote list

# Show pipeline information
dvc dag --dot
```

## Integration with CI/CD

### GitHub Actions

The project includes DVC integration in GitHub Actions:

```yaml
# Example workflow step
- name: Setup DVC
  run: |
    pip install dvc[gcs]
    dvc pull

- name: Run Pipeline
  run: |
    dvc repro

- name: Push Results
  run: |
    dvc push
```

### Environment Variables

Set these environment variables for CI/CD:

```bash
# GCS credentials
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# DVC configuration
DVC_REMOTE=gcs
```

## Monitoring and Metrics

### Pipeline Metrics

DVC automatically tracks metrics from pipeline outputs:

```bash
# Show metrics
dvc metrics show

# Compare metrics
dvc metrics diff

# Plot metrics
dvc plots show
```

### Performance Tracking

Track pipeline performance:

```bash
# Show execution times
dvc repro --dry-run

# Profile pipeline
dvc repro --profile
```

## Security Considerations

### Credential Management

- **Never commit credentials** to git
- **Use environment variables** for sensitive data
- **Rotate credentials** regularly

### Access Control

- **Limit remote access** to authorized users
- **Use service accounts** for automated access
- **Monitor access logs** regularly

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/tutorial)
- [DVC Best Practices](https://dvc.org/doc/user-guide/best-practices)
- [Google Cloud Storage](https://cloud.google.com/storage)

## Support

For DVC-related issues:

1. Check the [DVC documentation](https://dvc.org/doc)
2. Review the [troubleshooting guide](https://dvc.org/doc/user-guide/troubleshooting)
3. Ask questions on [DVC Discord](https://discord.gg/dvc)
4. Report issues on [DVC GitHub](https://github.com/iterative/dvc) 