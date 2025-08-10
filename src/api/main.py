from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.api.schema import (
    HousingFeatures,
    HousingPredictionResponse,
    IrisFeatures,
    IrisPredictionResponse
)
import numpy as np
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv
import os
import time
import uuid
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger, prediction_logger

# Import centralized MLflow configuration
try:
    from mlflow_config import setup_mlflow
    MLFLOW_CONFIG_AVAILABLE = True
except ImportError:
    MLFLOW_CONFIG_AVAILABLE = False
 
# Load environment variables from .env file
load_dotenv()
 
# Set up logging
logger = setup_logger("api")
 
# Read env variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# IRIS_RUN_ID = os.getenv("IRIS_RUN_ID")
HOUSING_RUN_ID = os.getenv("HOUSING_RUN_ID")  # This may be None for now
 
# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Latency (seconds)',
    ['method', 'endpoint']
)

# Create a separate metrics app for Prometheus
metrics_app = make_asgi_app()

app = FastAPI()

# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info("=== YugenAI API Starting ===")
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"HOUSING_RUN_ID: {HOUSING_RUN_ID}")
    
    # Set up MLflow with centralized configuration
    if MLFLOW_CONFIG_AVAILABLE:
        mlflow_success = setup_mlflow(MLFLOW_TRACKING_URI)
        if mlflow_success:
            logger.info("MLflow configured successfully")
        else:
            logger.warning("MLflow configuration failed, continuing without MLflow")
    else:
        # Fallback to direct MLflow setup
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        else:
            logger.info("No MLFLOW_TRACKING_URI set, MLflow not configured")
    
    prediction_logger.log_model_status("application", "startup", "YugenAI API application starting")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== YugenAI API Shutting Down ===")
    prediction_logger.log_model_status("application", "shutdown", "YugenAI API application shutting down")

# Add middleware to track request metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    # Skip metrics endpoint from being tracked
    if endpoint == "/metrics":
        return await call_next(request)
    
    # Record start time for latency calculation
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate request duration
    request_duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(request_duration)
    
    return response
 
# Set MLflow tracking URI from env
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) # This line is now handled by mlflow_config or direct setup
 
# Load Iris model from local file
iris_model = None
logger.info("=== Loading Iris model ===")
prediction_logger.log_model_status("iris", "loading", "Starting Iris model loading process")

try:
    # Define the model path relative to the project root
    project_root = Path(__file__).parent.parent.parent  # Goes up to yugenai directory
    iris_model_path = project_root / "src" / "models" / "saved" / "iris_model.pkl"
    
    logger.info(f"Attempting to load Iris model from: {iris_model_path}")
    
    # Check if file exists
    if not os.path.exists(iris_model_path):
        error_msg = f"ERROR: Iris model file not found at {iris_model_path}"
        logger.error(error_msg)
        prediction_logger.log_model_status("iris", "error", error_msg)
    else:
        # Load the model
        iris_model = joblib.load(iris_model_path)
        logger.info("Successfully loaded Iris model from specified path")
        logger.info(f"Model type: {type(iris_model)}")
        prediction_logger.log_model_status("iris", "loaded", f"Model type: {type(iris_model)}")
        
        # Verify the model has the predict method
        if not hasattr(iris_model, 'predict'):
            error_msg = "ERROR: Loaded model does not have predict() method"
            logger.error(error_msg)
            prediction_logger.log_model_status("iris", "error", error_msg)
            iris_model = None
        else:
            prediction_logger.log_model_status("iris", "ready", "Model loaded and ready for predictions")
except Exception as e:
    error_msg = f"Error loading Iris model: {str(e)}"
    logger.error(error_msg)
    prediction_logger.log_model_status("iris", "error", error_msg)
 
# Load Housing model when run_id is available
logger.info("=== Starting model loading process ===")
logger.info(f"HOUSING_RUN_ID from .env: {HOUSING_RUN_ID}")
logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
prediction_logger.log_model_status("housing", "loading", "Starting Housing model loading process")

housing_model = None

# Try to load the model from the specified path
try:
    import os
    import joblib
    from pathlib import Path
    
    # Define the model path relative to the project root
    project_root = Path(__file__).parent.parent.parent  # Goes up to yugenai directory
    model_path = project_root / "src" / "models" / "saved" / "housing_model.pkl"
    
    logger.info(f"Attempting to load model from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        error_msg = f"ERROR: Model file not found at {model_path}"
        logger.error(error_msg)
        prediction_logger.log_model_status("housing", "error", error_msg)
        raise FileNotFoundError(f"Housing model not found at {model_path}")
    
    logger.info("Model file exists. Attempting to load...")
    
        # Load the model and scaler
    model_data = joblib.load(model_path)
    
    # Extract model and scaler from the loaded data
    if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
        housing_model = model_data['model']
        feature_scaler = model_data['scaler']
        logger.info("Successfully loaded housing model and scaler from specified path")
        prediction_logger.log_model_status("housing", "loaded", "Model and scaler loaded successfully")
    else:
        # For backward compatibility if the model was saved without the scaler
        housing_model = model_data
        feature_scaler = None
        logger.warning("Warning: Loaded model does not include a scaler. Using unscaled features.")
        prediction_logger.log_model_status("housing", "loaded", "Model loaded without scaler")
    
    logger.info(f"Model type: {type(housing_model)}")
    
    # Verify the model has the predict method
    if not hasattr(housing_model, 'predict'):
        error_msg = "Loaded model does not have predict() method"
        logger.error(error_msg)
        prediction_logger.log_model_status("housing", "error", error_msg)
        raise AttributeError("Loaded model does not have predict() method")
    
    prediction_logger.log_model_status("housing", "ready", "Model loaded and ready for predictions")

except Exception as e:
    error_msg = f"Error loading housing model: {str(e)}"
    logger.error(error_msg)
    prediction_logger.log_model_status("housing", "error", error_msg)
    housing_model = None
 
# Define allowed categories for one-hot encoding (used in housing prediction)
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
 
@app.get("/")
async def read_root():
    logger.info("API root endpoint accessed")
    return {"message": "Welcome to YugenAI API. Use /docs for API documentation."}

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Log viewer endpoint (for debugging and monitoring)
@app.get("/logs")
async def view_logs(log_type: str = "api", lines: int = 50):
    """View recent log entries. log_type can be 'api', 'prediction', or 'all'"""
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            return {"error": "Logs directory not found"}
        
        log_content = []
        
        if log_type in ["api", "all"]:
            api_log_file = log_dir / "api" / "api.log"
            if api_log_file.exists():
                with open(api_log_file, 'r') as f:
                    api_lines = f.readlines()[-lines:]
                    log_content.extend([f"[API] {line.strip()}" for line in api_lines])
        
        if log_type in ["prediction", "all"]:
            prediction_log_file = log_dir / "predictions" / "predictions.log"
            if prediction_log_file.exists():
                with open(prediction_log_file, 'r') as f:
                    pred_lines = f.readlines()[-lines:]
                    log_content.extend([f"[PREDICTION] {line.strip()}" for line in pred_lines])
        
        # Sort by timestamp (assuming ISO format at start of line)
        log_content.sort(key=lambda x: x.split(' - ')[0] if ' - ' in x else x)
        
        return {
            "log_type": log_type,
            "lines_requested": lines,
            "total_lines": len(log_content),
            "logs": log_content[-lines:]
        }
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return {"error": f"Error reading logs: {str(e)}"}

# Housing Prediction endpoint (placeholder if model not loaded)
@app.post("/predict_housing", response_model=HousingPredictionResponse)
def predict_housing(features: HousingFeatures, request: Request):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Get client information
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    # Log incoming request
    input_features = features.dict()
    prediction_logger.log_request("housing", request_id, input_features, client_ip, user_agent)
    logger.info(f"Housing prediction request {request_id} received from {client_ip}")
    
    # Check if housing_model is loaded
    if housing_model is None:
        error_msg = "Housing model is not loaded"
        logger.warning(f"Request {request_id}: {error_msg}")
        processing_time = time.time() - start_time
        prediction_logger.log_response(request_id, "housing", None, processing_time, False, error_msg)
        return HousingPredictionResponse(predicted_price=0.0)
   
    try:
        # Convert input features to a dictionary and then to DataFrame
        input_data = features.dict()
        
        logger.info(f"Request {request_id}: Input features received: {input_data}")
        
        # Create a DataFrame with all expected columns
        # The model expects these specific feature names for ocean_proximity
        feature_columns = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
        ]
        
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Map input features to the expected columns
        for col in input_data:
            if col == 'ocean_proximity':
                # Handle one-hot encoding for ocean_proximity
                ocean_value = input_data[col].upper()
                
                # Map input to expected feature name
                ocean_mapping = {
                    '<1H OCEAN': None,  # This is the reference category (all zeros)
                    'INLAND': 'ocean_proximity_INLAND',
                    'ISLAND': 'ocean_proximity_ISLAND',
                    'NEAR BAY': 'ocean_proximity_NEAR BAY',
                    'NEAR OCEAN': 'ocean_proximity_NEAR OCEAN'
                }
                
                ocean_col = ocean_mapping.get(ocean_value)
                if ocean_col and ocean_col in input_df.columns:
                    input_df[ocean_col] = 1
                elif ocean_value not in ocean_mapping:
                    logger.warning(f"Request {request_id}: Could not map ocean_proximity value '{input_data[col]}' to a valid feature column")
            elif col in input_df.columns:
                input_df[col] = input_data[col]
        
        # Ensure all expected features are present
        missing_cols = set(feature_columns) - set(input_df.columns)
        if missing_cols:
            logger.warning(f"Request {request_id}: Missing expected columns: {missing_cols}")
            for col in missing_cols:
                input_df[col] = 0
        
        logger.info(f"Request {request_id}: Processed input features: {input_df.iloc[0].to_dict()}")
        
        # Scale numerical features if scaler is available
        if 'feature_scaler' in globals() and feature_scaler is not None:
            # Get numerical columns (exclude the one-hot encoded columns)
            numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income']
            
            # Scale the numerical features
            input_df[numerical_cols] = feature_scaler.transform(input_df[numerical_cols])
            logger.info(f"Request {request_id}: Applied feature scaling to numerical columns")
        
        # Make prediction
        prediction = housing_model.predict(input_df)[0]
        logger.info(f"Request {request_id}: Raw prediction: {prediction}")
        
        # If the model was trained on log-transformed target, apply inverse transform
        if hasattr(housing_model, 'target_transform_') and housing_model.target_transform_ == 'log':
            prediction = np.expm1(prediction)  # Inverse of log1p
            logger.info(f"Request {request_id}: Transformed prediction (expm1): {prediction}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log successful response
        prediction_logger.log_response(request_id, "housing", float(prediction), processing_time, True)
        logger.info(f"Request {request_id}: Housing prediction completed successfully in {processing_time:.4f}s")
            
        return HousingPredictionResponse(predicted_price=float(prediction))
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(f"Request {request_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Log error response
        prediction_logger.log_response(request_id, "housing", None, processing_time, False, error_msg)
        
        return HousingPredictionResponse(predicted_price=0.0)
 
# Iris Prediction endpoint
@app.post("/predict_iris", response_model=IrisPredictionResponse)
def predict_iris(features: IrisFeatures, request: Request):
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Get client information
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    # Log incoming request
    input_features = features.dict()
    prediction_logger.log_request("iris", request_id, input_features, client_ip, user_agent)
    logger.info(f"Iris prediction request {request_id} received from {client_ip}")
    
    # Check if iris_model is loaded
    if iris_model is None:
        error_msg = "Iris model is not loaded"
        logger.error(f"Request {request_id}: {error_msg}")
        processing_time = time.time() - start_time
        prediction_logger.log_response(request_id, "iris", None, processing_time, False, error_msg)
        raise HTTPException(status_code=503, detail="Iris model is not loaded")
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            "SepalLengthCm": features.SepalLengthCm,
            "SepalWidthCm": features.SepalWidthCm,
            "PetalLengthCm": features.PetalLengthCm,
            "PetalWidthCm": features.PetalWidthCm
        }])
        
        logger.info(f"Request {request_id}: Input features: {input_features}")
     
        # Make prediction
        prediction = iris_model.predict(input_data)[0]
        logger.info(f"Request {request_id}: Raw prediction: {prediction}")
        
        # Map prediction to class name
        class_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        predicted_class = class_map.get(prediction, "Unknown")
        
        logger.info(f"Request {request_id}: Predicted class: {predicted_class}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log successful response
        prediction_logger.log_response(request_id, "iris", predicted_class, processing_time, True)
        logger.info(f"Request {request_id}: Iris prediction completed successfully in {processing_time:.4f}s")
     
        return IrisPredictionResponse(predicted_class=predicted_class)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(f"Request {request_id}: {error_msg}")
        
        # Log error response
        prediction_logger.log_response(request_id, "iris", None, processing_time, False, error_msg)
        
        raise HTTPException(status_code=500, detail=str(e))