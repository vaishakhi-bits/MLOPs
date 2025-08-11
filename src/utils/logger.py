import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(name, log_level=logging.INFO):
    """Set up a logger with both file and console handlers."""
    # Create logs directory structure if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create subdirectories for different log types
    api_log_dir = log_dir / "api"
    api_log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create file handler
    log_file = api_log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_prediction_logger():
    """Set up a specialized logger for prediction requests and responses."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create predictions subdirectory
    predictions_log_dir = log_dir / "predictions"
    predictions_log_dir.mkdir(exist_ok=True)

    # Create prediction logger
    logger = logging.getLogger("prediction")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create file handler for predictions
    prediction_log_file = predictions_log_dir / "predictions.log"
    file_handler = logging.FileHandler(prediction_log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter for structured logging
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


class PredictionLogger:
    """Helper class for logging prediction requests and responses."""

    def __init__(self):
        self.logger = setup_prediction_logger()

    def log_request(
        self,
        model_type: str,
        request_id: str,
        features: Dict[str, Any],
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log incoming prediction request."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "prediction_request",
            "request_id": request_id,
            "model_type": model_type,
            "features": features,
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
        self.logger.info(f"REQUEST: {json.dumps(log_entry)}")

    def log_response(
        self,
        request_id: str,
        model_type: str,
        prediction: Any,
        processing_time: float,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log prediction response."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "prediction_response",
            "request_id": request_id,
            "model_type": model_type,
            "prediction": prediction,
            "processing_time_seconds": round(processing_time, 4),
            "success": success,
            "error": error,
        }
        self.logger.info(f"RESPONSE: {json.dumps(log_entry)}")

    def log_model_status(
        self, model_type: str, status: str, details: Optional[str] = None
    ):
        """Log model loading/status information."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_status",
            "model_type": model_type,
            "status": status,
            "details": details,
        }
        self.logger.info(f"MODEL_STATUS: {json.dumps(log_entry)}")


# Global prediction logger instance
prediction_logger = PredictionLogger()
