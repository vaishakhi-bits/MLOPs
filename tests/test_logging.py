import pytest
import json
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime

from src.utils.logger import setup_logger, setup_prediction_logger, PredictionLogger

class TestLoggerSetup:
    """Test logger setup functionality"""
    
    def test_setup_logger_creates_directory(self, tmp_path):
        """Test that setup_logger creates logs directory"""
        with patch('src.utils.logger.Path') as mock_path:
            mock_logs_dir = Mock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.mkdir = Mock()
            
            logger = setup_logger("test_logger")
            
            mock_logs_dir.mkdir.assert_called_once_with(exist_ok=True)
    
    def test_setup_logger_creates_file_handler(self, tmp_path):
        """Test that setup_logger creates file handler"""
        with patch('src.utils.logger.Path') as mock_path:
            mock_logs_dir = Mock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.mkdir = Mock()
            
            # Mock the log file
            mock_log_file = Mock()
            mock_logs_dir.__truediv__.return_value = mock_log_file
            
            logger = setup_logger("test_logger")
            
            # Check that file handler was created
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
    
    def test_setup_logger_creates_console_handler(self, tmp_path):
        """Test that setup_logger creates console handler"""
        with patch('src.utils.logger.Path') as mock_path:
            mock_logs_dir = Mock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.mkdir = Mock()
            
            # Mock the log file
            mock_log_file = Mock()
            mock_logs_dir.__truediv__.return_value = mock_log_file
            
            logger = setup_logger("test_logger")
            
            # Check that console handler was created
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
            assert len(console_handlers) == 1
    
    def test_setup_logger_prevents_duplicate_handlers(self, tmp_path):
        """Test that setup_logger doesn't add duplicate handlers"""
        with patch('src.utils.logger.Path') as mock_path:
            mock_logs_dir = Mock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.mkdir = Mock()
            
            # Mock the log file
            mock_log_file = Mock()
            mock_logs_dir.__truediv__.return_value = mock_log_file
            
            # Call setup_logger twice
            logger1 = setup_logger("test_logger")
            logger2 = setup_logger("test_logger")
            
            # Should be the same logger instance
            assert logger1 is logger2
            # Should have only one set of handlers
            assert len(logger1.handlers) == 2  # File + Console

class TestPredictionLogger:
    """Test prediction logger functionality"""
    
    def test_setup_prediction_logger(self, tmp_path):
        """Test setup_prediction_logger creates prediction logger"""
        with patch('src.utils.logger.Path') as mock_path:
            mock_logs_dir = Mock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.mkdir = Mock()
            
            # Mock the prediction log file
            mock_pred_log_file = Mock()
            mock_logs_dir.__truediv__.return_value = mock_pred_log_file
            
            logger = setup_prediction_logger()
            
            assert logger.name == "prediction"
            assert logger.level == logging.INFO
    
    def test_prediction_logger_initialization(self, tmp_path):
        """Test PredictionLogger class initialization"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            mock_setup.assert_called_once()
            assert pred_logger.logger == mock_logger
    
    def test_log_request(self, tmp_path):
        """Test logging prediction request"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Test data
            model_type = "iris"
            request_id = "test-request-123"
            features = {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5}
            client_ip = "127.0.0.1"
            user_agent = "test-agent"
            
            pred_logger.log_request(model_type, request_id, features, client_ip, user_agent)
            
            # Verify logger.info was called
            mock_logger.info.assert_called_once()
            
            # Verify the log message contains expected data
            call_args = mock_logger.info.call_args[0][0]
            assert call_args.startswith("REQUEST: ")
            
            # Parse the JSON part
            json_str = call_args.replace("REQUEST: ", "")
            log_entry = json.loads(json_str)
            
            assert log_entry["event_type"] == "prediction_request"
            assert log_entry["request_id"] == request_id
            assert log_entry["model_type"] == model_type
            assert log_entry["features"] == features
            assert log_entry["client_ip"] == client_ip
            assert log_entry["user_agent"] == user_agent
            assert "timestamp" in log_entry
    
    def test_log_response(self, tmp_path):
        """Test logging prediction response"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Test data
            request_id = "test-request-123"
            model_type = "iris"
            prediction = "Iris-setosa"
            processing_time = 0.1234
            success = True
            error = None
            
            pred_logger.log_response(request_id, model_type, prediction, processing_time, success, error)
            
            # Verify logger.info was called
            mock_logger.info.assert_called_once()
            
            # Verify the log message contains expected data
            call_args = mock_logger.info.call_args[0][0]
            assert call_args.startswith("RESPONSE: ")
            
            # Parse the JSON part
            json_str = call_args.replace("RESPONSE: ", "")
            log_entry = json.loads(json_str)
            
            assert log_entry["event_type"] == "prediction_response"
            assert log_entry["request_id"] == request_id
            assert log_entry["model_type"] == model_type
            assert log_entry["prediction"] == prediction
            assert log_entry["processing_time_seconds"] == 0.1234
            assert log_entry["success"] == success
            assert log_entry["error"] == error
            assert "timestamp" in log_entry
    
    def test_log_response_with_error(self, tmp_path):
        """Test logging prediction response with error"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Test data with error
            request_id = "test-request-123"
            model_type = "iris"
            prediction = None
            processing_time = 0.5678
            success = False
            error = "Model not loaded"
            
            pred_logger.log_response(request_id, model_type, prediction, processing_time, success, error)
            
            # Verify logger.info was called
            mock_logger.info.assert_called_once()
            
            # Parse the JSON part
            call_args = mock_logger.info.call_args[0][0]
            json_str = call_args.replace("RESPONSE: ", "")
            log_entry = json.loads(json_str)
            
            assert log_entry["success"] == False
            assert log_entry["error"] == "Model not loaded"
            assert log_entry["prediction"] is None
    
    def test_log_model_status(self, tmp_path):
        """Test logging model status"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Test data
            model_type = "iris"
            status = "loaded"
            details = "Model loaded successfully"
            
            pred_logger.log_model_status(model_type, status, details)
            
            # Verify logger.info was called
            mock_logger.info.assert_called_once()
            
            # Verify the log message contains expected data
            call_args = mock_logger.info.call_args[0][0]
            assert call_args.startswith("MODEL_STATUS: ")
            
            # Parse the JSON part
            json_str = call_args.replace("MODEL_STATUS: ", "")
            log_entry = json.loads(json_str)
            
            assert log_entry["event_type"] == "model_status"
            assert log_entry["model_type"] == model_type
            assert log_entry["status"] == status
            assert log_entry["details"] == details
            assert "timestamp" in log_entry
    
    def test_log_model_status_no_details(self, tmp_path):
        """Test logging model status without details"""
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Test data without details
            model_type = "housing"
            status = "error"
            
            pred_logger.log_model_status(model_type, status)
            
            # Parse the JSON part
            call_args = mock_logger.info.call_args[0][0]
            json_str = call_args.replace("MODEL_STATUS: ", "")
            log_entry = json.loads(json_str)
            
            assert log_entry["details"] is None

class TestLoggingIntegration:
    """Test logging integration with API"""
    
    def test_logging_in_api_request(self, tmp_path):
        """Test that logging is properly integrated in API requests"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Mock the prediction logger
        with patch('src.api.main.prediction_logger') as mock_pred_logger:
            # Mock the iris model
            with patch('src.api.main.iris_model') as mock_model:
                mock_model.predict.return_value = [0]  # Iris-setosa
                
                # Make a request
                response = client.post("/predict_iris", json={
                    "SepalLengthCm": 5.1,
                    "SepalWidthCm": 3.5,
                    "PetalLengthCm": 1.4,
                    "PetalWidthCm": 0.2
                })
                
                # Verify that logging methods were called
                assert mock_pred_logger.log_request.called
                assert mock_pred_logger.log_response.called
                
                # Check the request logging call
                request_call = mock_pred_logger.log_request.call_args
                assert request_call[0][0] == "iris"  # model_type
                assert "request_id" in request_call[0][2]  # features
                
                # Check the response logging call
                response_call = mock_pred_logger.log_response.call_args
                assert response_call[0][1] == "iris"  # model_type
                assert response_call[0][2] == "Iris-setosa"  # prediction
                assert response_call[0][3] > 0  # processing_time > 0
                assert response_call[0][4] == True  # success
    
    def test_logging_in_api_error(self, tmp_path):
        """Test that logging is properly integrated in API error handling"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Mock the prediction logger
        with patch('src.api.main.prediction_logger') as mock_pred_logger:
            # Mock the iris model to raise an error
            with patch('src.api.main.iris_model') as mock_model:
                mock_model.predict.side_effect = Exception("Test error")
                
                # Make a request
                response = client.post("/predict_iris", json={
                    "SepalLengthCm": 5.1,
                    "SepalWidthCm": 3.5,
                    "PetalLengthCm": 1.4,
                    "PetalWidthCm": 0.2
                })
                
                # Verify that logging methods were called
                assert mock_pred_logger.log_request.called
                assert mock_pred_logger.log_response.called
                
                # Check the response logging call for error
                response_call = mock_pred_logger.log_response.call_args
                assert response_call[0][4] == False  # success = False
                assert "Test error" in response_call[0][5]  # error message

class TestLogFileOperations:
    """Test actual log file operations"""
    
    def test_log_file_creation(self, tmp_path):
        """Test that log files are actually created"""
        # Create a temporary logs directory
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        
        with patch('src.utils.logger.Path') as mock_path:
            mock_path.return_value = logs_dir
            
            # Setup logger
            logger = setup_logger("test_logger")
            
            # Write a test message
            logger.info("Test log message")
            
            # Check that log file was created
            log_file = logs_dir / "test_logger.log"
            assert log_file.exists()
            
            # Check that message was written
            content = log_file.read_text()
            assert "Test log message" in content
    
    def test_prediction_log_file_creation(self, tmp_path):
        """Test that prediction log files are actually created"""
        # Create a temporary logs directory
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        
        with patch('src.utils.logger.Path') as mock_path:
            mock_path.return_value = logs_dir
            
            # Setup prediction logger
            logger = setup_prediction_logger()
            
            # Write a test message
            logger.info("Test prediction log message")
            
            # Check that log file was created
            log_file = logs_dir / "predictions.log"
            assert log_file.exists()
            
            # Check that message was written
            content = log_file.read_text()
            assert "Test prediction log message" in content

class TestLoggingPerformance:
    """Test logging performance characteristics"""
    
    def test_logging_performance(self, tmp_path):
        """Test that logging doesn't significantly impact performance"""
        import time
        
        with patch('src.utils.logger.setup_prediction_logger') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            pred_logger = PredictionLogger()
            
            # Time the logging operation
            start_time = time.time()
            
            for i in range(100):
                pred_logger.log_request("iris", f"req-{i}", {"test": i}, "127.0.0.1", "test")
                pred_logger.log_response(f"req-{i}", "iris", "Iris-setosa", 0.001, True, None)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 200 logging operations in reasonable time (< 1 second)
            assert total_time < 1.0
            assert mock_logger.info.call_count == 200

if __name__ == "__main__":
    pytest.main([__file__]) 