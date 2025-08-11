"""
Model retraining system with automatic triggers based on data changes.
"""

import os
import time
import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import mlflow
from prometheus_client import Counter, Histogram, Gauge

from .train import train_iris_model, train_housing_model
from src.utils.logger import setup_logger

logger = setup_logger("model_retraining")

# Prometheus metrics for retraining
RETRAINING_TRIGGERED = Counter(
    'model_retraining_triggered_total',
    'Total number of model retraining triggers',
    ['model_name', 'trigger_type']
)

RETRAINING_DURATION = Histogram(
    'model_retraining_duration_seconds',
    'Model retraining duration in seconds',
    ['model_name', 'status']
)

RETRAINING_SUCCESS = Counter(
    'model_retraining_success_total',
    'Total number of successful model retrainings',
    ['model_name']
)

RETRAINING_FAILURE = Counter(
    'model_retraining_failure_total',
    'Total number of failed model retrainings',
    ['model_name', 'error_type']
)

MODEL_AGE = Gauge(
    'model_age_seconds',
    'Age of the current model in seconds',
    ['model_name']
)

@dataclass
class RetrainingConfig:
    """Configuration for model retraining"""
    model_name: str
    data_path: str
    check_interval: int = 300  # 5 minutes
    min_data_size: int = 100
    max_model_age_hours: int = 24
    performance_threshold: float = 0.05  # 5% performance degradation
    enable_auto_retraining: bool = True
    notification_callback: Optional[Callable] = None

@dataclass
class RetrainingTrigger:
    """Represents a retraining trigger event"""
    model_name: str
    trigger_type: str  # 'data_change', 'time_based', 'performance', 'manual'
    trigger_time: datetime
    data_hash: Optional[str] = None
    performance_metrics: Optional[Dict] = None
    description: str = ""

class DataChangeHandler(FileSystemEventHandler):
    """Handles file system events for data changes"""
    
    def __init__(self, retraining_manager):
        self.retraining_manager = retraining_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Check if this is a data file we're monitoring
        for config in self.retraining_manager.configs.values():
            if event.src_path == config.data_path:
                self._handle_data_change(config, event.src_path)
    
    def _handle_data_change(self, config, file_path):
        """Handle data file change"""
        current_time = time.time()
        
        # Avoid duplicate triggers for the same file
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 10:  # 10 second cooldown
                return
        
        self.last_modified[file_path] = current_time
        
        logger.info(f"Data change detected for {config.model_name}: {file_path}")
        
        # Trigger retraining
        self.retraining_manager.trigger_retraining(
            config.model_name,
            "data_change",
            f"Data file modified: {file_path}"
        )

class ModelRetrainingManager:
    """Manages automatic model retraining based on various triggers"""
    
    def __init__(self, configs: Dict[str, RetrainingConfig]):
        self.configs = configs
        self.observer = Observer()
        self.retraining_history = []
        self.is_running = False
        self.retraining_locks = {}  # Prevent concurrent retraining of same model
        
        # Initialize locks for each model
        for model_name in configs.keys():
            self.retraining_locks[model_name] = threading.Lock()
        
        # Load retraining history
        self._load_history()
    
    def start_monitoring(self):
        """Start monitoring for retraining triggers"""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        logger.info("Starting model retraining monitoring...")
        
        # Set up file system monitoring
        event_handler = DataChangeHandler(self)
        for config in self.configs.values():
            if os.path.exists(os.path.dirname(config.data_path)):
                self.observer.schedule(event_handler, os.path.dirname(config.data_path), recursive=False)
                logger.info(f"Monitoring data directory: {os.path.dirname(config.data_path)}")
        
        self.observer.start()
        self.is_running = True
        
        # Start background monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Model retraining monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop monitoring for retraining triggers"""
        if not self.is_running:
            return
        
        logger.info("Stopping model retraining monitoring...")
        self.is_running = False
        self.observer.stop()
        self.observer.join()
        logger.info("Model retraining monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for time-based and performance triggers"""
        while self.is_running:
            try:
                for config in self.configs.values():
                    if not config.enable_auto_retraining:
                        continue
                    
                    # Check time-based triggers
                    self._check_time_based_triggers(config)
                    
                    # Check performance-based triggers
                    self._check_performance_triggers(config)
                
                # Sleep for the shortest check interval
                min_interval = min(config.check_interval for config in self.configs.values())
                time.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_time_based_triggers(self, config: RetrainingConfig):
        """Check if model needs retraining based on age"""
        try:
            # Get current model age
            model_age = self._get_model_age(config.model_name)
            if model_age is None:
                return
            
            MODEL_AGE.labels(model_name=config.model_name).set(model_age.total_seconds())
            
            # Check if model is too old
            max_age = timedelta(hours=config.max_model_age_hours)
            if model_age > max_age:
                logger.info(f"Model {config.model_name} is {model_age} old, triggering retraining")
                self.trigger_retraining(
                    config.model_name,
                    "time_based",
                    f"Model age: {model_age}, max age: {max_age}"
                )
        
        except Exception as e:
            logger.error(f"Error checking time-based triggers for {config.model_name}: {e}")
    
    def _check_performance_triggers(self, config: RetrainingConfig):
        """Check if model needs retraining based on performance degradation"""
        try:
            # Get recent performance metrics
            recent_performance = self._get_recent_performance(config.model_name)
            if recent_performance is None:
                return
            
            # Compare with baseline performance
            baseline_performance = self._get_baseline_performance(config.model_name)
            if baseline_performance is None:
                return
            
            # Calculate performance degradation
            degradation = baseline_performance - recent_performance
            if degradation > config.performance_threshold:
                logger.info(f"Performance degradation detected for {config.model_name}: {degradation:.3f}")
                self.trigger_retraining(
                    config.model_name,
                    "performance",
                    f"Performance degradation: {degradation:.3f}"
                )
        
        except Exception as e:
            logger.error(f"Error checking performance triggers for {config.model_name}: {e}")
    
    def trigger_retraining(self, model_name: str, trigger_type: str, description: str = ""):
        """Trigger model retraining"""
        if model_name not in self.configs:
            logger.error(f"Unknown model: {model_name}")
            return
        
        config = self.configs[model_name]
        
        # Check if retraining is enabled
        if not config.enable_auto_retraining:
            logger.info(f"Auto-retraining disabled for {model_name}")
            return
        
        # Check if data is sufficient
        if not self._check_data_sufficiency(config):
            logger.warning(f"Insufficient data for {model_name} retraining")
            return
        
        # Create trigger record
        trigger = RetrainingTrigger(
            model_name=model_name,
            trigger_type=trigger_type,
            trigger_time=datetime.now(),
            data_hash=self._get_data_hash(config.data_path),
            description=description
        )
        
        # Update metrics
        RETRAINING_TRIGGERED.labels(model_name=model_name, trigger_type=trigger_type).inc()
        
        # Start retraining in background thread
        retraining_thread = threading.Thread(
            target=self._perform_retraining,
            args=(config, trigger),
            daemon=True
        )
        retraining_thread.start()
        
        logger.info(f"Retraining triggered for {model_name}: {trigger_type}")
    
    def _perform_retraining(self, config: RetrainingConfig, trigger: RetrainingTrigger):
        """Perform the actual model retraining"""
        model_name = config.model_name
        
        # Acquire lock to prevent concurrent retraining
        if not self.retraining_locks[model_name].acquire(blocking=False):
            logger.warning(f"Retraining already in progress for {model_name}")
            return
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting retraining for {model_name}")
            
            # Perform retraining based on model type
            if model_name == "iris":
                results = train_iris_model(config.data_path, f"{model_name}_retraining")
            elif model_name == "housing":
                results = train_housing_model(config.data_path, f"{model_name}_retraining")
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            RETRAINING_DURATION.labels(model_name=model_name, status="success").observe(duration)
            RETRAINING_SUCCESS.labels(model_name=model_name).inc()
            
            # Record successful retraining
            self._record_retraining(trigger, "success", duration, results)
            
            # Send notification
            if config.notification_callback:
                config.notification_callback(
                    model_name=model_name,
                    status="success",
                    duration=duration,
                    results=results,
                    trigger=trigger
                )
            
            logger.info(f"Retraining completed successfully for {model_name} in {duration:.2f}s")
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Update metrics
            RETRAINING_DURATION.labels(model_name=model_name, status="failure").observe(duration)
            RETRAINING_FAILURE.labels(model_name=model_name, error_type=type(e).__name__).inc()
            
            # Record failed retraining
            self._record_retraining(trigger, "failure", duration, error=str(e))
            
            # Send notification
            if config.notification_callback:
                config.notification_callback(
                    model_name=model_name,
                    status="failure",
                    duration=duration,
                    error=str(e),
                    trigger=trigger
                )
            
            logger.error(f"Retraining failed for {model_name}: {e}")
        
        finally:
            # Release lock
            self.retraining_locks[model_name].release()
    
    def _check_data_sufficiency(self, config: RetrainingConfig) -> bool:
        """Check if there's sufficient data for retraining"""
        try:
            if not os.path.exists(config.data_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(config.data_path)
            if file_size == 0:
                return False
            
            # Check number of records
            df = pd.read_csv(config.data_path)
            if len(df) < config.min_data_size:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking data sufficiency: {e}")
            return False
    
    def _get_data_hash(self, data_path: str) -> Optional[str]:
        """Get hash of data file for change detection"""
        try:
            if not os.path.exists(data_path):
                return None
            
            with open(data_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        
        except Exception as e:
            logger.error(f"Error calculating data hash: {e}")
            return None
    
    def _get_model_age(self, model_name: str) -> Optional[timedelta]:
        """Get the age of the current model"""
        try:
            # This would need to be implemented based on your model storage system
            # For now, we'll use a placeholder
            model_path = f"src/models/saved/{model_name}_model.pkl"
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                return timedelta(seconds=time.time() - mtime)
            return None
        
        except Exception as e:
            logger.error(f"Error getting model age: {e}")
            return None
    
    def _get_recent_performance(self, model_name: str) -> Optional[float]:
        """Get recent performance metrics for the model"""
        try:
            # This would need to be implemented based on your monitoring system
            # For now, we'll use a placeholder
            return None
        
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return None
    
    def _get_baseline_performance(self, model_name: str) -> Optional[float]:
        """Get baseline performance metrics for the model"""
        try:
            # This would need to be implemented based on your monitoring system
            # For now, we'll use a placeholder
            return None
        
        except Exception as e:
            logger.error(f"Error getting baseline performance: {e}")
            return None
    
    def _record_retraining(self, trigger: RetrainingTrigger, status: str, duration: float, **kwargs):
        """Record retraining event in history"""
        record = {
            "model_name": trigger.model_name,
            "trigger_type": trigger.trigger_type,
            "trigger_time": trigger.trigger_time.isoformat(),
            "status": status,
            "duration": duration,
            "data_hash": trigger.data_hash,
            "description": trigger.description,
            **kwargs
        }
        
        self.retraining_history.append(record)
        self._save_history()
    
    def _load_history(self):
        """Load retraining history from file"""
        history_file = Path("logs/retraining_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.retraining_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading retraining history: {e}")
                self.retraining_history = []
        else:
            self.retraining_history = []
    
    def _save_history(self):
        """Save retraining history to file"""
        history_file = Path("logs/retraining_history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.retraining_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving retraining history: {e}")
    
    def get_retraining_status(self, model_name: str = None) -> Dict:
        """Get current retraining status"""
        status = {
            "is_running": self.is_running,
            "models": {}
        }
        
        for name, config in self.configs.items():
            if model_name and name != model_name:
                continue
            
            model_status = {
                "enabled": config.enable_auto_retraining,
                "data_path": config.data_path,
                "check_interval": config.check_interval,
                "min_data_size": config.min_data_size,
                "max_model_age_hours": config.max_model_age_hours,
                "performance_threshold": config.performance_threshold,
                "is_retraining": self.retraining_locks[name].locked(),
                "model_age": self._get_model_age(name),
                "data_sufficient": self._check_data_sufficiency(config)
            }
            
            status["models"][name] = model_status
        
        return status
    
    def get_retraining_history(self, model_name: str = None, limit: int = 50) -> List[Dict]:
        """Get retraining history"""
        history = self.retraining_history
        
        if model_name:
            history = [h for h in history if h["model_name"] == model_name]
        
        # Sort by trigger time (newest first) and limit results
        history.sort(key=lambda x: x["trigger_time"], reverse=True)
        return history[:limit]

def create_default_configs() -> Dict[str, RetrainingConfig]:
    """Create default retraining configurations"""
    return {
        "iris": RetrainingConfig(
            model_name="iris",
            data_path="data/raw/iris.csv",
            check_interval=300,  # 5 minutes
            min_data_size=50,
            max_model_age_hours=24,
            performance_threshold=0.05,
            enable_auto_retraining=True
        ),
        "housing": RetrainingConfig(
            model_name="housing",
            data_path="data/raw/housing.csv",
            check_interval=600,  # 10 minutes
            min_data_size=100,
            max_model_age_hours=48,
            performance_threshold=0.03,
            enable_auto_retraining=True
        )
    }

def start_retraining_monitoring(configs: Dict[str, RetrainingConfig] = None):
    """Start the retraining monitoring system"""
    if configs is None:
        configs = create_default_configs()
    
    manager = ModelRetrainingManager(configs)
    manager.start_monitoring()
    return manager
