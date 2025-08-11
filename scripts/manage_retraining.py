#!/usr/bin/env python3
"""
CLI script to manage model retraining system.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.retraining import (
    ModelRetrainingManager, 
    RetrainingConfig, 
    create_default_configs,
    start_retraining_monitoring
)
from src.utils.logger import setup_logger

logger = setup_logger("retraining_cli")

class RetrainingCLI:
    """Command-line interface for model retraining management"""
    
    def __init__(self):
        self.manager: Optional[ModelRetrainingManager] = None
        self.config_file = Path("config/retraining_config.json")
    
    def load_configs(self) -> Dict[str, RetrainingConfig]:
        """Load retraining configurations from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                configs = {}
                for model_name, config_dict in config_data.items():
                    configs[model_name] = RetrainingConfig(**config_dict)
                
                logger.info(f"Loaded configurations for {len(configs)} models")
                return configs
            
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configurations")
        
        return create_default_configs()
    
    def save_configs(self, configs: Dict[str, RetrainingConfig]):
        """Save retraining configurations to file"""
        self.config_file.parent.mkdir(exist_ok=True)
        
        config_data = {}
        for model_name, config in configs.items():
            config_data[model_name] = {
                "model_name": config.model_name,
                "data_path": config.data_path,
                "check_interval": config.check_interval,
                "min_data_size": config.min_data_size,
                "max_model_age_hours": config.max_model_age_hours,
                "performance_threshold": config.performance_threshold,
                "enable_auto_retraining": config.enable_auto_retraining
            }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configurations saved to {self.config_file}")
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def start(self, args):
        """Start the retraining monitoring system"""
        print("Starting Model Retraining System...")
        
        configs = self.load_configs()
        
        # Override configs with command line arguments
        if args.iris_data:
            configs["iris"].data_path = args.iris_data
        if args.housing_data:
            configs["housing"].data_path = args.housing_data
        
        # Save updated configs
        self.save_configs(configs)
        
        # Start monitoring
        self.manager = start_retraining_monitoring(configs)
        
        print("Retraining system started successfully!")
        print(f"Monitoring {len(configs)} models:")
        for model_name, config in configs.items():
            status = "Enabled" if config.enable_auto_retraining else "❌ Disabled"
            print(f"   • {model_name}: {status}")
            print(f"     Data: {config.data_path}")
            print(f"     Check interval: {config.check_interval}s")
            print(f"     Max age: {config.max_model_age_hours}h")
        
        print("\nAvailable commands:")
        print("   python scripts/manage_retraining.py status")
        print("   python scripts/manage_retraining.py history")
        print("   python scripts/manage_retraining.py trigger <model>")
        print("   python scripts/manage_retraining.py stop")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping retraining system...")
            self.stop(args)
    
    def stop(self, args):
        """Stop the retraining monitoring system"""
        if self.manager:
            self.manager.stop_monitoring()
            print("Retraining system stopped")
        else:
            print("No retraining system running")
    
    def status(self, args):
        """Show retraining system status"""
        configs = self.load_configs()
        
        print("Model Retraining System Status")
        print("=" * 50)
        
        # Check if monitoring is running
        if self.manager and self.manager.is_running:
            print("Monitoring: Running")
        else:
            print("Monitoring: Stopped")
        
        print(f"\nConfiguration File: {self.config_file}")
        print(f"Configuration Status: {'Loaded' if self.config_file.exists() else 'Not found'}")
        
        print(f"\nModels ({len(configs)}):")
        for model_name, config in configs.items():
            status = "Enabled" if config.enable_auto_retraining else "Disabled"
            print(f"\n   {model_name.upper()}:")
            print(f"     Status: {status}")
            print(f"     Data Path: {config.data_path}")
            print(f"     Check Interval: {config.check_interval}s")
            print(f"     Min Data Size: {config.min_data_size}")
            print(f"     Max Model Age: {config.max_model_age_hours}h")
            print(f"     Performance Threshold: {config.performance_threshold}")
            
            # Check data file
            data_exists = Path(config.data_path).exists()
            print(f"     Data File: {'Exists' if data_exists else 'Missing'}")
            
            if data_exists:
                try:
                    import pandas as pd
                    df = pd.read_csv(config.data_path)
                    print(f"     Records: {len(df)}")
                except Exception as e:
                    print(f"     Records: Error reading file")
        
        # Show recent history if available
        if self.manager:
            history = self.manager.get_retraining_history(limit=5)
            if history:
                print(f"\nRecent Retraining History:")
                for record in history:
                    status_icon = "✅" if record["status"] == "success" else "❌"
                    print(f"   {status_icon} {record['model_name']} - {record['trigger_type']} - {record['trigger_time']}")
    
    def history(self, args):
        """Show retraining history"""
        configs = self.load_configs()
        
        print("📈 Model Retraining History")
        print("=" * 50)
        
        # Load history from file
        history_file = Path("logs/retraining_history.json")
        if not history_file.exists():
            print("No retraining history found")
            return
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            return
        
        if not history:
            print("No retraining events recorded")
            return
        
        # Filter by model if specified
        if args.model:
            history = [h for h in history if h["model_name"] == args.model]
            if not history:
                print(f"No history found for model: {args.model}")
                return
        
        # Sort by time (newest first)
        history.sort(key=lambda x: x["trigger_time"], reverse=True)
        
        # Limit results
        limit = args.limit if args.limit else 20
        history = history[:limit]
        
        print(f"📋 Showing last {len(history)} retraining events:")
        print()
        
        for i, record in enumerate(history, 1):
            status_icon = "✅" if record["status"] == "success" else "❌"
            duration = f"{record['duration']:.2f}s" if "duration" in record else "N/A"
            
            print(f"{i:2d}. {status_icon} {record['model_name'].upper()}")
            print(f"     Trigger: {record['trigger_type']}")
            print(f"     Time: {record['trigger_time']}")
            print(f"     Duration: {duration}")
            print(f"     Description: {record.get('description', 'N/A')}")
            
            if record["status"] == "failure" and "error" in record:
                print(f"     Error: {record['error']}")
            
            print()
    
    def trigger(self, args):
        """Manually trigger model retraining"""
        if not args.model:
            print("Please specify a model to retrain")
            print("   Usage: python scripts/manage_retraining.py trigger <model>")
            return
        
        configs = self.load_configs()
        
        if args.model not in configs:
            print(f"Unknown model: {args.model}")
            print(f"   Available models: {', '.join(configs.keys())}")
            return
        
        config = configs[args.model]
        
        print(f"Manually triggering retraining for {args.model}...")
        
        # Create temporary manager for manual trigger
        temp_manager = ModelRetrainingManager({args.model: config})
        
        # Trigger retraining
        temp_manager.trigger_retraining(
            args.model,
            "manual",
            "Manual trigger via CLI"
        )
        
        print(f"Retraining triggered for {args.model}")
        print("Check status with: python scripts/manage_retraining.py status")
    
    def config(self, args):
        """Configure retraining settings"""
        configs = self.load_configs()
        
        if args.model and args.model not in configs:
            print(f"Unknown model: {args.model}")
            print(f"   Available models: {', '.join(configs.keys())}")
            return
        
        if args.model:
            # Configure specific model
            config = configs[args.model]
            print(f"Configuring {args.model} retraining settings:")
            print(f"   Current settings:")
            print(f"     Enabled: {config.enable_auto_retraining}")
            print(f"     Check Interval: {config.check_interval}s")
            print(f"     Min Data Size: {config.min_data_size}")
            print(f"     Max Model Age: {config.max_model_age_hours}h")
            print(f"     Performance Threshold: {config.performance_threshold}")
            
            # Interactive configuration
            print(f"\n   Enter new values (press Enter to keep current):")
            
            enabled = input(f"   Enabled (y/n) [{config.enable_auto_retraining}]: ").strip()
            if enabled.lower() in ['y', 'yes']:
                config.enable_auto_retraining = True
            elif enabled.lower() in ['n', 'no']:
                config.enable_auto_retraining = False
            
            check_interval = input(f"   Check Interval (seconds) [{config.check_interval}]: ").strip()
            if check_interval:
                config.check_interval = int(check_interval)
            
            min_data_size = input(f"   Min Data Size [{config.min_data_size}]: ").strip()
            if min_data_size:
                config.min_data_size = int(min_data_size)
            
            max_age = input(f"   Max Model Age (hours) [{config.max_model_age_hours}]: ").strip()
            if max_age:
                config.max_model_age_hours = int(max_age)
            
            threshold = input(f"   Performance Threshold [{config.performance_threshold}]: ").strip()
            if threshold:
                config.performance_threshold = float(threshold)
        
        else:
            # Show all configurations
            print("Current Retraining Configurations:")
            for model_name, config in configs.items():
                print(f"\n   {model_name.upper()}:")
                print(f"     Enabled: {config.enable_auto_retraining}")
                print(f"     Data Path: {config.data_path}")
                print(f"     Check Interval: {config.check_interval}s")
                print(f"     Min Data Size: {config.min_data_size}")
                print(f"     Max Model Age: {config.max_model_age_hours}h")
                print(f"     Performance Threshold: {config.performance_threshold}")
        
        # Save configurations
        self.save_configs(configs)
        print(f"\nConfiguration saved to {self.config_file}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Model Retraining Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start retraining monitoring")
    start_parser.add_argument("--iris-data", help="Path to iris data file")
    start_parser.add_argument("--housing-data", help="Path to housing data file")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop retraining monitoring")
    
    # Status command
    subparsers.add_parser("status", help="Show retraining system status")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show retraining history")
    history_parser.add_argument("--model", help="Filter by model name")
    history_parser.add_argument("--limit", type=int, help="Limit number of records")
    
    # Trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Manually trigger retraining")
    trigger_parser.add_argument("model", help="Model to retrain")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure retraining settings")
    config_parser.add_argument("--model", help="Model to configure (if not specified, shows all)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = RetrainingCLI()
    
    try:
        if args.command == "start":
            cli.start(args)
        elif args.command == "stop":
            cli.stop(args)
        elif args.command == "status":
            cli.status(args)
        elif args.command == "history":
            cli.history(args)
        elif args.command == "trigger":
            cli.trigger(args)
        elif args.command == "config":
            cli.config(args)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
