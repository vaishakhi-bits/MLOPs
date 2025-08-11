#!/usr/bin/env python3
"""
Script to start the monitoring stack for the ML API.
This script starts Prometheus and Grafana using Docker Compose.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_docker():
    """Check if Docker is running"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_monitoring():
    """Start the monitoring stack"""
    print("🚀 Starting ML API Monitoring Stack...")
    
    # Check prerequisites
    if not check_docker():
        print("❌ Docker is not running or not installed. Please start Docker and try again.")
        sys.exit(1)
    
    if not check_docker_compose():
        print("❌ Docker Compose is not available. Please install Docker Compose and try again.")
        sys.exit(1)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    monitoring_compose_file = project_root / "docker-compose.monitoring.yml"
    
    if not monitoring_compose_file.exists():
        print(f"❌ Monitoring compose file not found: {monitoring_compose_file}")
        sys.exit(1)
    
    try:
        # Start the monitoring stack
        print("📊 Starting Prometheus and Grafana...")
        subprocess.run([
            "docker-compose", "-f", str(monitoring_compose_file), "up", "-d"
        ], check=True, cwd=project_root)
        
        print("✅ Monitoring stack started successfully!")
        print("\n📋 Access URLs:")
        print("   • Prometheus: http://localhost:9090")
        print("   • Grafana:    http://localhost:3000 (admin/admin)")
        print("\n📈 Dashboard: ML API Dashboard should be automatically loaded in Grafana")
        print("\n🛑 To stop the monitoring stack, run:")
        print(f"   docker-compose -f {monitoring_compose_file} down")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start monitoring stack: {e}")
        sys.exit(1)

def stop_monitoring():
    """Stop the monitoring stack"""
    print("🛑 Stopping ML API Monitoring Stack...")
    
    project_root = Path(__file__).parent.parent
    monitoring_compose_file = project_root / "docker-compose.monitoring.yml"
    
    try:
        subprocess.run([
            "docker-compose", "-f", str(monitoring_compose_file), "down"
        ], check=True, cwd=project_root)
        
        print("✅ Monitoring stack stopped successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop monitoring stack: {e}")
        sys.exit(1)

def show_status():
    """Show the status of the monitoring stack"""
    print("📊 ML API Monitoring Stack Status:")
    
    project_root = Path(__file__).parent.parent
    monitoring_compose_file = project_root / "docker-compose.monitoring.yml"
    
    try:
        result = subprocess.run([
            "docker-compose", "-f", str(monitoring_compose_file), "ps"
        ], check=True, capture_output=True, text=True, cwd=project_root)
        
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get status: {e}")
        sys.exit(1)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python start_monitoring.py [start|stop|status]")
        print("\nCommands:")
        print("  start  - Start the monitoring stack")
        print("  stop   - Stop the monitoring stack")
        print("  status - Show the status of the monitoring stack")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_monitoring()
    elif command == "stop":
        stop_monitoring()
    elif command == "status":
        show_status()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: start, stop, status")
        sys.exit(1)

if __name__ == "__main__":
    main()
