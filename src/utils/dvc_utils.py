"""
DVC utilities for YugenAI project
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DVCManager:
    """DVC management utilities"""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize DVC manager
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        
    def is_initialized(self) -> bool:
        """Check if DVC is initialized in the project"""
        return self.dvc_dir.exists()
    
    def init_dvc(self) -> bool:
        """Initialize DVC in the project directory"""
        try:
            logger.info("Initializing DVC...")
            result = subprocess.run(
                ["dvc", "init"], 
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("DVC initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize DVC: {e}")
            return False
    
    def add_remote(self, name: str, url: str, default: bool = False) -> bool:
        """Add a DVC remote"""
        try:
            logger.info(f"Adding remote '{name}' at {url}")
            subprocess.run(
                ["dvc", "remote", "add", name, url],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if default:
                subprocess.run(
                    ["dvc", "remote", "default", name],
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Set '{name}' as default remote")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add remote '{name}': {e}")
            return False
    
    def add_data_to_dvc(self, data_path: str) -> bool:
        """Add data to DVC tracking"""
        try:
            logger.info(f"Adding {data_path} to DVC tracking")
            subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add {data_path} to DVC: {e}")
            return False
    
    def push_data(self, remote: Optional[str] = None) -> bool:
        """Push data to the remote storage"""
        try:
            cmd = ["dvc", "push"]
            if remote:
                cmd.extend(["--remote", remote])
            
            logger.info("Pushing data to remote storage")
            subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push data: {e}")
            return False
    
    def pull_data(self, remote: Optional[str] = None) -> bool:
        """Pull data from the remote storage"""
        try:
            cmd = ["dvc", "pull"]
            if remote:
                cmd.extend(["--remote", remote])
            
            logger.info("Pulling data from remote storage")
            subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull data: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get DVC status"""
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            logger.error(f"Failed to get DVC status: {e}")
            return {"success": False, "output": "", "error": str(e)}
    
    def run_pipeline(self, pipeline_name: Optional[str] = None) -> bool:
        """Run DVC pipeline"""
        try:
            cmd = ["dvc", "repro"]
            if pipeline_name:
                cmd.append(pipeline_name)
            
            logger.info(f"Running DVC pipeline: {pipeline_name or 'all'}")
            subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run DVC pipeline: {e}")
            return False
    
    def list_remotes(self) -> List[Dict[str, str]]:
        """List DVC remotes"""
        try:
            result = subprocess.run(
                ["dvc", "remote", "list"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            remotes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        remotes.append({
                            "name": parts[0],
                            "url": parts[1]
                        })
            
            return remotes
        except Exception as e:
            logger.error(f"Failed to list remotes: {e}")
            return []
    
    def setup_project_data(self) -> bool:
        """Setup DVC for project data files"""
        data_files = [
            'data/raw/housing.csv',
            'data/raw/iris.csv',
            'data/processed/housing_preprocessed.csv',
            'data/processed/iris_preprocessed.csv'
        ]
        
        model_files = [
            'src/models/saved/housing_model.pkl',
            'src/models/saved/iris_model.pkl'
        ]
        
        success = True
        
        # Add data files
        for data_file in data_files:
            if (self.project_root / data_file).exists():
                if not self.add_data_to_dvc(data_file):
                    success = False
        
        # Add model files
        for model_file in model_files:
            if (self.project_root / model_file).exists():
                if not self.add_data_to_dvc(model_file):
                    success = False
        
        return success

def setup_dvc_remotes():
    """Setup DVC remotes for the project"""
    dvc = DVCManager()
    
    # Setup local storage
    local_storage = Path("dvc-storage")
    local_storage.mkdir(exist_ok=True)
    
    # Add local remote
    dvc.add_remote("default", str(local_storage.absolute()), default=True)
    
    # Add GCS remote if credentials are available
    gcs_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gcs_creds and Path(gcs_creds).exists():
        dvc.add_remote("gcs", "gs://rapid_care/mlops-dvc")
        logger.info("GCS remote added")
    else:
        logger.warning("GCS credentials not found. Using local storage only.")

def main():
    """Example usage"""
    dvc = DVCManager()
    
    if not dvc.is_initialized():
        dvc.init_dvc()
    
    # Setup remotes
    setup_dvc_remotes()
    
    # Setup project data
    dvc.setup_project_data()
    
    # Push data
    dvc.push_data()
    
    # Show status
    status = dvc.get_status()
    print("DVC Status:", status)

if __name__ == "__main__":
    main() 