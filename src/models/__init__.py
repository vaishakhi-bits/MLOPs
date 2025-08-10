"""
Model training and management module for YugenAI project
"""

from .train_iris import IrisModelTrainer
from .train_housing import HousingModelTrainer
from .train import train_iris_model, train_housing_model

__all__ = [
    'IrisModelTrainer',
    'HousingModelTrainer', 
    'train_iris_model',
    'train_housing_model'
] 