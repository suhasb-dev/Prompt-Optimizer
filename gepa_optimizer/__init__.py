"""
GEPA Universal Prompt Optimizer

A modern, modular Python library for universal prompt optimization powered by GEPA.
"""

from .core import GepaOptimizer
from .models import OptimizationConfig, OptimizationResult, OptimizedResult, ModelConfig
from .data import UniversalConverter, DataLoader, DataValidator
from .utils import setup_logging, calculate_metrics, sanitize_prompt, APIKeyManager
from .utils.exceptions import GepaOptimizerError, GepaDependencyError, InvalidInputError, DatasetError
from .llms import VisionLLMClient
from .evaluation import UITreeEvaluator

__version__ = "0.1.0"

__all__ = [
    "GepaOptimizer",
    "OptimizationConfig", 
    "OptimizationResult",
    "OptimizedResult",
    "ModelConfig",
    "UniversalConverter",
    "DataLoader",
    "DataValidator",
    "VisionLLMClient",
    "UITreeEvaluator",
    "APIKeyManager",
    "GepaOptimizerError",
    "GepaDependencyError",
    "InvalidInputError",
    "DatasetError",
    "setup_logging",
    "calculate_metrics",
    "sanitize_prompt"
]
