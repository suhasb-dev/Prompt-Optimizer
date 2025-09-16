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

# New exports for universal functionality
from .core.base_adapter import BaseGepaAdapter
from .core.universal_adapter import UniversalGepaAdapter
from .llms.base_llm import BaseLLMClient
from .evaluation.base_evaluator import BaseEvaluator

__version__ = "0.1.0"

__all__ = [
    # Core functionality
    "GepaOptimizer",
    "BaseGepaAdapter",
    "UniversalGepaAdapter",
    
    # Configuration
    "OptimizationConfig", 
    "OptimizationResult",
    "OptimizedResult",
    "ModelConfig",
    
    # Data processing
    "UniversalConverter",
    "DataLoader",
    "DataValidator",
    
    # LLM clients
    "VisionLLMClient",
    "BaseLLMClient",
    
    # Evaluators
    "UITreeEvaluator",
    "BaseEvaluator",
    
    # Utilities
    "APIKeyManager",
    "GepaOptimizerError",
    "GepaDependencyError",
    "InvalidInputError",
    "DatasetError",
    "setup_logging",
    "calculate_metrics",
    "sanitize_prompt"
]
