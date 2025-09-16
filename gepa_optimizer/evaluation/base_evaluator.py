"""
Base evaluator class for all evaluation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation strategies.
    
    This enforces a consistent interface while allowing complete customization
    of evaluation logic for any use case.
    """
    
    def __init__(self, metric_weights: Optional[Dict[str, float]] = None):
        """
        Initialize evaluator with optional metric weights.
        
        Args:
            metric_weights: Optional weights for different metrics.
                          If None, subclasses should provide defaults.
        """
        self.metric_weights = metric_weights or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def evaluate(self, predicted: Any, expected: Any) -> Dict[str, float]:
        """
        Evaluate predicted output against expected output.
        
        Args:
            predicted: The model's predicted output
            expected: The ground truth expected output
            
        Returns:
            Dictionary with metric names as keys and scores as values.
            Must include 'composite_score' key for GEPA integration.
        """
        pass
    
    def validate_weights(self) -> bool:
        """Validate that metric weights sum to approximately 1.0"""
        if not self.metric_weights:
            return True
        
        total = sum(self.metric_weights.values())
        return abs(total - 1.0) < 0.01  # Allow small floating point errors
