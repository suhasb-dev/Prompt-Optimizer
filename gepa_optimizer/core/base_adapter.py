"""
Base adapter class for all GEPA adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from gepa.core.adapter import GEPAAdapter, EvaluationBatch

from ..llms.base_llm import BaseLLMClient
from ..evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class BaseGepaAdapter(GEPAAdapter, ABC):
    """
    Abstract base class for GEPA adapters.
    
    Provides the foundation for creating task-specific adapters while
    maintaining compatibility with the GEPA framework.
    """
    
    def __init__(self, llm_client: BaseLLMClient, evaluator: BaseEvaluator):
        """
        Initialize adapter with LLM client and evaluator.
        
        Args:
            llm_client: LLM client for generating responses
            evaluator: Evaluator for scoring predictions
        """
        if not isinstance(llm_client, BaseLLMClient):
            raise TypeError("llm_client must be an instance of BaseLLMClient")
        if not isinstance(evaluator, BaseEvaluator):
            raise TypeError("evaluator must be an instance of BaseEvaluator")
        
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self._evaluation_count = 0
        self._best_score = 0.0
        self._best_candidate = None
    
    @abstractmethod
    def evaluate(self, batch: List[Dict[str, Any]], candidate: Dict[str, str], 
                capture_traces: bool = False) -> EvaluationBatch:
        """
        Evaluate candidate on a batch of data.
        
        Args:
            batch: List of data items to evaluate
            candidate: Prompt candidate to evaluate
            capture_traces: Whether to capture detailed traces
            
        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        pass
    
    @abstractmethod
    def make_reflective_dataset(self, candidate: Dict[str, str], 
                              eval_batch: EvaluationBatch, 
                              components_to_update: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create reflective dataset for GEPA's reflection process.
        
        Args:
            candidate: Current prompt candidate
            eval_batch: Results from evaluation
            components_to_update: List of components to update
            
        Returns:
            Dictionary mapping components to reflection data
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            'evaluation_count': self._evaluation_count,
            'best_score': self._best_score,
            'model_info': self.llm_client.get_model_info(),
            'evaluator_class': self.evaluator.__class__.__name__
        }
