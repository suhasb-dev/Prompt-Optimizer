"""
Result models for GEPA Optimizer
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

@dataclass
class OptimizationResult:
    """Complete optimization result with all metadata"""
    
    # Identifiers
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core results
    original_prompt: str = ""
    optimized_prompt: str = ""
    
    # Performance metrics
    improvement_data: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Process metadata
    optimization_time: float = 0.0
    dataset_size: int = 0
    total_iterations: int = 0
    
    # Status and error handling
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Reflection history
    reflection_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cost and resource usage
    estimated_cost: Optional[float] = None
    api_calls_made: int = 0
    
    def mark_completed(self):
        """Mark optimization as completed"""
        self.status = "completed"
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: str):
        """Mark optimization as failed"""
        self.status = "failed"
        self.error_message = error
        self.completed_at = datetime.now()

class OptimizedResult:
    """
    User-facing result class that provides clean interface
    """
    
    def __init__(self, result: OptimizationResult):
        self._result = result
    
    @property
    def prompt(self) -> str:
        """The optimized prompt ready for production use"""
        return self._result.optimized_prompt
    
    @property
    def original_prompt(self) -> str:
        """The original seed prompt for reference"""
        return self._result.original_prompt
    
    @property
    def session_id(self) -> str:
        """Unique session identifier"""
        return self._result.session_id
    
    @property
    def improvement_data(self) -> Dict[str, Any]:
        """Performance improvement data"""
        return self._result.improvement_data
    
    @property
    def status(self) -> str:
        """Optimization status"""
        return self._result.status
    
    @property
    def error_message(self) -> Optional[str]:
        """Error message if optimization failed"""
        return self._result.error_message

    @property
    def is_successful(self) -> bool:
        """Whether optimization completed successfully"""
        return (
            self._result.status == "completed" and 
            self._result.error_message is None
        )
    
    @property
    def optimization_time(self) -> float:
        """Time taken for optimization in seconds"""
        return self._result.optimization_time
    
    @property
    def dataset_size(self) -> int:
        """Size of dataset used for optimization"""
        return self._result.dataset_size
    
    @property
    def total_iterations(self) -> int:
        """Total optimization iterations performed"""
        return self._result.total_iterations
    
    @property
    def estimated_cost(self) -> Optional[float]:
        """Estimated cost in USD"""
        return self._result.estimated_cost
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements made"""
        summary = {
            'has_improvement': bool(self._result.improvement_data),
            'optimization_time': self.optimization_time,
            'iterations': self.total_iterations,
            'dataset_size': self.dataset_size
        }
        
        # Add improvement percentage if available
        if 'improvement_percent' in self._result.improvement_data:
            summary['improvement_percent'] = self._result.improvement_data['improvement_percent']
        
        return summary
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection process"""
        if not self._result.reflection_history:
            return {'total_reflections': 0}
        
        return {
            'total_reflections': len(self._result.reflection_history),
            'reflection_points': [
                r.get('summary', 'No summary') 
                for r in self._result.reflection_history[:3]  # First 3
            ]
        }
    
    def get_detailed_result(self) -> OptimizationResult:
        """Get the full detailed result for advanced users"""
        return self._result
    
    def __str__(self) -> str:
        """String representation"""
        status_emoji = "✅" if self.is_successful else "❌" if self.status == "failed" else "⏳"
        return f"OptimizedResult({status_emoji} {self.status}, time={self.optimization_time:.2f}s)"
    
    def __repr__(self) -> str:
        return self.__str__()
