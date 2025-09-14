"""
Result processing for GEPA Optimizer
Handles extraction and processing of GEPA optimization results
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ResultProcessor:
    """
    Processes raw GEPA optimization results into clean, usable formats
    """
    
    @staticmethod
    def extract_optimized_prompt(result: Any) -> str:
        """
        Extract the optimized prompt from GEPA result object
        
        Args:
            result: Raw GEPA optimization result
            
        Returns:
            str: The optimized prompt text
        """
        try:
            # Try multiple possible result structures
            if hasattr(result, 'best_candidate'):
                candidate = result.best_candidate
                
                if isinstance(candidate, dict):
                    # Try common prompt keys
                    for key in ['system_prompt', 'prompt', 'text']:
                        if key in candidate:
                            return str(candidate[key])
                    
                    # If no standard key found, return string representation
                    return str(candidate)
                else:
                    return str(candidate)
            
            # Fallback - convert entire result to string
            return str(result)
            
        except Exception as e:
            logger.warning(f"Failed to extract optimized prompt: {e}")
            return "Optimization completed (prompt extraction failed)"
    
    @staticmethod
    def extract_metrics(result: Any) -> Dict[str, Any]:
        """
        Extract performance metrics from GEPA result
        
        Args:
            result: Raw GEPA optimization result
            
        Returns:
            Dict[str, Any]: Extracted metrics
        """
        metrics = {}
        
        try:
            # Extract common metrics
            if hasattr(result, 'best_score'):
                metrics['best_score'] = float(result.best_score)
            
            if hasattr(result, 'baseline_score'):
                metrics['baseline_score'] = float(result.baseline_score)
            
            if hasattr(result, 'improvement'):
                metrics['improvement'] = float(result.improvement)
            
            if hasattr(result, 'iterations'):
                metrics['iterations'] = int(result.iterations)
            
            # Calculate improvement percentage if we have both scores
            if 'best_score' in metrics and 'baseline_score' in metrics:
                baseline = metrics['baseline_score']
                if baseline > 0:
                    improvement_percent = ((metrics['best_score'] - baseline) / baseline) * 100
                    metrics['improvement_percent'] = round(improvement_percent, 2)
            
            # Extract additional metadata
            if hasattr(result, 'metadata'):
                metrics['metadata'] = result.metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
        
        return metrics
    
    @staticmethod
    def extract_reflection_history(result: Any) -> list:
        """
        Extract reflection/optimization history from GEPA result
        
        Args:
            result: Raw GEPA optimization result
            
        Returns:
            list: List of reflection iterations
        """
        history = []
        
        try:
            if hasattr(result, 'optimization_history'):
                for i, iteration in enumerate(result.optimization_history):
                    history_item = {
                        'iteration': i,
                        'score': iteration.get('score', 0.0),
                        'candidate': iteration.get('candidate', {}),
                        'feedback': iteration.get('feedback', ''),
                        'improvement': iteration.get('improvement', 0.0)
                    }
                    history.append(history_item)
            
        except Exception as e:
            logger.warning(f"Failed to extract reflection history: {e}")
        
        return history
    
    @staticmethod
    def process_full_result(result: Any, original_prompt: str, optimization_time: float) -> Dict[str, Any]:
        """
        Process complete GEPA result into structured format
        
        Args:
            result: Raw GEPA optimization result
            original_prompt: Original seed prompt
            optimization_time: Time taken for optimization
            
        Returns:
            Dict[str, Any]: Complete processed result
        """
        return {
            'original_prompt': original_prompt,
            'optimized_prompt': ResultProcessor.extract_optimized_prompt(result),
            'metrics': ResultProcessor.extract_metrics(result),
            'reflection_history': ResultProcessor.extract_reflection_history(result),
            'optimization_time': optimization_time,
            'raw_result': result  # Keep raw result for advanced users
        }
