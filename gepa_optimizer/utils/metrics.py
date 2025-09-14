"""
Comprehensive metrics calculations for GEPA Optimizer
"""

from typing import Dict, List, Optional, Any
import re
import time
from collections import Counter

def calculate_metrics(original_prompt: str, 
                     optimized_prompt: str,
                     performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive improvement metrics between original and optimized prompts
    
    Args:
        original_prompt: Original seed prompt
        optimized_prompt: GEPA-optimized prompt
        performance_data: Optional performance metrics from GEPA
        
    Returns:
        Dict[str, float]: Comprehensive metrics dictionary
    """
    metrics = {}
    
    # Basic length metrics
    orig_len = len(original_prompt)
    opt_len = len(optimized_prompt)
    
    if orig_len > 0:
        metrics['length_change_percent'] = ((opt_len - orig_len) / orig_len) * 100
    else:
        metrics['length_change_percent'] = 0.0
    
    metrics['original_length'] = orig_len
    metrics['optimized_length'] = opt_len
    
    # Word count metrics
    orig_words = len(original_prompt.split())
    opt_words = len(optimized_prompt.split())
    
    if orig_words > 0:
        metrics['word_change_percent'] = ((opt_words - orig_words) / orig_words) * 100
    else:
        metrics['word_change_percent'] = 0.0
    
    metrics['original_words'] = orig_words
    metrics['optimized_words'] = opt_words
    
    # Complexity metrics
    metrics['original_complexity'] = calculate_text_complexity(original_prompt)
    metrics['optimized_complexity'] = calculate_text_complexity(optimized_prompt)
    metrics['complexity_change'] = metrics['optimized_complexity'] - metrics['original_complexity']
    
    # Similarity metrics
    metrics['similarity_score'] = calculate_similarity(original_prompt, optimized_prompt)
    
    # Include GEPA performance data if available
    if performance_data:
        for key, value in performance_data.items():
            if isinstance(value, (int, float)):
                metrics[f'gepa_{key}'] = float(value)
    
    return metrics

def calculate_text_complexity(text: str) -> float:
    """
    Calculate a simple complexity score for text
    
    Args:
        text: Text to analyze
        
    Returns:
        float: Complexity score (higher = more complex)
    """
    if not text:
        return 0.0
    
    # Count various complexity indicators
    sentence_count = len(re.findall(r'[.!?]+', text))
    word_count = len(text.split())
    char_count = len(text)
    unique_words = len(set(text.lower().split()))
    
    # Avoid division by zero
    if word_count == 0:
        return 0.0
    
    # Simple complexity calculation
    avg_word_length = char_count / word_count
    lexical_diversity = unique_words / word_count
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Weighted complexity score
    complexity = (
        avg_word_length * 0.3 +
        lexical_diversity * 0.4 +
        avg_sentence_length * 0.3
    )
    
    return round(complexity, 3)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple word overlap
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    similarity = intersection / union
    return round(similarity, 3)

def track_optimization_progress(iteration: int, 
                               score: float, 
                               improvement: float,
                               time_elapsed: float) -> Dict[str, Any]:
    """
    Track progress during optimization iterations
    
    Args:
        iteration: Current iteration number
        score: Current performance score
        improvement: Improvement over baseline
        time_elapsed: Time elapsed in seconds
        
    Returns:
        Dict[str, Any]: Progress metrics
    """
    return {
        'iteration': iteration,
        'score': round(score, 4),
        'improvement': round(improvement, 4),
        'time_elapsed': round(time_elapsed, 2),
        'score_per_second': round(score / max(time_elapsed, 0.001), 4)
    }

def calculate_cost_efficiency(improvement_percent: float, 
                            estimated_cost: float) -> Dict[str, float]:
    """
    Calculate cost efficiency metrics
    
    Args:
        improvement_percent: Performance improvement percentage
        estimated_cost: Estimated cost in USD
        
    Returns:
        Dict[str, float]: Cost efficiency metrics
    """
    if estimated_cost <= 0:
        return {'improvement_per_dollar': 0.0, 'cost_efficiency': 0.0}
    
    improvement_per_dollar = improvement_percent / estimated_cost
    
    # Cost efficiency score (higher is better)
    cost_efficiency = min(improvement_per_dollar / 10.0, 1.0)  # Normalized to 0-1
    
    return {
        'improvement_per_dollar': round(improvement_per_dollar, 3),
        'cost_efficiency': round(cost_efficiency, 3),
        'estimated_cost': estimated_cost
    }

def summarize_optimization_results(metrics: Dict[str, float]) -> str:
    """
    Create a human-readable summary of optimization results
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        
    Returns:
        str: Human-readable summary
    """
    summary_parts = []
    
    # Length changes
    length_change = metrics.get('length_change_percent', 0)
    if length_change > 5:
        summary_parts.append(f"Prompt expanded by {length_change:.1f}%")
    elif length_change < -5:
        summary_parts.append(f"Prompt condensed by {abs(length_change):.1f}%")
    else:
        summary_parts.append("Prompt length remained similar")
    
    # Complexity changes
    complexity_change = metrics.get('complexity_change', 0)
    if complexity_change > 0.1:
        summary_parts.append("increased complexity")
    elif complexity_change < -0.1:
        summary_parts.append("reduced complexity")
    else:
        summary_parts.append("maintained similar complexity")
    
    # Similarity
    similarity = metrics.get('similarity_score', 1.0)
    if similarity > 0.8:
        summary_parts.append(f"high similarity to original ({similarity:.2f})")
    elif similarity > 0.5:
        summary_parts.append(f"moderate changes from original ({similarity:.2f})")
    else:
        summary_parts.append(f"significant changes from original ({similarity:.2f})")
    
    return f"Optimization results: {', '.join(summary_parts)}"
