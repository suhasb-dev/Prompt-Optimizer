"""
Universal GEPA adapter for user-defined metrics and LLM clients.
"""

from .base_adapter import BaseGepaAdapter
from ..data.converters import UniversalConverter
from typing import Any, Dict, List, Optional
import logging
import sys
from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

class UniversalGepaAdapter(BaseGepaAdapter):
    """
    Universal GEPA adapter that works with any LLM client and evaluator.
    
    This adapter uses the existing UniversalConverter for data processing
    and delegates LLM generation and evaluation to user-provided components.
    """
    
    def __init__(self, llm_client, evaluator, data_converter=None):
        """
        Initialize universal adapter.
        
        Args:
            llm_client: User-provided LLM client (must inherit from BaseLLMClient)
            evaluator: User-provided evaluator (must inherit from BaseEvaluator)
            data_converter: Optional custom data converter (uses UniversalConverter by default)
        """
        super().__init__(llm_client, evaluator)
        
        # Use existing UniversalConverter for data processing
        self.data_converter = data_converter or UniversalConverter()
        
        # Track candidates for logging
        self._last_candidate = None
        self._evaluation_count = 0
        
        self.logger.info(f"🚀 Initialized Universal adapter with {llm_client.get_model_info()}")
    
    def evaluate(self, batch: List[Dict[str, Any]], candidate: Dict[str, str], 
                capture_traces: bool = False) -> EvaluationBatch:
        """
        Evaluate candidates using user-provided LLM client and evaluator.
        
        This method works with any data type supported by UniversalConverter.
        """
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None
        
        system_prompt = candidate.get('system_prompt', '')
        
        # Check if this is a new candidate (different from last one)
        if self._last_candidate != system_prompt:
            self._evaluation_count += 1
            self.log_proposed_candidate(candidate, self._evaluation_count)
            self._last_candidate = system_prompt
        
        self.logger.info(f"📊 Evaluating {len(batch)} samples with prompt: '{system_prompt[:50]}...'")
        
        for i, item in enumerate(batch):
            # Use existing data processing logic
            standardized_item = self.data_converter._standardize([item])[0]
            
            # Prepare generation parameters
            generation_params = {
                'system_prompt': system_prompt,
                'user_prompt': standardized_item['input']
            }
            
            # Add image if present
            if standardized_item.get('image'):
                generation_params['image_base64'] = standardized_item['image']
            
            # Generate response using user's LLM client
            llm_response = self.llm_client.generate(**generation_params)
            
            # Extract content
            if isinstance(llm_response, dict):
                predicted_output = llm_response.get("content", "")
            else:
                predicted_output = str(llm_response)
            
            outputs.append(predicted_output)
            
            # Evaluate using user's evaluator
            evaluation_results = self.evaluator.evaluate(
                predicted_output, 
                standardized_item['output']
            )
            
            composite_score = evaluation_results.get("composite_score", 0.0)
            scores.append(composite_score)
            
            # Update performance tracking
            self._evaluation_count += 1
            if composite_score > self._best_score:
                self._best_score = composite_score
                self._best_candidate = candidate.copy()
            
            # Capture traces if requested
            if capture_traces:
                trajectories.append({
                    'input_data': standardized_item,
                    'predicted_output': predicted_output,
                    'evaluation_results': evaluation_results
                })
            
            # Logging
            if composite_score < 0.1:
                self.logger.warning(f"⚠️  Sample {i+1}: Low score {composite_score:.4f}")
            else:
                self.logger.info(f"✅ Sample {i+1}: Score {composite_score:.4f}")
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.info(f"📈 Batch evaluation complete - Average score: {avg_score:.4f}")
        
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
    
    def make_reflective_dataset(self, candidate: Dict[str, str], eval_batch: EvaluationBatch, 
                              components_to_update: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create reflective dataset using user-provided evaluator.
        
        This method generates feedback based on the evaluation results
        from the user's custom evaluator.
        """
        reflective_dataset = {}
        system_prompt = candidate.get('system_prompt', '')
        
        self.logger.info(f"📝 Creating reflection dataset for prompt: '{system_prompt[:100]}...'")
        
        # Pretty print reflection dataset creation
        self.logger.info(f"🔍 DEBUG: About to call _log_reflection_dataset_creation")
        self.logger.info(f"🔍 DEBUG: eval_batch has {len(eval_batch.trajectories) if eval_batch.trajectories else 0} trajectories")
        self.logger.info(f"🔍 DEBUG: components_to_update: {components_to_update}")
        
        self._log_reflection_dataset_creation(candidate, eval_batch, components_to_update)
        
        self.logger.info(f"🔍 DEBUG: Finished _log_reflection_dataset_creation")
        
        for component in components_to_update:
            reflective_dataset[component] = []
            for trace in eval_batch.trajectories:
                # Generate feedback based on evaluation results
                feedback = self._generate_feedback(trace['evaluation_results'])
                
                reflective_dataset[component].append({
                    "current_prompt": system_prompt,
                    "input_data": trace['input_data'],
                    "predicted_output": trace['predicted_output'],
                    "score": trace['evaluation_results'].get("composite_score", 0.0),
                    "feedback": feedback,
                    "detailed_scores": trace['evaluation_results']
                })
        
        total_samples = sum(len(data) for data in reflective_dataset.values())
        avg_score = sum(trace['score'] for data in reflective_dataset.values() for trace in data) / total_samples if total_samples > 0 else 0.0
        self.logger.info(f"📝 Reflection dataset created - {total_samples} samples, avg score: {avg_score:.4f}")
        
        return reflective_dataset
    
    def _generate_feedback(self, evaluation_results: Dict[str, float]) -> str:
        """
        Generate feedback based on evaluation results.
        
        This is a generic implementation that works with any evaluator.
        Users can override this in their custom adapters if needed.
        """
        composite_score = evaluation_results.get("composite_score", 0.0)
        
        feedback_parts = []
        
        # Overall quality assessment
        if composite_score >= 0.8:
            feedback_parts.append("The overall quality is good.")
        elif composite_score >= 0.5:
            feedback_parts.append("The overall quality is moderate.")
        else:
            feedback_parts.append("The overall quality needs improvement.")
        
        # Specific metric feedback
        for metric, score in evaluation_results.items():
            if metric != "composite_score" and score < 0.7:
                feedback_parts.append(f"{metric.replace('_', ' ').title()} is low. Focus on improving this aspect.")
        
        return " ".join(feedback_parts)
    
    def get_best_candidate(self) -> Optional[Dict[str, str]]:
        """Get the best candidate found so far."""
        return self._best_candidate
    
    def get_best_score(self) -> float:
        """Get the best score found so far."""
        return self._best_score
    
    def log_proposed_candidate(self, candidate: Dict[str, str], iteration: int = 0):
        """
        Pretty print the new proposed candidate prompt.
        
        Args:
            candidate: The new candidate prompt from GEPA
            iteration: Current optimization iteration
        """
        system_prompt = candidate.get('system_prompt', '')
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🚀 NEW PROPOSED CANDIDATE (Iteration {iteration})")
        self.logger.info("="*80)
        self.logger.info(f"\n📝 PROPOSED PROMPT:")
        self.logger.info("-" * 40)
        self.logger.info(f'"{system_prompt}"')
        self.logger.info("-" * 40)
        self.logger.info(f"📊 Prompt Length: {len(system_prompt)} characters")
        self.logger.info(f"📊 Word Count: {len(system_prompt.split())} words")
        self.logger.info("="*80)
    
    def _log_reflection_dataset_creation(self, candidate: Dict[str, str], eval_batch: EvaluationBatch, 
                                       components_to_update: List[str]):
        """
        Pretty print the reflection dataset creation process.
        
        Args:
            candidate: Current candidate being evaluated
            eval_batch: Evaluation results
            components_to_update: Components being updated
        """
        system_prompt = candidate.get('system_prompt', '')
        
        self.logger.info(f"🔍 DEBUG: Inside _log_reflection_dataset_creation")
        self.logger.info(f"🔍 DEBUG: system_prompt length: {len(system_prompt)}")
        self.logger.info(f"🔍 DEBUG: eval_batch.scores: {eval_batch.scores}")
        self.logger.info(f"🔍 DEBUG: eval_batch.trajectories: {len(eval_batch.trajectories) if eval_batch.trajectories else 0}")
        
        # Use logger for the main output too
        self.logger.info("\n" + "="*80)
        self.logger.info("🔍 REFLECTION DATASET CREATION")
        self.logger.info("="*80)
        
        self.logger.info(f"\n📋 CURRENT PROMPT BEING ANALYZED:")
        self.logger.info("-" * 40)
        self.logger.info(f'"{system_prompt}"')
        self.logger.info("-" * 40)
        
        self.logger.info(f"\n📊 EVALUATION SUMMARY:")
        self.logger.info("-" * 40)
        if eval_batch.scores:
            avg_score = sum(eval_batch.scores) / len(eval_batch.scores)
            min_score = min(eval_batch.scores)
            max_score = max(eval_batch.scores)
            self.logger.info(f"   • Average Score: {avg_score:.4f}")
            self.logger.info(f"   • Min Score: {min_score:.4f}")
            self.logger.info(f"   • Max Score: {max_score:.4f}")
            self.logger.info(f"   • Total Samples: {len(eval_batch.scores)}")
        
        self.logger.info(f"\n🎯 COMPONENTS TO UPDATE:")
        self.logger.info("-" * 40)
        for i, component in enumerate(components_to_update, 1):
            self.logger.info(f"   {i}. {component}")
        
        if eval_batch.trajectories:
            self.logger.info(f"\n🔍 DETAILED ANALYSIS:")
            self.logger.info("-" * 40)
            for i, trace in enumerate(eval_batch.trajectories[:3], 1):  # Show first 3 samples
                evaluation_results = trace['evaluation_results']
                composite_score = evaluation_results.get("composite_score", 0.0)
                
                self.logger.info(f"\n   📝 Sample {i} (Score: {composite_score:.4f}):")
                
                # Show input data (truncated)
                input_data = trace['input_data']
                input_text = input_data.get('input', '')[:100] + "..." if len(input_data.get('input', '')) > 100 else input_data.get('input', '')
                self.logger.info(f"      Input: \"{input_text}\"")
                
                # Show predicted output (truncated)
                predicted_output = trace['predicted_output'][:100] + "..." if len(trace['predicted_output']) > 100 else trace['predicted_output']
                self.logger.info(f"      Output: \"{predicted_output}\"")
                
                # Show detailed scores
                self.logger.info(f"      Detailed Scores:")
                for metric, score in evaluation_results.items():
                    if metric != "composite_score":
                        self.logger.info(f"        • {metric.replace('_', ' ').title()}: {score:.4f}")
                
                # Show generated feedback
                feedback = self._generate_feedback(evaluation_results)
                self.logger.info(f"      Feedback: \"{feedback}\"")
            
            if len(eval_batch.trajectories) > 3:
                self.logger.info(f"\n   ... and {len(eval_batch.trajectories) - 3} more samples")
        
        self.logger.info("="*80)
