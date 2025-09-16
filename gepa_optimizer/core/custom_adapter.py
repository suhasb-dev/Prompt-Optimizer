"""
Custom GEPA Adapter for the GEPA Universal Prompt Optimizer
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

# Import ModelConfig
from ..models import ModelConfig

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from ..llms.vision_llm import VisionLLMClient
from ..evaluation.ui_evaluator import UITreeEvaluator
from .base_adapter import BaseGepaAdapter

logger = logging.getLogger(__name__)

class CustomGepaAdapter(BaseGepaAdapter):
    """
    Custom adapter for the GEPA Universal Prompt Optimizer.
    """

    def __init__(self, model_config: 'ModelConfig', metric_weights: Optional[Dict[str, float]] = None):
        """Initialize the custom GEPA adapter with model configuration."""
        # Convert string model to ModelConfig if needed
        if not isinstance(model_config, ModelConfig):
            model_config = ModelConfig(
                provider='openai',
                model_name=str(model_config),
                api_key=None
            )
        
        # Initialize components
        llm_client = VisionLLMClient(
            provider=model_config.provider,
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            frequency_penalty=model_config.frequency_penalty,
            presence_penalty=model_config.presence_penalty
        )
        
        evaluator = UITreeEvaluator(metric_weights=metric_weights)
        
        # Initialize parent class
        super().__init__(llm_client, evaluator)
        
        # Track candidates for logging
        self._last_candidate = None
        self._evaluation_count = 0
        
        self.logger.info(f"🚀 Initialized UI Tree adapter with {model_config.provider}/{model_config.model_name}")

    def _parse_json_safely(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string to dictionary with enhanced parsing and repair."""
        if not json_str or not isinstance(json_str, str):
            return {}
        
        # Try direct parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the string
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try repair and parse
        repaired_json = self._repair_json(json_str)
        if repaired_json:
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                pass
        
        self.logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
        return {}

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues."""
        try:
            # Remove markdown formatting
            json_str = re.sub(r'```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            
            # Remove extra text before/after JSON
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            
            # Fix common issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # Quote unquoted keys
            
            return json_str
        except Exception as e:
            self.logger.warning(f"🔧 JSON repair failed: {e}")
            return ""

    def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate the candidate on a batch of data."""
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
            input_text = item.get('input', '')
            image_base64 = item.get('image', '')
            ground_truth_json = item.get('output', '')

            # Call the LLM client
            llm_response = self.llm_client.generate(system_prompt, input_text, image_base64=image_base64)
            
            # Extract content from the response dictionary
            if isinstance(llm_response, dict):
                llm_output_json_str = llm_response.get("content", "")
                if not llm_output_json_str:
                    llm_output_json_str = str(llm_response)
            else:
                llm_output_json_str = str(llm_response) if llm_response else ""
            
            # 🔍 DEBUG: Log essential info only (removed verbose JSON content)
            self.logger.debug(f"🔍 Sample {i+1} - LLM Response Type: {type(llm_response)}")
            self.logger.debug(f"🔍 Sample {i+1} - Response Length: {len(llm_output_json_str)} chars")
            
            outputs.append(llm_output_json_str)

            # Parse JSON strings to dictionaries for evaluation
            llm_output_dict = self._parse_json_safely(llm_output_json_str)
            ground_truth_dict = self._parse_json_safely(ground_truth_json)

            # Initialize evaluation_results with default values
            evaluation_results = {
                "composite_score": 0.0,
                "element_completeness": 0.0,
                "element_type_accuracy": 0.0,
                "text_content_accuracy": 0.0,
                "hierarchy_accuracy": 0.0,
                "style_accuracy": 0.0
            }

            # Calculate composite score and evaluation results
            if not llm_output_dict and not ground_truth_dict:
                composite_score = 0.1
                evaluation_results = {k: 0.1 for k in evaluation_results.keys()}
                self.logger.warning(f"⚠️  Sample {i+1}: Empty results - using default score: {composite_score}")
            elif not llm_output_dict or not ground_truth_dict:
                composite_score = 0.05
                evaluation_results = {k: 0.05 for k in evaluation_results.keys()}
                self.logger.warning(f"⚠️  Sample {i+1}: Incomplete results - using low score: {composite_score}")
            else:
                # Calculate score using evaluator with parsed dictionaries
                evaluation_results = self.evaluator.evaluate(llm_output_dict, ground_truth_dict)
                composite_score = evaluation_results["composite_score"]
                
                # Clean, readable logging (removed verbose JSON dumps)
                llm_children = len(llm_output_dict.get('children', []))
                gt_children = len(ground_truth_dict.get('children', []))
                
                if composite_score < 0.1:
                    self.logger.warning(f"⚠️  Sample {i+1}: Low score {composite_score:.4f} - LLM: {llm_children} elements, GT: {gt_children} elements")
                    self.logger.debug(f"   Score breakdown: {evaluation_results}")
                else:
                    self.logger.info(f"✅ Sample {i+1}: Score {composite_score:.4f} - LLM: {llm_children} elements, GT: {gt_children} elements")
            
            scores.append(composite_score)

            if capture_traces:
                trajectories.append({
                    'input_text': input_text,
                    'image_base64': image_base64,
                    'ground_truth_json': ground_truth_json,
                    'llm_output_json': llm_output_json_str,
                    'evaluation_results': evaluation_results
                })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Update performance tracking (handled by parent class)
        if avg_score > self._best_score:
            self._best_score = avg_score
            self._best_candidate = candidate.copy()
            self.logger.info(f"🎯 New best candidate found with score: {avg_score:.4f}")
        
        self.logger.info(f"📈 Batch evaluation complete - Average score: {avg_score:.4f}")

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create a reflective dataset from the evaluation results."""
        reflective_dataset = {}
        system_prompt = candidate.get('system_prompt', '')

        # 🎯 NEW: Log the proposed new prompt being evaluated
        self.logger.info(f"📝 Creating reflection dataset for prompt: '{system_prompt[:100]}...'")
        
        # Pretty print reflection dataset creation
        self._log_reflection_dataset_creation(candidate, eval_batch, components_to_update)

        for component in components_to_update:
            reflective_dataset[component] = []
            for i, trace in enumerate(eval_batch.trajectories):
                feedback = self._generate_feedback(trace['evaluation_results'])
                reflective_dataset[component].append({
                    "current_prompt": system_prompt,
                    "input_text": trace['input_text'],
                    "image_base64": trace['image_base64'],
                    "generated_json": trace['llm_output_json'],
                    "ground_truth_json": trace['ground_truth_json'],
                    "score": trace['evaluation_results']["composite_score"],
                    "feedback": feedback,
                    "detailed_scores": trace['evaluation_results']
                })

        # 🎯 NEW: Log reflection dataset summary
        total_samples = sum(len(data) for data in reflective_dataset.values())
        avg_score = sum(trace['score'] for data in reflective_dataset.values() for trace in data) / total_samples if total_samples > 0 else 0.0
        self.logger.info(f"📝 Reflection dataset created - {total_samples} samples, avg score: {avg_score:.4f}")

        return reflective_dataset

    def _generate_feedback(self, evaluation_results: Dict[str, float]) -> str:
        """Generate textual feedback based on evaluation results."""
        composite_score = evaluation_results.get("composite_score", 0.0)
        
        feedback_parts = []
        
        # Overall quality assessment
        if composite_score >= 0.8:
            feedback_parts.append("The overall quality is good.")
        elif composite_score >= 0.5:
            feedback_parts.append("The overall quality is moderate.")
        else:
            feedback_parts.append("The overall quality is low. Focus on fundamental accuracy.")
        
        # Specific metric feedback
        if evaluation_results.get("element_completeness", 0.0) < 0.7:
            feedback_parts.append("Element completeness is low. Ensure all UI elements are captured.")
        
        if evaluation_results.get("element_type_accuracy", 0.0) < 0.7:
            feedback_parts.append("Element type accuracy is low. Verify correct UI element identification (Button, Text, Image, etc.).")
        
        if evaluation_results.get("text_content_accuracy", 0.0) < 0.7:
            feedback_parts.append("Text content accuracy is low. Improve text extraction fidelity.")
        
        if evaluation_results.get("hierarchy_accuracy", 0.0) < 0.7:
            feedback_parts.append("Hierarchy accuracy is low. Ensure correct parent-child relationships.")
        
        if evaluation_results.get("style_accuracy", 0.0) < 0.7:
            feedback_parts.append("Style accuracy is low. Capture more styling properties (colors, sizes, positioning).")
        
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
        
        print("\n" + "="*80)
        print(f"🚀 NEW PROPOSED CANDIDATE (Iteration {iteration})")
        print("="*80)
        print(f"\n📝 PROPOSED PROMPT:")
        print("-" * 40)
        print(f'"{system_prompt}"')
        print("-" * 40)
        print(f"📊 Prompt Length: {len(system_prompt)} characters")
        print(f"📊 Word Count: {len(system_prompt.split())} words")
        print("="*80)
    
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
        
        print("\n" + "="*80)
        print("🔍 REFLECTION DATASET CREATION")
        print("="*80)
        
        print(f"\n📋 CURRENT PROMPT BEING ANALYZED:")
        print("-" * 40)
        print(f'"{system_prompt}"')
        print("-" * 40)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print("-" * 40)
        if eval_batch.scores:
            avg_score = sum(eval_batch.scores) / len(eval_batch.scores)
            min_score = min(eval_batch.scores)
            max_score = max(eval_batch.scores)
            print(f"   • Average Score: {avg_score:.4f}")
            print(f"   • Min Score: {min_score:.4f}")
            print(f"   • Max Score: {max_score:.4f}")
            print(f"   • Total Samples: {len(eval_batch.scores)}")
        
        print(f"\n🎯 COMPONENTS TO UPDATE:")
        print("-" * 40)
        for i, component in enumerate(components_to_update, 1):
            print(f"   {i}. {component}")
        
        if eval_batch.trajectories:
            print(f"\n🔍 DETAILED ANALYSIS:")
            print("-" * 40)
            for i, trace in enumerate(eval_batch.trajectories[:3], 1):  # Show first 3 samples
                evaluation_results = trace['evaluation_results']
                composite_score = evaluation_results.get("composite_score", 0.0)
                
                print(f"\n   📝 Sample {i} (Score: {composite_score:.4f}):")
                
                # Show input data (truncated)
                input_text = trace['input_text'][:100] + "..." if len(trace['input_text']) > 100 else trace['input_text']
                print(f"      Input: \"{input_text}\"")
                
                # Show predicted output (truncated)
                predicted_output = trace['llm_output_json'][:100] + "..." if len(trace['llm_output_json']) > 100 else trace['llm_output_json']
                print(f"      Output: \"{predicted_output}\"")
                
                # Show detailed scores
                print(f"      Detailed Scores:")
                for metric, score in evaluation_results.items():
                    if metric != "composite_score":
                        print(f"        • {metric.replace('_', ' ').title()}: {score:.4f}")
                
                # Show generated feedback
                feedback = self._generate_feedback(evaluation_results)
                print(f"      Feedback: \"{feedback}\"")
            
            if len(eval_batch.trajectories) > 3:
                print(f"\n   ... and {len(eval_batch.trajectories) - 3} more samples")
        
        print("="*80)
