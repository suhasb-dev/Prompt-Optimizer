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

logger = logging.getLogger(__name__)

class CustomGepaAdapter(GEPAAdapter):
    """
    Custom adapter for the GEPA Universal Prompt Optimizer.
    """

    def __init__(self, model_config: 'ModelConfig', metric_weights: Optional[Dict[str, float]] = None):
        """Initialize the custom GEPA adapter with model configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Convert string model to ModelConfig if needed
        if not isinstance(model_config, ModelConfig):
            model_config = ModelConfig(
                provider='openai',
                model_name=str(model_config),
                api_key=None
            )
        
        # Initialize the vision LLM client
        self.vision_llm_client = VisionLLMClient(
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
        
        self.ui_tree_evaluator = UITreeEvaluator(metric_weights=metric_weights)
        self.logger.info(f"🚀 Initialized VisionLLMClient with {model_config.provider}/{model_config.model_name}")

    def _parse_json_safely(self, json_str: str) -> Dict[str, Any]:
        """Safely parse JSON string to dictionary with enhanced parsing and repair."""
        try:
            if isinstance(json_str, dict):
                return json_str
            if isinstance(json_str, str):
                json_str = json_str.strip()
                
                # Remove markdown code blocks
                if json_str.startswith('```json'):
                    json_str = json_str[7:]
                if json_str.startswith('```'):
                    json_str = json_str[3:]
                if json_str.endswith('```'):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                # Try to find JSON within the text if it's embedded
                if '{' in json_str and '}' in json_str:
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    json_str = json_str[start:end]
                
                # Try to parse as-is first
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Attempt JSON repair for common issues
                    repaired_json = self._repair_json(json_str)
                    if repaired_json:
                        return json.loads(repaired_json)
                    else:
                        raise e
                        
            return {}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.warning(f"⚠️  JSON parsing failed: {e}")
            self.logger.debug(f"🔍 Raw LLM output (first 500 chars): {json_str[:500]}")
            self.logger.debug(f"🔍 Raw LLM output (last 200 chars): {json_str[-200:]}")
            return {}
    
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        try:
            # Remove any trailing text after the last }
            if '}' in json_str:
                last_brace = json_str.rfind('}')
                json_str = json_str[:last_brace + 1]
            
            # Fix common issues
            
            # Fix unquoted property names (basic cases)
            # This regex is more careful to avoid double-quoting already quoted properties
            json_str = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # Fix single quotes to double quotes
            json_str = json_str.replace("'", '"')
            
            # Try to close incomplete JSON structures
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # Try to close incomplete arrays
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)
            
            # Test if the repaired JSON is valid
            json.loads(json_str)
            self.logger.info(f"🔧 Successfully repaired malformed JSON")
            return json_str
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.debug(f"🔧 JSON repair failed: {e}")
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

        self.logger.info(f"📊 Evaluating {len(batch)} samples with prompt: '{system_prompt[:50]}...'")

        for i, item in enumerate(batch):
            input_text = item.get('input', '')
            image_base64 = item.get('image', '')
            ground_truth_json = item.get('output', '')

            # Call the VisionLLMClient
            llm_response = self.vision_llm_client.generate(system_prompt, input_text, image_base64)
            
            # Extract content from the response dictionary
            if isinstance(llm_response, dict):
                llm_output_json_str = llm_response.get("content", "")
                if not llm_output_json_str:
                    llm_output_json_str = str(llm_response)
            else:
                llm_output_json_str = str(llm_response) if llm_response else ""
            
            # 🔍 DEBUG: Log what the LLM actually returned
            self.logger.info(f"🔍 Sample {i+1} - LLM Response Type: {type(llm_response)}")
            self.logger.info(f"🔍 Sample {i+1} - LLM Response Keys: {list(llm_response.keys()) if isinstance(llm_response, dict) else 'Not a dict'}")
            self.logger.info(f"🔍 Sample {i+1} - LLM Content Length: {len(llm_output_json_str)}")
            self.logger.info(f"🔍 Sample {i+1} - LLM Content (first 300 chars): {llm_output_json_str[:300]}")
            if len(llm_output_json_str) > 500:
                self.logger.info(f"🔍 Sample {i+1} - LLM Content (last 200 chars): {llm_output_json_str[-200:]}")
            
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
                logger.info(f"LLM Output: {llm_output_dict}")
                logger.info(f"Ground Truth: {ground_truth_dict}")
                # Calculate score using UITreeEvaluator with parsed dictionaries
                evaluation_results = self.ui_tree_evaluator.evaluate(llm_output_dict, ground_truth_dict)
                composite_score = evaluation_results["composite_score"]
                
                # Clean, readable logging
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
        
        self.logger.info(f"�� Created reflective dataset with {len(eval_batch.trajectories)} samples")
        return reflective_dataset

    def _generate_feedback(self, evaluation_results: Dict[str, float]) -> str:
        """Generates textual feedback based on detailed evaluation results."""
        feedback_parts = []
        composite_score = evaluation_results.get("composite_score", 0.0)

        if composite_score < 0.5:
            feedback_parts.append("The overall quality is low. Focus on fundamental accuracy.")
        elif composite_score < 0.8:
            feedback_parts.append("The overall quality is moderate. Aim for higher precision.")
        else:
            feedback_parts.append("The overall quality is good. Refine for perfection.")

        # Add specific feedback based on individual metric scores
        if evaluation_results.get("element_completeness", 1.0) < 0.7:
            feedback_parts.append("Element completeness is low. Ensure all UI elements are captured.")
        if evaluation_results.get("element_type_accuracy", 1.0) < 0.7:
            feedback_parts.append("Element type accuracy is low. Verify correct UI element identification (Button, Text, Image, etc.).")
        if evaluation_results.get("text_content_accuracy", 1.0) < 0.7:
            feedback_parts.append("Text content accuracy is low. Improve text extraction fidelity.")
        if evaluation_results.get("hierarchy_accuracy", 1.0) < 0.7:
            feedback_parts.append("Hierarchy accuracy is low. Ensure correct parent-child relationships.")
        if evaluation_results.get("style_accuracy", 1.0) < 0.7:
            feedback_parts.append("Style accuracy is low. Capture more styling properties (colors, sizes, positioning).")

        return " ".join(feedback_parts)
