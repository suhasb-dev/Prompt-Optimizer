"""
Main GepaOptimizer class - the heart of the optimization system
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union
import asyncio
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import gepa
from ..utils.api_keys import APIKeyManager
from .custom_adapter import CustomGepaAdapter
from .result import ResultProcessor
from ..data.converters import UniversalConverter
from ..models.result import OptimizationResult, OptimizedResult
from ..models.config import OptimizationConfig, ModelConfig
from ..utils.helpers import sanitize_prompt
from ..utils.exceptions import GepaDependencyError, InvalidInputError, DatasetError, GepaOptimizerError

logger = logging.getLogger(__name__)

class GepaOptimizer:
    """
    Main class for prompt optimization using GEPA
    
    This is the primary interface that users interact with.
    Provides both simple and advanced optimization capabilities.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None, llm_model_name: Optional[str] = None, metric_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the optimizer
        
        Args:
            config: Optimization configuration (required)
            llm_model_name: [Deprecated] Use config.model instead. Will be removed in future versions.
            metric_weights: Optional dictionary of weights for evaluation metrics.
            
        Raises:
            ValueError: If required configuration is missing
            GepaDependencyError: If GEPA library is not available
        """
        if config is None:
            raise ValueError("config parameter is required. Use OptimizationConfig to configure the optimizer.")
            
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.converter = UniversalConverter()
        self.api_manager = APIKeyManager()
        self.result_processor = ResultProcessor()
        
        # Initialize adapter with model config
        self.custom_adapter = CustomGepaAdapter(
            model_config=self.config.model,
            metric_weights=metric_weights
        )
        
        # Log model configuration
        model_config = self.config.model
        model_info = {
            'provider': getattr(model_config, 'provider', 'unknown'),
            'model_name': getattr(model_config, 'model_name', str(model_config)),
        }
        self.logger.info(f"Initialized model with config: {model_info}")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Validate GEPA availability
        if gepa is None:
            raise GepaDependencyError("GEPA library is not available. Please install it with: pip install gepa")

    async def train(self,
                   seed_prompt: str,
                   dataset: Union[List[Any], str, Dict, Any],
                   **kwargs) -> OptimizedResult:
        """
        Main training method for prompt optimization
        
        Args:
            seed_prompt: Initial prompt to optimize
            dataset: Training data in any format
            **kwargs: Additional parameters that can override config
            
        Returns:
            OptimizedResult: Optimization result with improved prompt
            
        Raises:
            InvalidInputError: For invalid input parameters
            DatasetError: For issues with dataset processing
            GepaOptimizerError: For optimization failures
        """
        start_time = time.time()
        session_id = f"opt_{int(start_time)}_{id(self)}"
        
        try:
            self.logger.info(f"Starting optimization session: {session_id}")
            self.logger.info(f"Using model: {self.config.model.model_name} (provider: {self.config.model.provider})")
            
            # Update config with any overrides from kwargs
            self._update_config_from_kwargs(kwargs)
            
            # Step 1: Validate inputs
            self._validate_inputs(seed_prompt)
            
            # Step 2: Convert dataset to GEPA format
            self.logger.info("Converting dataset to GEPA format...")
            trainset, valset = self.converter.convert(dataset)
            
            if not trainset:
                raise DatasetError("Dataset appears to be empty after conversion")
            
            self.logger.info(f"Dataset converted: {len(trainset)} train, {len(valset)} validation samples")
            
            # Step 3: Create seed candidate
            seed_candidate = self._create_seed_candidate(seed_prompt)
            
            # Step 4: Run GEPA optimization
            print("🚀 Starting GEPA optimization...")
            gepa_result = await self._run_gepa_optimization(
                adapter=self.custom_adapter,
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                **kwargs
            )
            
            # Step 5: Process results
            optimization_time = time.time() - start_time
            
            processed_result = self.result_processor.process_full_result(
                result=gepa_result,
                original_prompt=seed_prompt,
                optimization_time=optimization_time
            )
            
            # Step 6: Create result objects
            result = OptimizedResult(
                original_prompt=seed_prompt,
                optimized_prompt=processed_result.get('optimized_prompt', seed_prompt),
                improvement_data=processed_result.get('improvement_data', {}),
                optimization_time=optimization_time,
                dataset_size=len(trainset) + len(valset),
                total_iterations=processed_result.get('total_iterations', 0),
                status=processed_result.get('status', 'completed'),
                error_message=processed_result.get('error_message'),
                detailed_result=OptimizationResult(
                    session_id=session_id,
                    original_prompt=seed_prompt,
                    optimized_prompt=processed_result.get('optimized_prompt', seed_prompt),
                    improvement_data=processed_result.get('improvement_data', {}),
                    optimization_time=optimization_time,
                    dataset_size=len(trainset) + len(valset),
                    total_iterations=processed_result.get('total_iterations', 0),
                    status=processed_result.get('status', 'completed'),
                    error_message=processed_result.get('error_message')
                )
            )
            
            self.logger.info(f"✅ Optimization completed in {optimization_time:.2f}s")
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            error_msg = f"Optimization failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return failed result
            return OptimizedResult(
                original_prompt=seed_prompt,
                optimized_prompt=seed_prompt,  # Return original on failure
                improvement_data={'error': error_msg},
                optimization_time=optimization_time,
                dataset_size=0,
                total_iterations=0,
                status='failed',
                error_message=error_msg
            )

    def _update_config_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration with runtime overrides from kwargs."""
        updated_params = []
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                updated_params.append(f"{key}={value}")
            else:
                self.logger.warning(f"Unknown parameter '{key}' ignored")
        
        if updated_params:
            self.logger.info(f"Updated config parameters: {', '.join(updated_params)}")

    def _validate_inputs(self, seed_prompt: str) -> None:
        """
        Validate input parameters for optimization
        
        Args:
            seed_prompt: The seed prompt to validate
            
        Raises:
            InvalidInputError: If validation fails
        """
        if not seed_prompt or not isinstance(seed_prompt, str):
            raise InvalidInputError("Seed prompt must be a non-empty string")
        
        if len(seed_prompt.strip()) < 10:
            raise InvalidInputError("Seed prompt is too short (minimum 10 characters)")
        
        # Validate model configuration
        model_config = self.config.model
        if not hasattr(model_config, 'model_name') or not model_config.model_name:
            raise InvalidInputError("Model name is required")
        
        reflection_config = self.config.reflection_model
        if not hasattr(reflection_config, 'model_name') or not reflection_config.model_name:
            raise InvalidInputError("Reflection model name is required")

    def _validate_models(self, task_lm, reflection_lm):
        """Validate if specified models are supported."""
        supported_prefixes = ['openai/', 'anthropic/', 'gpt', 'claude', 'llama', 'mistral', 'cohere', 'ai21']
        
        task_model_str = str(task_lm).lower()
        reflection_model_str = str(reflection_lm).lower()
        
        task_supported = any(task_model_str.startswith(prefix) for prefix in supported_prefixes)
        reflection_supported = any(reflection_model_str.startswith(prefix) for prefix in supported_prefixes)
        
        if not task_supported:
            self.logger.warning(f"Task model '{task_lm}' may not be supported")
        if not reflection_supported:
            self.logger.warning(f"Reflection model '{reflection_lm}' may not be supported")

    def _create_seed_candidate(self, seed_prompt: str) -> Dict[str, str]:
        """Create a seed candidate from the input prompt."""
        sanitized_prompt = sanitize_prompt(seed_prompt)
        return {'system_prompt': sanitized_prompt}

    async def _run_gepa_optimization(self, adapter: CustomGepaAdapter, seed_candidate: Any, trainset: List[Any], valset: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Run GEPA optimization with the given adapter and data
        
        Args:
            adapter: Custom adapter for GEPA
            seed_candidate: Initial prompt candidate
            trainset: Training dataset
            valset: Validation dataset
            **kwargs: Additional optimization parameters that can override config
            
        Returns:
            Dict with optimization results
            
        Raises:
            GepaOptimizerError: If optimization fails
            
        Note:
            The following parameters are required in the config:
            - max_metric_calls: Maximum number of metric evaluations
            - batch_size: Batch size for evaluation
            - max_iterations: Maximum number of optimization iterations
        """
        try:
            # Get optimization parameters from config (these are required fields)
            max_metric_calls = self.config.max_metric_calls
            batch_size = self.config.batch_size
            max_iterations = self.config.max_iterations

            # Create reflection model client
            from ..llms.vision_llm import VisionLLMClient
            reflection_lm_client = VisionLLMClient(
                provider=self.config.reflection_model.provider,
                model_name=self.config.reflection_model.model_name,
                api_key=self.config.reflection_model.api_key,
                base_url=self.config.reflection_model.base_url,
                temperature=self.config.reflection_model.temperature,
                max_tokens=self.config.reflection_model.max_tokens,
                top_p=self.config.reflection_model.top_p,
                frequency_penalty=self.config.reflection_model.frequency_penalty,
                presence_penalty=self.config.reflection_model.presence_penalty
            )

            # Create callable wrapper for GEPA
            def reflection_lm_callable(prompt: str) -> str:
                """Callable wrapper for reflection model that GEPA expects"""
                try:
                    # For reflection, we only need text generation (no images)
                    result = reflection_lm_client.generate(
                        system_prompt="You are a helpful assistant that provides feedback and suggestions for improving prompts.",
                        user_prompt=prompt,
                        image_base64=""  # No image for reflection
                    )
                    
                    # Extract string content from the result dictionary
                    if isinstance(result, dict):
                        return result.get("content", str(result))
                    else:
                        return str(result)
                    
                except Exception as e:
                    self.logger.error(f"Reflection model error: {e}")
                    return prompt  # Return original prompt on error

            self.logger.info(
                f"Starting GEPA optimization with {max_iterations} iterations, "
                f"batch size {batch_size}, max metric calls: {max_metric_calls}"
            )
            self.logger.info(
                f"GEPA parameters: candidate_selection_strategy=pareto, "
                f"reflection_minibatch_size={batch_size}, "
                f"skip_perfect_score=False, "
                f"module_selector=round_robin"
            )
            
            # Prepare optimization parameters with ONLY valid GEPA parameters
            gepa_params = {
                'adapter': adapter,
                'seed_candidate': seed_candidate,
                'trainset': trainset,
                'valset': valset,
                'max_metric_calls': max_metric_calls,
                'reflection_lm': reflection_lm_callable,  # Pass callable function
                
                # Valid GEPA parameters based on actual library
                'candidate_selection_strategy': 'pareto',  # Use Pareto selection
                'skip_perfect_score': False,  # Don't skip perfect scores
                'reflection_minibatch_size': batch_size,  # Use batch size for reflection
                'perfect_score': 1.0,  # Perfect score threshold
                'module_selector': 'round_robin',  # Cycle through components
                'display_progress_bar': self.config.verbose,  # Show progress if verbose
                'raise_on_exception': True,  # Raise exceptions for debugging
                
                **kwargs
            }
            
            # 🎯 NEW: Capture GEPA's internal logging for pareto front information
            gepa_output = io.StringIO()
            
            # Run GEPA optimization (synchronous call wrapped in async)
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._run_gepa_with_logging(gepa_params, gepa_output)
            )
            
            # 🎯 NEW: Process and log pareto front information
            gepa_logs = gepa_output.getvalue()
            self._log_pareto_front_info(gepa_logs)
            
            return result
        except Exception as e:
            # Try to extract partial results before failing
            self.logger.warning(f"GEPA optimization failed: {e}")
            
            # Check if we have any cached results from the adapter
            best_candidate = adapter.get_best_candidate()
            best_score = adapter.get_best_score()
            
            if best_candidate and best_score > 0:
                self.logger.info(f"🎯 Using cached best result with score: {best_score:.4f}")
                
                # Create a mock GEPA result with the best candidate found
                return {
                    'best_candidate': best_candidate,
                    'best_score': best_score,
                    'partial_result': True,
                    'error': f'GEPA failed but returning best result found: {str(e)}'
                }
            else:
                # If no cached results, re-raise the error
                raise GepaOptimizerError(f"GEPA optimization failed: {str(e)}")
    
    def _run_gepa_with_logging(self, gepa_params: Dict[str, Any], output_buffer: io.StringIO) -> Any:
        """Run GEPA optimization while capturing its output."""
        # Capture GEPA's print statements and logging
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            return gepa.optimize(**gepa_params)
    
    def _log_pareto_front_info(self, gepa_logs: str) -> None:
        """Extract and log pareto front information from GEPA logs."""
        lines = gepa_logs.split('\n')
        
        for line in lines:
            # Look for pareto front information
            if 'pareto front' in line.lower() or 'new program' in line.lower():
                self.logger.info(f"🎯 GEPA: {line.strip()}")
            elif 'iteration' in line.lower() and ('score' in line.lower() or 'program' in line.lower()):
                self.logger.info(f" GEPA: {line.strip()}")
            elif 'best' in line.lower() and 'score' in line.lower():
                self.logger.info(f"🏆 GEPA: {line.strip()}")
    
    def optimize_sync(self,
                     model: str,
                     seed_prompt: str,
                     dataset: Any,
                     reflection_lm: str,
                     max_metric_calls: int = 150,
                     **kwargs) -> OptimizedResult:
        """
        Synchronous version of the optimization method
        
        Args:
            model: Target model to optimize for
            seed_prompt: Initial prompt to optimize
            dataset: Training data in any format
            reflection_lm: Model for reflection
            max_metric_calls: Budget for optimization attempts
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizedResult: Optimization result
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.train(model, seed_prompt, dataset, reflection_lm, max_metric_calls, **kwargs)
            )
            return result
        finally:
            loop.close()


# Convenience function for quick optimization
def optimize_prompt(
    model: Union[str, ModelConfig],
    seed_prompt: str,
    dataset: Any,
    reflection_model: Optional[Union[str, ModelConfig]] = None,
    **kwargs
) -> OptimizedResult:
    """
    Convenience function for quick prompt optimization without creating optimizer instance
    
    Args:
        model: Target model configuration
        seed_prompt: Initial prompt to optimize
        dataset: Training data
        reflection_model: Model for reflection (optional)
        **kwargs: Additional optimization parameters
        
    Returns:
        OptimizedResult: Optimization result
    """
    # Create default config if not provided
    if reflection_model is None:
        reflection_model = model
    
    config = OptimizationConfig(
        model=model,
        reflection_model=reflection_model,
        max_iterations=kwargs.get('max_iterations', 10),
        max_metric_calls=kwargs.get('max_metric_calls', 50),
        batch_size=kwargs.get('batch_size', 4)
    )
    
    optimizer = GepaOptimizer(config=config)
    return asyncio.run(optimizer.train(seed_prompt, dataset, **kwargs))