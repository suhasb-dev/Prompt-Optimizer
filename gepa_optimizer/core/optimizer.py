"""
Main GepaOptimizer class - the heart of the optimization system
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union
import asyncio

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
            # 'temperature': getattr(model_config, 'temperature', 'default'),
            # 'max_tokens': getattr(model_config, 'max_tokens', 'default')
        }
        self.logger.info(f"Initialized model with config: {model_info}")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Validate GEPA availability
        if gepa is None:
            raise GepaDependencyError("GEPA library is not available. Install with: pip install gepa")
            
        # Log configuration summary
        model_name = (self.config.model.model_name 
                     if isinstance(self.config.model, ModelConfig) 
                     else str(self.config.model))
        provider = (self.config.model.provider 
                   if isinstance(self.config.model, ModelConfig) 
                   else 'unknown')
        
        self.logger.info(f"Initialized GepaOptimizer with model: {model_name} "
                        f"(provider: {provider})")
                        
        if self.config.reflection_model:
            ref_model_name = (self.config.reflection_model.model_name 
                            if isinstance(self.config.reflection_model, ModelConfig)
                            else str(self.config.reflection_model))
            ref_provider = (self.config.reflection_model.provider 
                          if isinstance(self.config.reflection_model, ModelConfig)
                          else 'unknown')
            self.logger.info(f"Reflection model: {ref_model_name} "
                           f"(provider: {ref_provider})")
    
    async def train(self,
                   seed_prompt: str,
                   dataset: Union[List[Any], str, Dict, Any],
                   **kwargs) -> OptimizedResult:
        """
        Main training method for prompt optimization
        
        Args:
            seed_prompt: Initial prompt to optimize
            dataset: Training data in ANY format
            **kwargs: Additional optimization parameters
                - max_metric_calls: Override for config.max_metric_calls
                - batch_size: Override for config.batch_size
                - Any other parameters to override in config
                
        Returns:
            OptimizedResult: Optimization result with optimized prompt
            
        Example:
            >>> config = OptimizationConfig(
            ...     model="openai/gpt-4-turbo",
            ...     reflection_model="anthropic/claude-3-opus",
            ...     max_iterations=50,
            ...     max_metric_calls=300,
            ...     batch_size=8
            ... )
            >>> optimizer = GepaOptimizer(config=config)
            >>> result = await optimizer.train(
            ...     seed_prompt="You are a helpful assistant",
            ...     dataset=[{"input": "Hi", "output": "Hello!"}]
            ... )
            >>> print(result.prompt)
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
            result = OptimizationResult(
                session_id=session_id,
                original_prompt=seed_prompt,
                optimized_prompt=processed_result['optimized_prompt'],
                improvement_data=processed_result['metrics'],
                optimization_time=optimization_time,
                dataset_size=len(trainset),
                status="completed"
            )
            
            print(f"✅ Optimization completed in {optimization_time:.2f}s")
            
            # Log improvement if available
            metrics = processed_result['metrics']
            if 'improvement_percent' in metrics:
                improvement = metrics['improvement_percent']
                print(f"📈 Performance improvement: {improvement:.2f}%")
            
            return OptimizedResult(result)

        except (GepaOptimizerError, Exception) as e:
            # Handle optimization failure gracefully
            self.logger.error(f"Optimization failed: {str(e)}")
            print(f"❌ Optimization failed: {str(e)}")
            
            # Return failed result with original prompt
            optimization_time = time.time() - start_time
            failed_result = OptimizationResult(
                session_id=session_id,
                original_prompt=seed_prompt,
                optimized_prompt=seed_prompt,  # Return original on failure
                improvement_data={'error': str(e)},
                optimization_time=optimization_time,
                dataset_size=0,
                status="failed",
                error_message=str(e)
            )
            
            return OptimizedResult(failed_result)
    
    def _update_config_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Update config with values from kwargs"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config.{key} = {value}")
            else:
                self.logger.warning(f"Ignoring unknown config parameter: {key}")
    
    def _validate_inputs(self, seed_prompt: str) -> None:
        """
        Validate input parameters
        
        Args:
            seed_prompt: The initial prompt to validate
            
        Raises:
            InvalidInputError: If any input is invalid
        """
        if not seed_prompt or not isinstance(seed_prompt, str):
            raise InvalidInputError("Seed prompt must be a non-empty string")
            
        # Validate model configuration
        if not hasattr(self.config, 'model') or not self.config.model:
            raise InvalidInputError("Model configuration is required in config")
            
        # Validate reflection model if specified
        if hasattr(self.config, 'reflection_model') and self.config.reflection_model:
            if not isinstance(self.config.reflection_model, ModelConfig):
                raise InvalidInputError("Reflection model must be a ModelConfig instance")
        
        # Log configuration summary
        self.logger.info("Optimization configuration:")
        self.logger.info(f"- Model: {self.config.model.model_name} (provider: {self.config.model.provider})")
        if hasattr(self.config, 'reflection_model') and self.config.reflection_model:
            self.logger.info(f"- Reflection model: {self.config.reflection_model.model_name} (provider: {self.config.reflection_model.provider}")
        self.logger.info(f"- Max iterations: {self.config.max_iterations}")
        self.logger.info(f"- Max metric calls: {self.config.max_metric_calls}")
        self.logger.info(f"- Batch size: {self.config.batch_size}")
    
    def _create_seed_candidate(self, seed_prompt: str) -> Dict[str, str]:
        """Create GEPA-compatible seed candidate"""
        sanitized_prompt = sanitize_prompt(seed_prompt)
        return {
            'system_prompt': sanitized_prompt
        }
    
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
            
            # Run GEPA optimization (synchronous call wrapped in async)
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: gepa.optimize(**gepa_params)
            )
            return result
        except Exception as e:
            raise GepaOptimizerError(f"GEPA optimization failed: {str(e)}")
    
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
    Convenience function for quick prompt optimization
    
    Args:
        model: Target model to optimize for (can be string or ModelConfig)
        seed_prompt: Initial prompt to optimize
        dataset: Training data in any format
        reflection_model: Optional model for reflection (can be string or ModelConfig)
        **kwargs: Additional optimization parameters
            - max_iterations: Maximum number of optimization iterations
            - max_metric_calls: Maximum number of metric evaluations
            - batch_size: Batch size for evaluation
            - Any other OptimizationConfig parameters
            
    Returns:
        OptimizedResult: Optimized prompt result
        
    Example:
        >>> result = optimize_prompt(
        ...     model="openai/gpt-4-turbo",
        ...     seed_prompt="You are a helpful assistant",
        ...     dataset=[{"input": "Hi", "output": "Hello!"}],
        ...     reflection_model="anthropic/claude-3-opus",
        ...     max_iterations=50
        ... )
    """
    # Create model configurations if strings are provided
    if isinstance(model, str):
        model = ModelConfig(model_name=model)
    if isinstance(reflection_model, str):
        reflection_model = ModelConfig(model_name=reflection_model)
    
    # Create configuration with provided parameters
    config = OptimizationConfig(
        model=model,
        reflection_model=reflection_model,
        **{k: v for k, v in kwargs.items() 
           if hasattr(OptimizationConfig, k) or k in ['max_iterations', 'max_metric_calls', 'batch_size']}
    )
    
    # Filter out config parameters from kwargs
    train_kwargs = {k: v for k, v in kwargs.items() 
                   if not (hasattr(OptimizationConfig, k) or k in ['max_iterations', 'max_metric_calls', 'batch_size'])}
    
    # Create optimizer and run training
    optimizer = GepaOptimizer(config=config)
    return optimizer.train(
        seed_prompt=seed_prompt,
        dataset=dataset,
        **train_kwargs
    )


def optimize_prompt_sync(
    model: Union[str, ModelConfig],
    seed_prompt: str,
    dataset: Any,
    reflection_model: Optional[Union[str, ModelConfig]] = None,
    **kwargs
) -> OptimizedResult:
    """
    Synchronous convenience function for quick prompt optimization
    
    Args:
        model: Target model to optimize for (can be string or ModelConfig)
        seed_prompt: Initial prompt to optimize
        dataset: Training data in any format
        reflection_model: Optional model for reflection (can be string or ModelConfig)
        **kwargs: Additional optimization parameters
            - max_iterations: Maximum number of optimization iterations
            - max_metric_calls: Maximum number of metric evaluations
            - batch_size: Batch size for evaluation
            - Any other OptimizationConfig parameters
            
    Returns:
        OptimizedResult: Optimized prompt result
        
    Example:
        >>> result = optimize_prompt_sync(
        ...     model="openai/gpt-4-turbo",
        ...     seed_prompt="You are a helpful assistant",
        ...     dataset=[{"input": "Hi", "output": "Hello!"}],
        ...     reflection_model="anthropic/claude-3-opus"
        ... )
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        optimize_prompt(model, seed_prompt, dataset, reflection_model, **kwargs)
    )