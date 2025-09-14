"""
Configuration models for GEPA Optimizer
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

@dataclass
class ModelConfig:
    """Configuration for any LLM provider"""
    provider: str  # Required: "openai", "anthropic", "huggingface", "vllm", etc.
    model_name: str  # Required: actual model name
    api_key: str  # Required: API key for the provider
    base_url: Optional[str] = None  # Optional: custom endpoint URL
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        """Validate required fields after initialization"""
        if not self.provider:
            raise ValueError("Provider is required (e.g., 'openai', 'anthropic', 'huggingface')")
        if not self.model_name:
            raise ValueError("Model name is required (e.g., 'gpt-4', 'claude-3-opus')")
        if not self.api_key:
            raise ValueError(f"API key is required for {self.provider} provider")
    
    @classmethod
    def from_string(cls, model_string: str) -> 'ModelConfig':
        """Create ModelConfig from string like 'openai/gpt-4' or 'gpt-4'"""
        if "/" in model_string:
            provider, model_name = model_string.split("/", 1)
        else:
            # Default to OpenAI if no provider specified
            provider = "openai"
            model_name = model_string
        
        # Get API key from environment
        api_key = cls._get_api_key_for_provider(provider)
        if not api_key:
            raise ValueError(
                f"No API key found for {provider}. Please set {provider.upper()}_API_KEY environment variable"
            )
        
        return cls(
            provider=provider,
            model_name=model_name,
            api_key=api_key
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert ModelConfig to dictionary"""
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }
    
    @staticmethod
    def _get_api_key_for_provider(provider: str) -> Optional[str]:
        """Get API key for provider from environment variables"""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "ai21": "AI21_API_KEY",
            "together": "TOGETHER_API_KEY",
            "replicate": "REPLICATE_API_TOKEN",
            "groq": "GROQ_API_KEY",
            "ollama": "OLLAMA_API_KEY"
        }
        
        env_var = env_var_map.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        
        # Fallback: try generic pattern
        return os.getenv(f"{provider.upper()}_API_KEY")

@dataclass
class OptimizationConfig:
    """Configuration class for GEPA optimization process"""
    
    # Core models - REQUIRED by user
    model: Union[str, ModelConfig]  # No default - user must specify
    reflection_model: Union[str, ModelConfig]  # No default - user must specify
    
    # Optimization parameters - REQUIRED by user
    max_iterations: int  # No default - user decides their budget
    max_metric_calls: int  # No default - user sets their budget
    batch_size: int  # No default - user decides based on memory
    
    # Optional optimization settings with sensible fallbacks
    early_stopping: bool = True
    learning_rate: float = 0.01
    
    # Multi-objective optimization
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["accuracy"])
    
    # Advanced settings
    custom_metrics: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    parallel_evaluation: bool = False
    
    # Data settings
    train_split_ratio: float = 0.8
    min_dataset_size: int = 2
    
    # Cost and budget - user controlled
    max_cost_usd: Optional[float] = None
    timeout_seconds: Optional[int] = None
    
    # GEPA-specific optimization parameters (based on actual GEPA library)
    candidate_selection_strategy: str = 'pareto'  # Use Pareto selection strategy
    skip_perfect_score: bool = False  # Don't skip perfect scores
    reflection_minibatch_size: Optional[int] = None  # Will use batch_size if None
    perfect_score: float = 1.0  # Perfect score threshold
    module_selector: str = 'round_robin'  # Component selection strategy
    verbose: bool = True  # Enable detailed GEPA logging
    
    def __post_init__(self):
        """Validate and process configuration after initialization"""
        # Convert string models to ModelConfig objects
        self.model = self._parse_model_config(self.model, "model")
        self.reflection_model = self._parse_model_config(self.reflection_model, "reflection_model")
        
        # Validate required parameters
        self._validate_required_params()
        
        # Validate ranges
        self._validate_ranges()
    
    def _parse_model_config(self, model: Union[str, ModelConfig], field_name: str) -> ModelConfig:
        """Parse string model specification into ModelConfig"""
        if isinstance(model, ModelConfig):
            return model
        
        if isinstance(model, str):
            # Parse "provider/model-name" format
            if "/" in model:
                provider, model_name = model.split("/", 1)
            else:
                # Default to openai if no provider specified
                provider = "openai"
                model_name = model
            
            # Try to get API key from environment
            api_key = self._get_api_key_for_provider(provider)
            if not api_key:
                raise ValueError(
                    f"No API key found for {provider}. Please set environment variable "
                    f"or provide ModelConfig with api_key for {field_name}"
                )
            
            return ModelConfig(
                provider=provider,
                model_name=model_name,
                api_key=api_key
            )
        
        raise ValueError(f"{field_name} must be either a string or ModelConfig object")
    
    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for provider from environment variables"""
        return ModelConfig._get_api_key_for_provider(provider)
    
    def _validate_required_params(self):
        """Validate that all required parameters are provided"""
        required_fields = {
            "max_iterations": self.max_iterations,
            "max_metric_calls": self.max_metric_calls,
            "batch_size": self.batch_size,
        }
        
        for field_name, value in required_fields.items():
            if value is None:
                raise ValueError(f"{field_name} is required and must be specified by user")
    
    def _validate_ranges(self):
        """Validate parameter ranges"""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if self.max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not (0.1 <= self.train_split_ratio <= 0.9):
            raise ValueError("train_split_ratio must be between 0.1 and 0.9")
            
        if hasattr(self.model, 'max_tokens') and self.model.max_tokens <= 0:
            raise ValueError("model.max_tokens must be a positive integer")
            
    def validate_api_connectivity(self) -> Dict[str, bool]:
        """Test API connectivity for both models"""
        results = {}
        
        for model_name, model_config in [("model", self.model), ("reflection_model", self.reflection_model)]:
            try:
                # This would be implemented to actually test the API
                # For now, just check if we have the required info
                if model_config.api_key and model_config.provider and model_config.model_name:
                    results[model_name] = True
                else:
                    results[model_name] = False
            except Exception:
                results[model_name] = False
        
        return results
    
    def get_estimated_cost(self) -> Dict[str, Any]:
        """Estimate cost based on configuration"""
        # This would calculate estimated costs based on:
        # - max_metric_calls
        # - model pricing
        # - expected tokens per call
        return {
            "max_calls": self.max_metric_calls,
            "estimated_cost_range": "To be calculated based on provider pricing",
            "cost_factors": {
                "model_calls": self.max_metric_calls,
                "reflection_calls": self.max_iterations,
                "batch_size": self.batch_size
            }
        }
    
    @classmethod
    def create_example_config(cls, provider: str = "openai") -> str:
        """Generate example configuration code for users"""
        examples = {
            "openai": '''
# Example OpenAI Configuration
config = OptimizationConfig(
    model="openai/gpt-4-turbo",  # or ModelConfig(...)
    reflection_model="openai/gpt-4-turbo",
    max_iterations=50,  # Your choice based on budget
    max_metric_calls=300,  # Your choice based on budget
    batch_size=8,  # Your choice based on memory
    early_stopping=True,
    learning_rate=0.01
)
''',
            "anthropic": '''
# Example Anthropic Configuration
config = OptimizationConfig(
    model=ModelConfig(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        api_key="your-anthropic-key",
        temperature=0.7
    ),
    reflection_model="anthropic/claude-3-sonnet-20240229",
    max_iterations=30,
    max_metric_calls=200,
    batch_size=4
)
''',
            "mixed": '''
# Example Mixed Providers Configuration
config = OptimizationConfig(
    model="openai/gpt-4-turbo",  # Main model
    reflection_model="anthropic/claude-3-opus",  # Reflection model
    max_iterations=25,
    max_metric_calls=250,
    batch_size=6,
    max_cost_usd=100.0,  # Budget limit
    timeout_seconds=3600  # 1 hour limit
)
'''
        }
        
        return examples.get(provider, examples["openai"])
