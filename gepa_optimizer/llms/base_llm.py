"""
Base LLM client class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    
    Provides a consistent interface for different LLM providers and models.
    """
    
    def __init__(self, provider: str, model_name: str, **kwargs):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (e.g., 'openai', 'anthropic')
            model_name: Specific model name
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Store additional configuration
        self.config = kwargs
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User's input prompt
            **kwargs: Additional generation parameters (e.g., image_base64)
            
        Returns:
            Dictionary with 'content' key containing the generated response
            and additional metadata
        """
        pass
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information for logging and debugging"""
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'class': self.__class__.__name__
        }
