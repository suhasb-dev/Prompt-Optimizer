"""
Vision LLM Client for GEPA Optimizer
"""

import json
import logging
import time
from enum import Enum
import requests
from typing import Dict, Optional, Any, TYPE_CHECKING, Union

# Assuming APIKeyManager is available from utils
from ..utils.api_keys import APIKeyManager

# Import ModelConfig only for type checking to avoid circular imports
if TYPE_CHECKING:
    from ..models.config import ModelConfig

class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    GOOGLE = "google"
    GEMINI = "gemini"

class ErrorType(str, Enum):
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"

class GepaLLMError(Exception):
    """Base exception for GEPA LLM related errors"""
    def __init__(self, message: str, error_type: ErrorType, status_code: Optional[int] = None):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self):
        if self.status_code:
            return f"{self.error_type.value} (HTTP {self.status_code}): {self.message}"
        return f"{self.error_type.value}: {self.message}"

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

class VisionLLMClient:
    """
    A client for interacting with multi-modal Vision LLMs (e.g., OpenAI GPT-4 Vision).
    
    Example:
        ```python
        # Basic usage
        client = VisionLLMClient(
            provider="openai",
            model_name="gpt-4-vision-preview",
            temperature=0.7,
            max_tokens=2048
        )
        
        # With custom configuration
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4-vision-preview",
            temperature=0.5,
            max_tokens=1024
        )
        client = VisionLLMClient.from_config(config)
        ```
    """

    def __init__(
        self,
        provider: Union[str, ProviderType],
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout: int = 120,  # Increase to 2 minutes for large prompts
        max_retries: int = 3
    ):
        """
        Initializes the VisionLLMClient with model configuration.

        Args:
            provider: The provider of the model (e.g., 'openai', 'anthropic')
            model_name: The name of the multi-modal LLM model to use (e.g., "gpt-4-vision-preview").
            api_key: Optional API key. If not provided, it will be fetched from APIKeyManager.
            base_url: Optional base URL for the API endpoint.
            temperature: Controls randomness in the response generation.
            max_tokens: Maximum number of tokens to generate.
            top_p: Controls diversity via nucleus sampling.
            frequency_penalty: Penalizes repeated tokens.
            presence_penalty: Penalizes new tokens based on their presence in the text so far.
        """
        # Input validation
        try:
            self.provider = ProviderType(provider.lower())
        except ValueError:
            raise GepaLLMError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {[p.value for p in ProviderType]}",
                ErrorType.VALIDATION_ERROR
            )
            
        if not model_name:
            raise GepaLLMError("model_name cannot be empty", ErrorType.VALIDATION_ERROR)
            
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise GepaLLMError(
                f"temperature must be between 0 and 2, got {temperature}",
                ErrorType.VALIDATION_ERROR
            )
            
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise GepaLLMError(
                f"max_tokens must be a positive integer, got {max_tokens}",
                ErrorType.VALIDATION_ERROR
            )
            
        # Initialize API key
        try:
            self.api_key = api_key or APIKeyManager().get_api_key(self.provider.value)
            if not self.api_key:
                raise GepaLLMError(
                    f"No API key found for provider: {self.provider}",
                    ErrorType.VALIDATION_ERROR
                )
        except Exception as e:
            raise GepaLLMError(
                f"Failed to initialize API key: {str(e)}",
                ErrorType.API_ERROR
            ) from e
            
        self.model_name = model_name
        self.base_url = base_url or OPENAI_API_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configure session with retry
        self.session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Basic check for model compatibility (can be expanded)
        if self.provider == 'openai' and "gpt-4-vision" not in self.model_name.lower():
            logger.warning(f"Model {self.model_name} might not be a vision model.")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key based on provider"""
        if self.provider == 'openai':
            return APIKeyManager().get_api_key('openai')
        elif self.provider == 'anthropic':
            return APIKeyManager().get_api_key('anthropic')
        elif self.provider in ['google', 'gemini']:
            return APIKeyManager().get_api_key('google')
        # Add other providers as needed
        return None

    @classmethod
    def from_config(cls, config: 'ModelConfig') -> 'VisionLLMClient':
        """Create a VisionLLMClient from a ModelConfig object.
        
        Args:
            config: ModelConfig instance with provider and model settings
            
        Returns:
            Configured VisionLLMClient instance
            
        Example:
            ```python
            config = ModelConfig(
                provider="openai",
                model_name="gpt-4-vision-preview",
                temperature=0.7
            )
            client = VisionLLMClient.from_config(config)
            ```
        """
        return cls(
            provider=config.provider,
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )
        
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        image_base64: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generates a response from the Vision LLM.

        Args:
            system_prompt: The system-level instructions for the LLM.
            user_prompt: The user's query or task.
            image_base64: Optional Base64 encoded image string.
            **generation_kwargs: Additional model-specific generation parameters

        Returns:
            A dictionary containing the generated response and metadata.
            
        Raises:
            GepaLLMError: If there's an error during generation
            
        Example:
            ```python
            response = client.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="What's in this image?",
                image_base64="base64_encoded_image"
            )
            ```
        """
        if not system_prompt or not user_prompt:
            raise GepaLLMError(
                "system_prompt and user_prompt are required",
                ErrorType.VALIDATION_ERROR
            )
            
        try:
            if self.provider == ProviderType.OPENAI:
                return self._generate_openai(system_prompt, user_prompt, image_base64, **generation_kwargs)
            elif self.provider in [ProviderType.GOOGLE, ProviderType.GEMINI]:
                return self._generate_google(system_prompt, user_prompt, image_base64, **generation_kwargs)
            else:
                raise GepaLLMError(
                    f"Provider {self.provider} is not yet supported",
                    ErrorType.VALIDATION_ERROR
                )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during generation: {str(e)}")
            raise GepaLLMError(
                f"Network error: {str(e)}",
                ErrorType.NETWORK_ERROR,
                getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            ) from e
        except GepaLLMError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during generation: {str(e)}")
            raise GepaLLMError(
                f"Generation failed: {str(e)}",
                ErrorType.API_ERROR
            ) from e

    def _generate_openai(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        image_base64: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI's API with configured parameters.
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: User's input prompt
            image_base64: Optional base64 encoded image
            
        Returns:
            Dictionary containing the API response
            
        Raises:
            GepaDependencyError: If API call fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "GepaOptimizer/1.0 (Python)"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        if image_base64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            # "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        
        self.logger.debug(f"Sending request to {self.base_url} with model {self.model_name}")
        
        try:
            self.logger.debug(f"Sending request to {self.model_name}")
            
            # Make the API request with retry
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=300
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                self.logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self._generate_openai(system_prompt, user_prompt, image_base64, **generation_kwargs)
                
            response.raise_for_status()
            
            result = response.json()
            self.logger.debug(f"Received response from {self.model_name}")
            
            # Extract and validate the response
            try:
                message = result["choices"][0]["message"]
                llm_response_content = message["content"]
                
                # Log token usage if available
                if "usage" in result:
                    usage = result["usage"]
                    self.logger.info(
                        f"Tokens used - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                        f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                        f"Total: {usage.get('total_tokens', 'N/A')}"
                    )
                
                # Try to parse JSON if the response looks like JSON
                if isinstance(llm_response_content, str) and (
                    llm_response_content.startswith('{') or 
                    llm_response_content.startswith('[')
                ):
                    try:
                        return json.loads(llm_response_content)
                    except json.JSONDecodeError:
                        pass
                
                # Default response format
                return {
                    "content": llm_response_content,
                    "role": message.get("role", "assistant"),
                    "model": self.model_name,
                    "provider": self.provider.value
                }
                
            except (KeyError, IndexError) as e:
                self.logger.error(f"Unexpected response format: {result}")
                raise GepaLLMError(
                    f"Unexpected response format from {self.provider} API",
                    ErrorType.API_ERROR,
                    response.status_code
                ) from e
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            error_msg = f"HTTP error {status_code} from {self.provider} API"
            
            try:
                error_data = e.response.json()
                error_msg = error_data.get('error', {}).get('message', error_msg)
            except Exception:
                error_msg = str(e)
                
            self.logger.error(f"{error_msg}: {error_data if 'error_data' in locals() else str(e)}")
            raise GepaLLMError(
                error_msg,
                ErrorType.RATE_LIMIT if status_code == 429 else ErrorType.API_ERROR,
                status_code
            ) from e
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Request to {self.provider} API timed out after {self.timeout} seconds")
            raise GepaLLMError(
                f"Request timed out after {self.timeout} seconds",
                ErrorType.TIMEOUT
            )
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error: {str(e)}")
            raise GepaLLMError(
                f"Network error: {str(e)}",
                ErrorType.NETWORK_ERROR
            ) from e
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise GepaLLMError(
                f"Unexpected error: {str(e)}",
                ErrorType.API_ERROR
            ) from e

    def _generate_google(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        image_base64: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Google Gemini API with configured parameters.
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: User's input prompt
            image_base64: Optional base64 encoded image
            
        Returns:
            Dictionary containing the API response
            
        Raises:
            GepaLLMError: If API call fails
        """
        try:
            import google.generativeai as genai
            import base64
            from PIL import Image
            import io
        except ImportError as e:
            raise GepaLLMError(
                f"Required dependencies for Google Gemini not installed: {str(e)}. "
                f"Please install: pip install google-generativeai Pillow",
                ErrorType.VALIDATION_ERROR
            ) from e
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Map model names to Gemini models
        model_mapping = {
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-pro': 'gemini-1.5-pro',
            'gemini-flash': 'gemini-1.5-flash',
            'gemini-1.0-pro': 'gemini-1.0-pro'
        }
        
        gemini_model_name = model_mapping.get(self.model_name.lower(), 'gemini-1.5-pro')
        
        try:
            model = genai.GenerativeModel(gemini_model_name)
        except Exception as e:
            raise GepaLLMError(
                f"Failed to initialize Gemini model {gemini_model_name}: {str(e)}",
                ErrorType.API_ERROR
            ) from e
        
        # Prepare content
        content_parts = []
        
        # Add system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        content_parts.append(full_prompt)
        
        # Add image if provided
        if image_base64:
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                content_parts.append(image)
                self.logger.debug(f"Added image to Gemini request")
            except Exception as e:
                self.logger.warning(f"Failed to process image for Gemini: {str(e)}")
                # Continue without image rather than failing
        
        self.logger.debug(f"Sending request to Gemini model {gemini_model_name}")
        
        try:
            # Generate response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Configure generation parameters
                    generation_config = genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        top_p=self.top_p,
                    )
                    
                    response = model.generate_content(
                        content_parts,
                        generation_config=generation_config
                    )
                    
                    # Check if response was blocked
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        raise GepaLLMError(
                            f"Gemini blocked the prompt: {response.prompt_feedback.block_reason}",
                            ErrorType.VALIDATION_ERROR
                        )
                    
                    # Check if response was blocked
                    if not response.text:
                        if response.candidates and response.candidates[0].finish_reason:
                            finish_reason = response.candidates[0].finish_reason
                            if finish_reason == genai.types.FinishReason.SAFETY:
                                raise GepaLLMError(
                                    "Gemini response blocked due to safety concerns",
                                    ErrorType.VALIDATION_ERROR
                                )
                            elif finish_reason == genai.types.FinishReason.RECITATION:
                                raise GepaLLMError(
                                    "Gemini response blocked due to recitation concerns",
                                    ErrorType.VALIDATION_ERROR
                                )
                        raise GepaLLMError(
                            "Gemini returned empty response",
                            ErrorType.API_ERROR
                        )
                    
                    self.logger.debug(f"Received response from Gemini model {gemini_model_name}")
                    
                    # Log usage information if available
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        usage = response.usage_metadata
                        self.logger.info(
                            f"Tokens used - Prompt: {usage.prompt_token_count}, "
                            f"Completion: {usage.candidates_token_count}, "
                            f"Total: {usage.total_token_count}"
                        )
                    
                    # Try to parse JSON if the response looks like JSON
                    response_text = response.text
                    if isinstance(response_text, str) and (
                        response_text.startswith('{') or 
                        response_text.startswith('[')
                    ):
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError:
                            pass
                    
                    # Default response format
                    return {
                        "content": response_text,
                        "role": "assistant",
                        "model": gemini_model_name,
                        "provider": "google"
                    }
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise
                        
        except GepaLLMError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error with Gemini API: {str(e)}")
            raise GepaLLMError(
                f"Gemini API error: {str(e)}",
                ErrorType.API_ERROR
            ) from e
