"""
API Key Management for GEPA Optimizer
"""

import os
from dotenv import load_dotenv
from typing import Optional, Dict, List

class APIKeyManager:
    """Handles API keys securely without hardcoding"""
    
    def __init__(self):
        # Load .env file if present
        load_dotenv()
        self._keys: Dict[str, str] = {}
        self._load_from_env()

    def _load_from_env(self):
        """Load API keys from environment variables"""
        env_mappings = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'ai21': 'AI21_API_KEY',
            'together': 'TOGETHER_API_KEY',
            'replicate': 'REPLICATE_API_TOKEN',
            'groq': 'GROQ_API_KEY',
            'ollama': 'OLLAMA_API_KEY',
            'google': 'GOOGLE_API_KEY',  # ← ADD THIS
            'gemini': 'GOOGLE_API_KEY' 
        }
        
        for provider, env_var in env_mappings.items():
            key = os.getenv(env_var)
            if key:
                self._keys[provider] = key

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return self._keys.get(provider.lower())
    
    def set_api_key(self, provider: str, key: str):
        """Set API key for a provider at runtime"""
        provider_lower = provider.lower()
        self._keys[provider_lower] = key
    
    # Handle aliases - if setting google, also set gemini and vice versa
        if provider_lower == 'google':
            self._keys['gemini'] = key
        elif provider_lower == 'gemini':
            self._keys['google'] = key
    
    def has_key(self, provider: str) -> bool:
        """Check if API key exists for provider"""
        return provider.lower() in self._keys
    
    def get_missing_keys(self, providers: List[str]) -> List[str]:
        """Get list of providers missing API keys"""
        return [p for p in providers if not self.has_key(p)]
    
    def validate_keys(self, providers: List[str]) -> Dict[str, bool]:
        """Validate API keys for multiple providers"""
        return {provider: self.has_key(provider) for provider in providers}

    # Legacy methods for backward compatibility
    def set_openai_key(self, key: str):
        """Set OpenAI API key at runtime"""
        self.set_api_key('openai', key)

    def set_anthropic_key(self, key: str):
        """Set Anthropic API key at runtime"""
        self.set_api_key('anthropic', key)

    def set_google_key(self, key: str):
        """Set Google API key at runtime"""
        self.set_api_key('google', key)

    def set_gemini_key(self, key: str):
        """Set Gemini API key at runtime (alias for Google)"""
        self.set_api_key('google', key)

    def get_openai_key(self) -> str:
        """Get OpenAI key or raise error if missing"""
        key = self.get_api_key('openai')
        if not key:
            raise RuntimeError(
                "OpenAI API key missing. Set via:\n"
                "1. Environment variable: OPENAI_API_KEY=your_key\n"
                "2. .env file: OPENAI_API_KEY=your_key\n"
                "3. Code: api_manager.set_api_key('openai', 'your_key')"
            )
        return key

    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic key (optional)"""
        return self.get_api_key('anthropic')

    def get_google_key(self) -> Optional[str]:
        """Get Google key (optional)"""
        return self.get_api_key('google')

    def get_gemini_key(self) -> Optional[str]:
        """Get Gemini key (alias for Google)"""
        return self.get_api_key('google')

    def has_required_keys(self) -> bool:
        """Check if required keys are available"""
        return bool(self.get_api_key('openai'))
