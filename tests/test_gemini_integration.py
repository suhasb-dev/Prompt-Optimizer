"""
Tests for Google Gemini API integration
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from gepa_optimizer.llms.vision_llm import VisionLLMClient, ProviderType, GepaLLMError, ErrorType
from gepa_optimizer.utils.api_keys import APIKeyManager


class TestGeminiIntegration:
    """Test Google Gemini API integration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_api_key = "test-google-api-key"
        self.test_model = "gemini-1.5-pro"
        
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_client_initialization(self, mock_get_key):
        """Test Gemini client initialization"""
        mock_get_key.return_value = self.test_api_key
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        assert client.provider == ProviderType.GOOGLE
        assert client.model_name == self.test_model
        assert client.api_key == self.test_api_key

    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_alias_provider(self, mock_get_key):
        """Test that GEMINI provider works as alias for GOOGLE"""
        mock_get_key.return_value = self.test_api_key
        
        client = VisionLLMClient(
            provider=ProviderType.GEMINI,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        assert client.provider == ProviderType.GEMINI

    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_client_without_api_key_raises_error(self, mock_get_key):
        """Test that missing API key raises appropriate error"""
        mock_get_key.return_value = None
        
        with pytest.raises(GepaLLMError) as exc_info:
            VisionLLMClient(
                provider=ProviderType.GOOGLE,
                model_name=self.test_model
            )
        
        assert exc_info.value.error_type == ErrorType.VALIDATION_ERROR
        assert "No API key found" in str(exc_info.value)

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_generate_text_only(self, mock_get_key, mock_genai):
        """Test Gemini text-only generation"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "This is a test response from Gemini"
        mock_response.prompt_feedback = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        result = client.generate(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, how are you?"
        )
        
        assert result["content"] == "This is a test response from Gemini"
        assert result["provider"] == "google"
        assert result["model"] == "gemini-1.5-pro"
        assert result["role"] == "assistant"

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_generate_with_image(self, mock_get_key, mock_genai):
        """Test Gemini generation with image"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "I can see an image with a cat"
        mock_response.prompt_feedback = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        # Mock base64 image
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        result = client.generate(
            system_prompt="You are a helpful assistant",
            user_prompt="What do you see in this image?",
            image_base64=test_image_b64
        )
        
        assert result["content"] == "I can see an image with a cat"
        assert result["provider"] == "google"

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_model_mapping(self, mock_get_key, mock_genai):
        """Test that different model names map correctly"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.prompt_feedback = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test different model name mappings
        test_cases = [
            ("gemini-pro", "gemini-1.5-pro"),
            ("gemini-flash", "gemini-1.5-flash"),
            ("gemini-1.0-pro", "gemini-1.0-pro"),
            ("unknown-model", "gemini-1.5-pro")  # Default fallback
        ]
        
        for input_model, expected_model in test_cases:
            client = VisionLLMClient(
                provider=ProviderType.GOOGLE,
                model_name=input_model,
                api_key=self.test_api_key
            )
            
            result = client.generate(
                system_prompt="Test",
                user_prompt="Test"
            )
            
            assert result["model"] == expected_model

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_safety_blocking(self, mock_get_key, mock_genai):
        """Test handling of safety-blocked responses"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock Gemini response with safety blocking
        mock_response = Mock()
        mock_response.text = None
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = Mock()
        mock_response.candidates[0].finish_reason.name = "SAFETY"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        with pytest.raises(GepaLLMError) as exc_info:
            client.generate(
                system_prompt="Test",
                user_prompt="Test"
            )
        
        assert exc_info.value.error_type == ErrorType.VALIDATION_ERROR
        assert "blocked" in str(exc_info.value).lower()

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_json_response_parsing(self, mock_get_key, mock_genai):
        """Test that JSON responses are parsed correctly"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock JSON response
        json_response = {"elements": [{"type": "button", "text": "Click me"}]}
        mock_response = Mock()
        mock_response.text = json.dumps(json_response)
        mock_response.prompt_feedback = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        result = client.generate(
            system_prompt="Return JSON",
            user_prompt="Generate UI elements"
        )
        
        assert result == json_response

    @patch('gepa_optimizer.llms.vision_llm.genai')
    @patch('gepa_optimizer.utils.api_keys.APIKeyManager.get_api_key')
    def test_gemini_usage_metadata_logging(self, mock_get_key, mock_genai):
        """Test that usage metadata is logged correctly"""
        mock_get_key.return_value = self.test_api_key
        
        # Mock response with usage metadata
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.prompt_feedback = None
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = None
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = VisionLLMClient(
            provider=ProviderType.GOOGLE,
            model_name=self.test_model,
            api_key=self.test_api_key
        )
        
        with patch('gepa_optimizer.llms.vision_llm.logger') as mock_logger:
            client.generate(
                system_prompt="Test",
                user_prompt="Test"
            )
            
            # Check that usage was logged
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args[0][0]
            assert "Tokens used" in log_call
            assert "10" in log_call  # prompt tokens
            assert "5" in log_call   # completion tokens
            assert "15" in log_call  # total tokens

    def test_gemini_missing_dependencies(self):
        """Test error handling when Google Generative AI is not installed"""
        with patch('gepa_optimizer.llms.vision_llm.genai', side_effect=ImportError("No module named 'google.generativeai'")):
            client = VisionLLMClient(
                provider=ProviderType.GOOGLE,
                model_name=self.test_model,
                api_key=self.test_api_key
            )
            
            with pytest.raises(GepaLLMError) as exc_info:
                client.generate(
                    system_prompt="Test",
                    user_prompt="Test"
                )
            
            assert exc_info.value.error_type == ErrorType.VALIDATION_ERROR
            assert "Required dependencies" in str(exc_info.value)
            assert "google-generativeai" in str(exc_info.value)


class TestGeminiAPIKeyManager:
    """Test Google/Gemini API key management"""

    def test_google_api_key_loading(self):
        """Test loading Google API key from environment"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            manager = APIKeyManager()
            assert manager.get_api_key('google') == 'test-key'
            assert manager.get_api_key('gemini') == 'test-key'  # Alias

    def test_google_api_key_setting(self):
        """Test setting Google API key at runtime"""
        manager = APIKeyManager()
        manager.set_api_key('google', 'test-key')
        assert manager.get_api_key('google') == 'test-key'
        assert manager.get_api_key('gemini') == 'test-key'  # Alias

    def test_google_legacy_methods(self):
        """Test legacy methods for Google API key management"""
        manager = APIKeyManager()
        
        # Test set methods
        manager.set_google_key('test-google-key')
        assert manager.get_google_key() == 'test-google-key'
        
        manager.set_gemini_key('test-gemini-key')
        assert manager.get_gemini_key() == 'test-gemini-key'
        assert manager.get_google_key() == 'test-gemini-key'  # Should be the same

    def test_google_key_validation(self):
        """Test Google API key validation"""
        manager = APIKeyManager()
        
        # Test without key
        assert not manager.has_key('google')
        assert not manager.has_key('gemini')
        
        # Test with key
        manager.set_api_key('google', 'test-key')
        assert manager.has_key('google')
        assert manager.has_key('gemini')
        
        # Test missing keys list
        missing = manager.get_missing_keys(['openai', 'google', 'anthropic'])
        assert 'google' not in missing  # We have it
        assert 'openai' in missing or 'anthropic' in missing  # We don't have these

    def test_google_key_validation_dict(self):
        """Test Google API key validation dictionary"""
        manager = APIKeyManager()
        manager.set_api_key('google', 'test-key')
        
        validation = manager.validate_keys(['openai', 'google', 'gemini'])
        assert validation['google'] is True
        assert validation['gemini'] is True
        assert validation['openai'] is False
