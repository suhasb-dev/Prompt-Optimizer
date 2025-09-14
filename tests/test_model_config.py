"""
Tests for Model Configuration
"""

import pytest
from unittest.mock import patch
from gepa_optimizer.models.config import ModelConfig, OptimizationConfig


class TestModelConfig:
    """Test ModelConfig functionality"""
    
    def test_model_config_creation(self):
        """Test basic ModelConfig creation"""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7  # default value
    
    def test_model_config_validation(self):
        """Test ModelConfig validation"""
        # Test missing provider
        with pytest.raises(ValueError, match="Provider is required"):
            ModelConfig(provider="", model_name="gpt-4", api_key="test-key")
        
        # Test missing model name
        with pytest.raises(ValueError, match="Model name is required"):
            ModelConfig(provider="openai", model_name="", api_key="test-key")
        
        # Test missing API key
        with pytest.raises(ValueError, match="API key is required"):
            ModelConfig(provider="openai", model_name="gpt-4", api_key="")
    
    @patch('gepa_optimizer.models.config.os.getenv')
    def test_from_string_with_provider(self, mock_getenv):
        """Test ModelConfig.from_string with provider/model format"""
        mock_getenv.return_value = "test-key"
        
        config = ModelConfig.from_string("openai/gpt-4")
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
    
    @patch('gepa_optimizer.models.config.os.getenv')
    def test_from_string_without_provider(self, mock_getenv):
        """Test ModelConfig.from_string without provider (defaults to openai)"""
        mock_getenv.return_value = "test-key"
        
        config = ModelConfig.from_string("gpt-4")
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
    
    @patch('gepa_optimizer.models.config.os.getenv')
    def test_from_string_missing_key(self, mock_getenv):
        """Test ModelConfig.from_string with missing API key"""
        mock_getenv.return_value = None
        
        with pytest.raises(ValueError, match="No API key found"):
            ModelConfig.from_string("openai/gpt-4")
    
    def test_from_dict(self):
        """Test ModelConfig.from_dict"""
        config_dict = {
            "provider": "anthropic",
            "model_name": "claude-3-opus",
            "api_key": "test-key",
            "temperature": 0.5
        }
        
        config = ModelConfig.from_dict(config_dict)
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-opus"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
    
    def test_to_dict(self):
        """Test ModelConfig.to_dict"""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.8
        )
        
        config_dict = config.to_dict()
        assert config_dict["provider"] == "openai"
        assert config_dict["model_name"] == "gpt-4"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["temperature"] == 0.8


class TestOptimizationConfig:
    """Test OptimizationConfig functionality"""
    
    @patch('gepa_optimizer.models.config.os.getenv')
    def test_optimization_config_creation(self, mock_getenv):
        """Test basic OptimizationConfig creation"""
        mock_getenv.return_value = "test-key"
        
        config = OptimizationConfig(
            model="openai/gpt-4",
            reflection_model="openai/gpt-4",
            max_iterations=50,
            max_metric_calls=300,
            batch_size=8
        )
        
        assert config.model.provider == "openai"
        assert config.model.model_name == "gpt-4"
        assert config.reflection_model.provider == "openai"
        assert config.max_iterations == 50
        assert config.max_metric_calls == 300
        assert config.batch_size == 8
    
    def test_optimization_config_validation(self):
        """Test OptimizationConfig validation"""
        # Test missing max_iterations
        with pytest.raises(ValueError, match="max_iterations is required"):
            OptimizationConfig(
                model=ModelConfig("openai", "gpt-4", "key"),
                reflection_model=ModelConfig("openai", "gpt-4", "key"),
                max_iterations=None,
                max_metric_calls=300,
                batch_size=8
            )
        
        # Test invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            OptimizationConfig(
                model=ModelConfig("openai", "gpt-4", "key"),
                reflection_model=ModelConfig("openai", "gpt-4", "key"),
                max_iterations=0,
                max_metric_calls=300,
                batch_size=8
            )
    
    def test_validate_api_connectivity(self):
        """Test API connectivity validation"""
        config = OptimizationConfig(
            model=ModelConfig("openai", "gpt-4", "test-key"),
            reflection_model=ModelConfig("anthropic", "claude-3", "test-key"),
            max_iterations=10,
            max_metric_calls=100,
            batch_size=4
        )
        
        connectivity = config.validate_api_connectivity()
        assert connectivity["model"] is True
        assert connectivity["reflection_model"] is True
    
    def test_get_estimated_cost(self):
        """Test cost estimation"""
        config = OptimizationConfig(
            model=ModelConfig("openai", "gpt-4", "test-key"),
            reflection_model=ModelConfig("openai", "gpt-4", "test-key"),
            max_iterations=10,
            max_metric_calls=100,
            batch_size=4
        )
        
        cost_estimate = config.get_estimated_cost()
        assert cost_estimate["max_calls"] == 100
        assert "cost_factors" in cost_estimate
