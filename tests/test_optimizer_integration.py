"""
Integration tests for GEPA Optimizer
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from gepa_optimizer import GepaOptimizer, OptimizationConfig, ModelConfig


class TestGepaOptimizerIntegration:
    """Integration tests for GepaOptimizer"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return OptimizationConfig(
            model=ModelConfig("openai", "gpt-4", "test-key"),
            reflection_model=ModelConfig("openai", "gpt-4", "test-key"),
            max_iterations=5,
            max_metric_calls=20,
            batch_size=2
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        return [
            {
                "input": "Extract UI elements from this screenshot",
                "output": "Button: Login, Text: Welcome",
                "image": "base64_encoded_image_data"
            },
            {
                "input": "Analyze this interface",
                "output": "Form: Contact, Input: Email",
                "image": "base64_encoded_image_data_2"
            }
        ]
    
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    def test_optimizer_initialization(self, mock_api_manager, mock_gepa, mock_config):
        """Test optimizer initialization"""
        mock_api_manager.return_value.get_api_key.return_value = "test-key"
        
        optimizer = GepaOptimizer(config=mock_config)
        
        assert optimizer.config == mock_config
        assert optimizer.converter is not None
        assert optimizer.api_manager is not None
        assert optimizer.result_processor is not None
        assert optimizer.custom_adapter is not None
    
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    def test_optimizer_initialization_missing_config(self, mock_api_manager, mock_gepa):
        """Test optimizer initialization with missing config"""
        with pytest.raises(ValueError, match="config parameter is required"):
            GepaOptimizer(config=None)
    
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    def test_optimizer_initialization_missing_gepa(self, mock_api_manager, mock_gepa, mock_config):
        """Test optimizer initialization with missing GEPA library"""
        mock_gepa = None
        
        with pytest.raises(Exception):  # Should raise GepaDependencyError
            GepaOptimizer(config=mock_config)
    
    @pytest.mark.asyncio
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    async def test_train_method_success(self, mock_api_manager, mock_gepa, mock_config, sample_dataset):
        """Test successful training"""
        # Mock the GEPA optimization result
        mock_result = MagicMock()
        mock_result.best_candidate = {"system_prompt": "Optimized prompt"}
        mock_result.best_score = 0.85
        mock_result.baseline_score = 0.70
        mock_result.improvement = 0.15
        mock_result.iterations = 5
        
        mock_gepa.optimize.return_value = mock_result
        mock_api_manager.return_value.get_api_key.return_value = "test-key"
        
        optimizer = GepaOptimizer(config=mock_config)
        
        result = await optimizer.train(
            seed_prompt="Extract UI elements",
            dataset=sample_dataset
        )
        
        assert result is not None
        assert result.status == "completed"
        assert result.optimization_time > 0
        assert result.dataset_size == len(sample_dataset)
    
    @pytest.mark.asyncio
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    async def test_train_method_failure(self, mock_api_manager, mock_gepa, mock_config, sample_dataset):
        """Test training failure handling"""
        # Mock GEPA to raise an exception
        mock_gepa.optimize.side_effect = Exception("GEPA optimization failed")
        mock_api_manager.return_value.get_api_key.return_value = "test-key"
        
        optimizer = GepaOptimizer(config=mock_config)
        
        result = await optimizer.train(
            seed_prompt="Extract UI elements",
            dataset=sample_dataset
        )
        
        assert result is not None
        assert result.status == "failed"
        assert result.error_message is not None
        assert "GEPA optimization failed" in result.error_message
    
    @pytest.mark.asyncio
    @patch('gepa_optimizer.core.optimizer.gepa')
    @patch('gepa_optimizer.core.optimizer.APIKeyManager')
    async def test_train_method_empty_dataset(self, mock_api_manager, mock_gepa, mock_config):
        """Test training with empty dataset"""
        mock_api_manager.return_value.get_api_key.return_value = "test-key"
        
        optimizer = GepaOptimizer(config=mock_config)
        
        result = await optimizer.train(
            seed_prompt="Extract UI elements",
            dataset=[]
        )
        
        assert result is not None
        assert result.status == "failed"
        assert "empty" in result.error_message.lower()
    
    def test_validate_inputs(self, mock_config):
        """Test input validation"""
        with patch('gepa_optimizer.core.optimizer.gepa'), \
             patch('gepa_optimizer.core.optimizer.APIKeyManager') as mock_api_manager:
            
            mock_api_manager.return_value.get_api_key.return_value = "test-key"
            optimizer = GepaOptimizer(config=mock_config)
            
            # Test valid input
            optimizer._validate_inputs("Valid prompt")
            
            # Test invalid input
            with pytest.raises(Exception):  # Should raise InvalidInputError
                optimizer._validate_inputs("")
            
            with pytest.raises(Exception):  # Should raise InvalidInputError
                optimizer._validate_inputs(None)
    
    def test_create_seed_candidate(self, mock_config):
        """Test seed candidate creation"""
        with patch('gepa_optimizer.core.optimizer.gepa'), \
             patch('gepa_optimizer.core.optimizer.APIKeyManager') as mock_api_manager:
            
            mock_api_manager.return_value.get_api_key.return_value = "test-key"
            optimizer = GepaOptimizer(config=mock_config)
            
            seed_candidate = optimizer._create_seed_candidate("Test prompt")
            
            assert isinstance(seed_candidate, dict)
            assert "system_prompt" in seed_candidate
            assert seed_candidate["system_prompt"] == "Test prompt"
