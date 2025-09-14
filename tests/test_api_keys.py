"""
Tests for API Key Management
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from gepa_optimizer.utils.api_keys import APIKeyManager


class TestAPIKeyManager:
    """Test API key management functionality"""
    
    def test_init_loads_from_env(self):
        """Test that APIKeyManager loads keys from environment"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key'
        }):
            manager = APIKeyManager()
            assert manager.get_api_key('openai') == 'test-openai-key'
            assert manager.get_api_key('anthropic') == 'test-anthropic-key'
    
    def test_set_get_api_key(self):
        """Test setting and getting API keys"""
        manager = APIKeyManager()
        manager.set_api_key('openai', 'test-key')
        assert manager.get_api_key('openai') == 'test-key'
    
    def test_has_key(self):
        """Test key existence checking"""
        manager = APIKeyManager()
        assert not manager.has_key('openai')
        
        manager.set_api_key('openai', 'test-key')
        assert manager.has_key('openai')
    
    def test_get_missing_keys(self):
        """Test getting list of missing keys"""
        manager = APIKeyManager()
        manager.set_api_key('openai', 'test-key')
        
        missing = manager.get_missing_keys(['openai', 'anthropic'])
        assert 'anthropic' in missing
        assert 'openai' not in missing
    
    def test_validate_keys(self):
        """Test key validation"""
        manager = APIKeyManager()
        manager.set_api_key('openai', 'test-key')
        
        validation = manager.validate_keys(['openai', 'anthropic'])
        assert validation['openai'] is True
        assert validation['anthropic'] is False
    
    def test_legacy_methods(self):
        """Test backward compatibility methods"""
        manager = APIKeyManager()
        
        # Test set_openai_key
        manager.set_openai_key('legacy-key')
        assert manager.get_openai_key() == 'legacy-key'
        
        # Test get_openai_key raises error when missing
        manager._keys.clear()
        with pytest.raises(RuntimeError, match="OpenAI API key missing"):
            manager.get_openai_key()
    
    def test_case_insensitive_provider(self):
        """Test that provider names are case insensitive"""
        manager = APIKeyManager()
        manager.set_api_key('OPENAI', 'test-key')
        assert manager.get_api_key('openai') == 'test-key'
        assert manager.get_api_key('OpenAI') == 'test-key'
