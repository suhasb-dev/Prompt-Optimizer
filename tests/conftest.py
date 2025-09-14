"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'HUGGINGFACE_API_KEY': 'test-hf-key'
    }):
        yield


@pytest.fixture
def sample_ui_dataset():
    """Sample UI tree dataset for testing"""
    return [
        {
            "input": "Extract UI elements from this screenshot",
            "output": "Button: Login, Text: Welcome to our app",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "button",
                "text": "Login",
                "bounds": [100, 200, 150, 230]
            }
        },
        {
            "input": "Analyze this interface layout",
            "output": "Form: Contact, Input: Email, Button: Submit",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "form",
                "children": [
                    {"type": "input", "text": "Email"},
                    {"type": "button", "text": "Submit"}
                ]
            }
        }
    ]


@pytest.fixture
def mock_gepa_result():
    """Mock GEPA optimization result"""
    result = MagicMock()
    result.best_candidate = {
        "system_prompt": "You are an expert UI element extractor. Analyze the screenshot and provide detailed information about all visible UI elements including their type, text content, and spatial relationships."
    }
    result.best_score = 0.85
    result.baseline_score = 0.70
    result.improvement = 0.15
    result.iterations = 5
    result.optimization_history = [
        {"score": 0.70, "candidate": {"system_prompt": "Original prompt"}},
        {"score": 0.75, "candidate": {"system_prompt": "Improved prompt"}},
        {"score": 0.80, "candidate": {"system_prompt": "Better prompt"}},
        {"score": 0.85, "candidate": {"system_prompt": "Best prompt"}}
    ]
    return result


@pytest.fixture
def mock_vision_llm_response():
    """Mock vision LLM response"""
    return {
        "content": '{"elements": [{"type": "button", "text": "Login", "bounds": [100, 200, 150, 230]}]}',
        "role": "assistant",
        "model": "gpt-4o",
        "provider": "openai"
    }


@pytest.fixture
def mock_api_key_manager():
    """Mock API key manager"""
    manager = MagicMock()
    manager.get_api_key.return_value = "test-api-key"
    manager.has_key.return_value = True
    manager.get_missing_keys.return_value = []
    manager.validate_keys.return_value = {"openai": True, "anthropic": True}
    return manager
