import os
import sys
import json
import base64
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Add project root to Python path first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
import gepa_optimizer
from gepa_optimizer.models import OptimizationConfig

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

@pytest.mark.asyncio
async def test_basic_functionality(mocker):
    """Test basic GEPA optimizer functionality with UI screenshots and JSON tree"""
    # Mock the necessary components
    mocker.patch('gepa.optimize', return_value='optimized_prompt')
    mocker.patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    
    print("🧪 Testing GEPA Universal Prompt Optimizer with UI screenshots...")
    
    try:
        # Get all image and JSON pairs
        screenshots_dir = Path(__file__).parent.parent / 'screenshots'
        json_dir = Path(__file__).parent.parent / 'json_tree'
        
        # Get all image and JSON files
        image_files = sorted(screenshots_dir.glob('*.jpg'))
        json_files = sorted(json_dir.glob('*.json'))
        
        # Verify we have matching pairs
        assert len(image_files) == len(json_files), "Mismatch between number of images and JSON files"
        
        # Create optimization config with GPT-4o model
        config = OptimizationConfig(
            model="openai/gpt-4o",
            reflection_model="openai/gpt-4o",
            max_iterations=5,
            max_metric_calls=10,
            batch_size=2
        )
        
        # Initialize components
        optimizer = gepa_optimizer.GepaOptimizer(config=config)
        converter = gepa_optimizer.UniversalConverter()
        
        # Process all image and JSON pairs
        data = []
        for img_path, json_path in zip(image_files, json_files):
            # Read the screenshot as base64
            with open(img_path, 'rb') as f:
                screenshot_data = f.read()
            screenshot_base64 = base64.b64encode(screenshot_data).decode('utf-8')
            
            # Read the JSON tree
            with open(json_path, 'r', encoding='utf-8') as f:
                ui_tree = json.load(f)
            
            # Add to test data in the format expected by the converter
            data.append({
                "input": {
                    "screenshot": f"data:image/jpeg;base64,{screenshot_base64}",
                    "ui_tree": ui_tree
                },
                "output": f"Expected UI interaction result for {img_path.name}"
            })
            
            print(f"✅ Processed {img_path.name} with {json_path.name}")
        
        print(f"\n📊 Total test cases: {len(data)}")
        
        # Test data conversion
        train, val = converter.convert(data)
        print(f"✅ Data conversion: {len(train)} train, {len(val)} val")
        
        # Test optimizer initialization and basic properties
        assert optimizer is not None
        
        # Test training with the first data point
        if train:
            result = await optimizer.train(
                seed_prompt="Test prompt for UI optimization",
                dataset=train[0:1]  # Just use the first item for testing
            )
            assert result is not None
            assert hasattr(result, 'status')
            print(f"✅ Training completed with status: {result.status}")
        
        print("✅ Basic functionality test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        assert False, f"Test failed with exception: {e}"
