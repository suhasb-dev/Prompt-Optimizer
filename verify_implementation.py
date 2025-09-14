#!/usr/bin/env python3
"""
Comprehensive verification script to test all components of the GEPA optimizer.
This script verifies that all the fixes are working correctly.
"""

import sys
import json
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_ui_evaluator():
    """Test the UI evaluator with sample data"""
    print("🧪 Testing UI Evaluator...")
    
    try:
        from gepa_optimizer.evaluation.ui_evaluator import UITreeEvaluator
        
        # Load sample JSON
        with open("json_tree/2.json", 'r') as f:
            expected_json = json.load(f)
        
        # Create a partial prediction
        predicted_json = {
            "id": "test_screen",
            "type": "Screen",
            "children": [
                {
                    "id": "test_button",
                    "type": "Button",
                    "text": "Test Button"
                }
            ]
        }
        
        evaluator = UITreeEvaluator()
        results = evaluator.evaluate(predicted_json, expected_json)
        
        print(f"✅ UI Evaluator working - Composite score: {results['composite_score']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ UI Evaluator failed: {e}")
        return False

def test_custom_adapter():
    """Test the custom adapter"""
    print("🧪 Testing Custom Adapter...")
    
    try:
        from gepa_optimizer.core.custom_adapter import CustomGepaAdapter
        from gepa_optimizer.models.config import ModelConfig
        
        # Create a mock model config
        model_config = ModelConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test-key"
        )
        
        adapter = CustomGepaAdapter(model_config)
        
        # Test JSON parsing
        test_json = '{"test": "value"}'
        parsed = adapter._parse_json_safely(test_json)
        
        if parsed and parsed.get("test") == "value":
            print("✅ Custom Adapter working - JSON parsing successful")
            return True
        else:
            print("❌ Custom Adapter failed - JSON parsing issue")
            return False
            
    except Exception as e:
        print(f"❌ Custom Adapter failed: {e}")
        return False

def test_optimizer_config():
    """Test the optimizer configuration"""
    print("🧪 Testing Optimizer Configuration...")
    
    try:
        from gepa_optimizer.models.config import OptimizationConfig, ModelConfig
        
        # Create test config
        config = OptimizationConfig(
            model="gpt-4o-mini",
            reflection_model="gpt-4o-mini",
            max_iterations=5,
            max_metric_calls=15,
            batch_size=3
        )
        
        print("✅ Optimizer Configuration working - Config created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Optimizer Configuration failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing Data Loading...")
    
    try:
        from gepa_optimizer.data.loaders import DataLoader
        
        loader = DataLoader()
        
        # Test UI tree dataset loading
        dataset = loader.load_ui_tree_dataset("json_tree", "screenshots")
        
        if dataset and len(dataset) > 0:
            print(f"✅ Data Loading working - Loaded {len(dataset)} samples")
            return True
        else:
            print("❌ Data Loading failed - No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data Loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 GEPA Optimizer Implementation Verification")
    print("=" * 50)
    
    tests = [
        test_ui_evaluator,
        test_custom_adapter,
        test_optimizer_config,
        test_data_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 VERIFICATION RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation is working correctly.")
        return True
    else:
        print(f"\n💥 {total - passed} TESTS FAILED! Please fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
