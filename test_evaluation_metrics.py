#!/usr/bin/env python3
"""
Test script to verify the new UI tree evaluation metrics work correctly.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gepa_optimizer.evaluation.ui_evaluator import UITreeEvaluator
import json

def test_evaluation_metrics():
    """Test the new evaluation metrics with sample data"""
    
    print("🧪 Testing UI Tree Evaluation Metrics")
    print("=" * 50)
    
    # Load a sample JSON file
    json_file = Path("json_tree/2.json")
    if not json_file.exists():
        print("❌ Sample JSON file not found")
        return False
    
    with open(json_file, 'r') as f:
        expected_json = json.load(f)
    
    print(f"📄 Loaded expected JSON: {len(str(expected_json))} characters")
    print(f"📊 Expected elements: {count_elements(expected_json)}")
    
    # Create a partial prediction (missing some elements)
    predicted_json = create_partial_prediction(expected_json)
    
    print(f"📄 Created partial prediction: {len(str(predicted_json))} characters")
    print(f"📊 Predicted elements: {count_elements(predicted_json)}")
    
    # Test evaluation
    evaluator = UITreeEvaluator()
    results = evaluator.evaluate(predicted_json, expected_json)
    
    print("\n📈 Evaluation Results:")
    print("-" * 30)
    for metric, score in results.items():
        print(f"{metric:25}: {score:.4f}")
    
    # Verify scores are reasonable
    composite_score = results.get("composite_score", 0.0)
    if 0.0 <= composite_score <= 1.0:
        print(f"\n✅ Composite score is valid: {composite_score:.4f}")
        return True
    else:
        print(f"\n❌ Composite score is invalid: {composite_score}")
        return False

def count_elements(node):
    """Count elements in a JSON tree"""
    if not isinstance(node, dict):
        return 0
    count = 1
    for child in node.get("children", []):
        count += count_elements(child)
    return count

def create_partial_prediction(expected_json):
    """Create a partial prediction by removing some elements"""
    import copy
    predicted = copy.deepcopy(expected_json)
    
    # Remove some children from the auth_buttons_section
    if "children" in predicted:
        for child in predicted["children"]:
            if child.get("id") == "auth_buttons_section" and "children" in child:
                # Keep only the first button
                if len(child["children"]) > 1:
                    child["children"] = child["children"][:1]
                break
    
    return predicted

if __name__ == "__main__":
    success = test_evaluation_metrics()
    if success:
        print("\n🎉 Evaluation metrics test passed!")
    else:
        print("\n💥 Evaluation metrics test failed!")
        sys.exit(1)
