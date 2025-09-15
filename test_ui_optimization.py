#!/usr/bin/env python3
"""
UI Tree Optimization Test Script

This script tests the complete UI tree optimization pipeline using:
- Screenshots from screenshots/ directory
- JSON tree structures from json_tree/ directory
- GEPA optimization with vision LLM models
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

# Enable debug logging to see evaluation details
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('gepa_optimizer.evaluation.ui_evaluator').setLevel(logging.DEBUG)
logging.getLogger('gepa_optimizer.core.custom_adapter').setLevel(logging.DEBUG)

from gepa_optimizer import GepaOptimizer, OptimizationConfig, ModelConfig

async def test_ui_optimization():
    """Test UI tree optimization with BOTH images and JSON files"""
    
    print("🚀 Starting UI Tree Optimization Test")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Check if directories exist
    json_dir = Path("json_tree")
    screenshots_dir = Path("screenshots")
    
    if not json_dir.exists():
        print(f"❌ JSON directory not found: {json_dir}")
        return False
        
    if not screenshots_dir.exists():
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        return False
    
    # Count available files
    json_files = list(json_dir.glob("*.json"))
    image_files = list(screenshots_dir.glob("*.jpg")) + list(screenshots_dir.glob("*.jpeg")) + list(screenshots_dir.glob("*.png"))
    
    print(f"📁 Found {len(json_files)} JSON files in {json_dir}")
    print(f"📁 Found {len(image_files)} image files in {screenshots_dir}")
    
    if len(json_files) == 0 or len(image_files) == 0:
        print("❌ No data files found. Please ensure you have both JSON and image files.")
        return False
    
    # Create configuration
    print("\n🔧 Setting up configuration...")
    config = OptimizationConfig(
        model=ModelConfig(
            provider="openai",
            model_name="gpt-4o-mini",  # Best for vision tasks
            api_key=os.getenv('OPENAI_API_KEY')
        ),
        reflection_model=ModelConfig(
            provider="openai", 
            model_name="gpt-4o-mini",
            api_key=os.getenv('OPENAI_API_KEY')
        ),
        max_iterations=10,  # Reduced to avoid timeouts
        max_metric_calls=20,  # Reduced to avoid timeouts
        batch_size=3,
        early_stopping=False,
         # Reduced to avoid timeouts
        
        # GEPA-specific parameters for better optimization
        candidate_selection_strategy='pareto',  # Use Pareto selection
        skip_perfect_score=False,  # Don't skip perfect scores
        reflection_minibatch_size=3,  # Use 2 examples for reflection
        perfect_score=1.0,  # Perfect score threshold
        module_selector='round_robin',  # Cycle through components
        verbose=True
    )
    
    # Create optimizer
    print("🤖 Initializing GEPA Optimizer...")
    optimizer = GepaOptimizer(config=config)
    
    # Seed prompt (system prompt)
    seed_prompt = """You are a helpful assistant that extracts UI elements from screenshots and provides complete UI tree structures in JSON format. 

Your task is to:
1. Analyze the provided screenshot carefully
2. Identify all UI elements (buttons, text, images, containers, etc.)
3. Extract their properties (text content, positioning, styling, hierarchy)
4. Return a complete JSON tree structure that represents the UI layout

"""
    
    # Dataset configuration
    dataset_config = {
        'json_dir': str(json_dir),
        'screenshots_dir': str(screenshots_dir),
        'type': 'ui_tree_dataset'
    }
    
    print(f"📝 Seed prompt: {seed_prompt[:100]}...")
    print(f"📊 Dataset: {len(json_files)} JSON files + {len(image_files)} images")
    
    try:
        print("\n🚀 Starting GEPA optimization...")
        print("This may take a few minutes...")
        
        result = await optimizer.train(
            seed_prompt=seed_prompt,
            dataset=dataset_config
        )
        
        print("\n✅ Optimization completed successfully!")
        print("=" * 50)
        print(f"📈 Original prompt length: {len(result.original_prompt)} characters")
        print(f"🎯 Optimized prompt length: {len(result.prompt)} characters")
        print(f"⏱️  Optimization time: {result.optimization_time:.2f} seconds")
        print(f"📊 Dataset size: {result.dataset_size} samples")
        
        if hasattr(result, 'improvement_data') and result.improvement_data:
            print(f"📈 Improvement data: {result.improvement_data}")
        
        print("\n" + "="*80)
        print("📝 ORIGINAL SEED PROMPT:")
        print("="*80)
        print(result.original_prompt)
        print("="*80)
        
        print("\n" + "="*80)
        print("🎯 OPTIMIZED PROMPT:")
        print("="*80)
        print(result.prompt)
        print("="*80)
        
        # Calculate and show improvement metrics
        if result.original_prompt != result.prompt:
            print(f"\n PROMPT COMPARISON:")
            print(f"   Length change: {len(result.prompt) - len(result.original_prompt):+d} characters")
            print(f"   Length change: {((len(result.prompt) - len(result.original_prompt)) / len(result.original_prompt) * 100):+.1f}%")
        else:
            print(f"\n⚠️  No changes detected - optimized prompt is identical to original")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test"""
    print("UI Tree Optimization Test")
    print("=" * 50)
    
    try:
        success = asyncio.run(test_ui_optimization())
        
        if success:
            print("\n🎉 Test completed successfully!")
            print("Your UI tree optimization pipeline is working correctly.")
        else:
            print("\n💥 Test failed!")
            print("Please check the error messages above and fix any issues.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
