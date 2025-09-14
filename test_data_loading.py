#!/usr/bin/env python3
"""
Test script to verify UI tree dataset loading functionality
without requiring API keys or GEPA optimization
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gepa_optimizer.data.loaders import DataLoader
from gepa_optimizer.data.converters import UniversalConverter

def test_data_loading():
    """Test the UI tree dataset loading functionality"""
    
    print("🧪 Testing UI Tree Dataset Loading")
    print("=" * 50)
    
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
    
    try:
        # Test DataLoader
        print("\n🔧 Testing DataLoader...")
        loader = DataLoader()
        
        # Test loading UI tree dataset
        print("📊 Loading UI tree dataset...")
        dataset = loader.load_ui_tree_dataset(str(json_dir), str(screenshots_dir))
        
        if not dataset:
            print("❌ No dataset loaded")
            return False
        
        print(f"✅ Successfully loaded {len(dataset)} image-JSON pairs")
        
        # Show sample data
        if dataset:
            sample = dataset[0]
            print(f"\n📋 Sample dataset entry:")
            print(f"  - Input: {sample['input'][:50]}...")
            print(f"  - Output length: {len(sample['output'])} characters")
            print(f"  - Image (base64): {len(sample['image'])} characters")
        
        # Test UniversalConverter
        print("\n🔄 Testing UniversalConverter...")
        converter = UniversalConverter()
        
        # Test conversion
        dataset_config = {
            'json_dir': str(json_dir),
            'screenshots_dir': str(screenshots_dir),
            'type': 'ui_tree_dataset'
        }
        
        print("🔄 Converting dataset to GEPA format...")
        train_data, val_data = converter.convert(dataset_config)
        
        print(f"✅ Conversion successful!")
        print(f"  - Training samples: {len(train_data)}")
        print(f"  - Validation samples: {len(val_data)}")
        
        # Show sample converted data
        if train_data:
            sample = train_data[0]
            print(f"\n📋 Sample converted entry:")
            print(f"  - Input: {sample['input'][:50]}...")
            print(f"  - Output: {sample['output'][:100]}...")
            print(f"  - Image: {len(sample['image'])} characters (base64)")
        
        print("\n🎉 All tests passed!")
        print("✅ UI tree dataset loading is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test"""
    try:
        success = test_data_loading()
        
        if success:
            print("\n🚀 Ready for GEPA optimization!")
            print("Next steps:")
            print("1. Set your OPENAI_API_KEY environment variable")
            print("2. Run: python test_ui_optimization.py")
        else:
            print("\n💥 Data loading test failed!")
            print("Please check the error messages above and fix any issues.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
