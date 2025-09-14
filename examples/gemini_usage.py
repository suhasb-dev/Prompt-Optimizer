"""
Google Gemini Usage Examples for GEPA Optimizer

This example demonstrates how to use Google Gemini models with the GEPA Optimizer
for UI tree prompt optimization.
"""

import asyncio
import os
from gepa_optimizer import GepaOptimizer, OptimizationConfig, ModelConfig, APIKeyManager


async def gemini_basic_example():
    """Basic example using Google Gemini for prompt optimization"""
    
    print("🔮 Google Gemini Basic Example")
    print("-" * 40)
    
    # Method 1: Using environment variable GOOGLE_API_KEY
    config = OptimizationConfig(
        model="google/gemini-1.5-pro",  # Main model
        reflection_model="google/gemini-1.5-flash",  # Faster reflection model
        max_iterations=8,
        max_metric_calls=40,
        batch_size=3
    )
    
    # Create optimizer
    optimizer = GepaOptimizer(config=config)
    
    # Sample UI tree dataset
    dataset = [
        {
            "input": "Extract UI elements from this mobile app screenshot",
            "output": "Button: Sign In, Text: Welcome Back, Input: Email",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "screen",
                "children": [
                    {"type": "text", "text": "Welcome Back"},
                    {"type": "input", "placeholder": "Email"},
                    {"type": "button", "text": "Sign In"}
                ]
            }
        },
        {
            "input": "Analyze this e-commerce product page",
            "output": "Image: Product Photo, Text: Product Name, Button: Add to Cart",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "container",
                "children": [
                    {"type": "image", "src": "product.jpg"},
                    {"type": "text", "text": "Amazing Product"},
                    {"type": "button", "text": "Add to Cart", "action": "purchase"}
                ]
            }
        }
    ]
    
    # Optimize the prompt
    print("🚀 Starting optimization with Gemini...")
    result = await optimizer.train(
        seed_prompt="You are an AI assistant that extracts UI elements from screenshots. Analyze the image and identify all interactive elements, text labels, and layout components.",
        dataset=dataset
    )
    
    # Display results
    print("\n🎯 Optimization Results:")
    print(f"Status: {result.status}")
    print(f"Time: {result.optimization_time:.2f}s")
    print(f"Dataset Size: {result.dataset_size}")
    
    if result.is_successful:
        print(f"\n📝 Original Prompt:")
        print(result.original_prompt)
        print(f"\n✨ Optimized Prompt:")
        print(result.prompt)
        
        if 'improvement_percent' in result.improvement_data:
            print(f"\n📈 Improvement: {result.improvement_data['improvement_percent']:.2f}%")
    else:
        print(f"\n❌ Optimization failed: {result.error_message}")


def gemini_api_key_management():
    """Example showing different ways to manage Google API keys"""
    
    print("\n🔑 Google API Key Management Examples")
    print("-" * 40)
    
    # Method 1: Environment variable
    print("1. Using Environment Variable:")
    print("   export GOOGLE_API_KEY='your-api-key-here'")
    
    # Method 2: .env file
    print("\n2. Using .env file:")
    print("   Create .env file with: GOOGLE_API_KEY=your-api-key-here")
    
    # Method 3: Runtime setting
    print("\n3. Setting API key at runtime:")
    api_manager = APIKeyManager()
    api_manager.set_google_key("your-api-key-here")
    print("   ✅ API key set at runtime")
    
    # Method 4: Direct configuration
    print("\n4. Direct configuration in ModelConfig:")
    config = OptimizationConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-1.5-pro",
            api_key="your-api-key-here"
        ),
        reflection_model=ModelConfig(
            provider="google",
            model_name="gemini-1.5-flash",
            api_key="your-api-key-here"
        )
    )
    print("   ✅ Configuration created with direct API key")


def gemini_model_variants():
    """Example showing different Gemini model variants"""
    
    print("\n🤖 Gemini Model Variants")
    print("-" * 40)
    
    models = [
        ("gemini-1.5-pro", "Most capable model, best for complex tasks"),
        ("gemini-1.5-flash", "Faster model, good for simple tasks"),
        ("gemini-1.0-pro", "Previous generation, still capable"),
        ("gemini-pro", "Alias for gemini-1.5-pro"),
        ("gemini-flash", "Alias for gemini-1.5-flash")
    ]
    
    for model_name, description in models:
        print(f"• {model_name}: {description}")
    
    # Example configurations for different use cases
    print("\n📋 Recommended Configurations:")
    
    # High quality optimization
    high_quality_config = OptimizationConfig(
        model="google/gemini-1.5-pro",
        reflection_model="google/gemini-1.5-pro",
        max_iterations=15,
        max_metric_calls=100,
        batch_size=4
    )
    print("• High Quality: gemini-1.5-pro for both main and reflection")
    
    # Fast optimization
    fast_config = OptimizationConfig(
        model="google/gemini-1.5-flash",
        reflection_model="google/gemini-1.5-flash",
        max_iterations=8,
        max_metric_calls=50,
        batch_size=6
    )
    print("• Fast: gemini-1.5-flash for both main and reflection")
    
    # Balanced optimization
    balanced_config = OptimizationConfig(
        model="google/gemini-1.5-pro",
        reflection_model="google/gemini-1.5-flash",
        max_iterations=12,
        max_metric_calls=75,
        batch_size=4
    )
    print("• Balanced: gemini-1.5-pro for main, gemini-1.5-flash for reflection")


def gemini_vision_capabilities():
    """Example showing Gemini's vision capabilities for UI analysis"""
    
    print("\n👁️ Gemini Vision Capabilities for UI Analysis")
    print("-" * 40)
    
    # Example of how Gemini can analyze UI screenshots
    vision_example = {
        "system_prompt": "You are a UI analysis expert. Analyze the provided screenshot and extract all UI elements with their properties.",
        "user_prompt": "Extract all interactive elements, text labels, and layout components from this mobile app interface.",
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
        "expected_output": {
            "elements": [
                {
                    "type": "button",
                    "text": "Login",
                    "bounds": [100, 200, 150, 230],
                    "color": "#007AFF",
                    "action": "navigate_to_login"
                },
                {
                    "type": "text",
                    "content": "Welcome to our app",
                    "bounds": [50, 100, 300, 130],
                    "font_size": 18,
                    "color": "#000000"
                },
                {
                    "type": "input",
                    "placeholder": "Enter your email",
                    "bounds": [50, 150, 300, 180],
                    "input_type": "email"
                }
            ],
            "layout": {
                "type": "vertical_stack",
                "spacing": 20,
                "alignment": "center"
            }
        }
    }
    
    print("✅ Gemini can analyze:")
    print("  • UI element types (buttons, inputs, text, images)")
    print("  • Element properties (text, colors, sizes)")
    print("  • Spatial relationships and layout")
    print("  • Interactive behaviors and actions")
    print("  • Accessibility features")
    
    print("\n🎯 Perfect for UI tree extraction tasks!")


def gemini_error_handling():
    """Example showing error handling with Gemini API"""
    
    print("\n⚠️ Gemini Error Handling")
    print("-" * 40)
    
    print("Common error scenarios and how to handle them:")
    
    print("\n1. Missing API Key:")
    print("   Error: 'No API key found for provider: google'")
    print("   Solution: Set GOOGLE_API_KEY environment variable")
    
    print("\n2. Safety Blocking:")
    print("   Error: 'Gemini blocked the prompt: SAFETY'")
    print("   Solution: Modify prompt to avoid safety concerns")
    
    print("\n3. Rate Limiting:")
    print("   Error: 'Rate limit exceeded'")
    print("   Solution: Implement exponential backoff (built-in)")
    
    print("\n4. Invalid Model:")
    print("   Error: 'Failed to initialize Gemini model'")
    print("   Solution: Use supported model names")
    
    print("\n5. Network Issues:")
    print("   Error: 'Network error'")
    print("   Solution: Check internet connection, retry with backoff")


def gemini_best_practices():
    """Best practices for using Gemini with GEPA Optimizer"""
    
    print("\n💡 Best Practices for Gemini + GEPA")
    print("-" * 40)
    
    print("1. Model Selection:")
    print("   • Use gemini-1.5-pro for complex UI analysis")
    print("   • Use gemini-1.5-flash for faster iterations")
    print("   • Mix models: pro for main, flash for reflection")
    
    print("\n2. Prompt Engineering:")
    print("   • Be specific about UI element types")
    print("   • Include examples in your seed prompt")
    print("   • Use structured output formats (JSON)")
    
    print("\n3. Dataset Quality:")
    print("   • Include diverse UI patterns")
    print("   • Provide clear ground truth annotations")
    print("   • Balance complexity levels")
    
    print("\n4. Configuration:")
    print("   • Start with smaller batch sizes (3-4)")
    print("   • Use moderate iteration counts (8-12)")
    print("   • Monitor token usage and costs")
    
    print("\n5. Error Handling:")
    print("   • Always check for safety blocks")
    print("   • Implement proper retry logic")
    print("   • Log detailed error information")


async def main():
    """Run all Gemini examples"""
    
    print("🚀 Google Gemini + GEPA Optimizer Examples")
    print("=" * 60)
    
    # Check if API key is available
    api_manager = APIKeyManager()
    if not api_manager.has_key('google'):
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment")
        print("   Set it with: export GOOGLE_API_KEY='your-key'")
        print("   Or create a .env file with: GOOGLE_API_KEY=your-key")
        print("\n   Continuing with examples that don't require API calls...\n")
    else:
        print("✅ Google API key found!")
        print("   Running full examples...\n")
    
    # Run examples
    try:
        if api_manager.has_key('google'):
            await gemini_basic_example()
    except Exception as e:
        print(f"❌ Basic example failed: {e}")
        print("   This is expected if you don't have a valid API key")
    
    gemini_api_key_management()
    gemini_model_variants()
    gemini_vision_capabilities()
    gemini_error_handling()
    gemini_best_practices()
    
    print("\n🎉 All Gemini examples completed!")
    print("\nNext steps:")
    print("1. Get a Google API key from: https://makersuite.google.com/app/apikey")
    print("2. Set it as environment variable: export GOOGLE_API_KEY='your-key'")
    print("3. Run: python examples/gemini_usage.py")


if __name__ == "__main__":
    asyncio.run(main())
