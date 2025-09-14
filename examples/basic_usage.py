"""
Basic Usage Example for GEPA Optimizer

This example shows how to use the GEPA Optimizer for UI tree prompt optimization.
"""

import asyncio
from gepa_optimizer import GepaOptimizer, OptimizationConfig, ModelConfig


async def basic_optimization_example():
    """Basic example of prompt optimization"""
    
    # Create configuration
    config = OptimizationConfig(
        model="openai/gpt-4o",  # Will use OPENAI_API_KEY from environment
        reflection_model="openai/gpt-4o",
        max_iterations=10,
        max_metric_calls=50,
        batch_size=4
    )
    
    # Create optimizer
    optimizer = GepaOptimizer(config=config)
    
    # Sample dataset (UI tree extraction)
    dataset = [
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
    
    # Optimize the prompt
    result = await optimizer.train(
        seed_prompt="You are a helpful assistant that extracts UI elements from screenshots.",
        dataset=dataset
    )
    
    # Display results
    print("🎯 Optimization Results:")
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


def hardcoded_api_keys_example():
    """Example using hardcoded API keys"""
    
    # Create configuration with hardcoded API keys
    config = OptimizationConfig(
        model=ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            api_key="your-openai-api-key-here"
        ),
        reflection_model=ModelConfig(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            api_key="your-anthropic-api-key-here"
        ),
        max_iterations=20,
        max_metric_calls=100,
        batch_size=6
    )
    
    # Create optimizer
    optimizer = GepaOptimizer(config=config)
    
    print("✅ Optimizer created with hardcoded API keys")
    return optimizer


def mixed_providers_example():
    """Example using different providers for main and reflection models"""
    
    config = OptimizationConfig(
        model="openai/gpt-4o",  # Main model
        reflection_model="anthropic/claude-3-opus",  # Reflection model
        max_iterations=15,
        max_metric_calls=75,
        batch_size=5,
        max_cost_usd=50.0,  # Budget limit
        timeout_seconds=1800  # 30 minute timeout
    )
    
    optimizer = GepaOptimizer(config=config)
    print("✅ Optimizer created with mixed providers")
    return optimizer


def gemini_optimization_example():
    """Example using Google Gemini for optimization"""
    
    # Using Gemini with environment variable GOOGLE_API_KEY
    config = OptimizationConfig(
        model="google/gemini-1.5-pro",  # Main model
        reflection_model="google/gemini-1.5-flash",  # Faster reflection model
        max_iterations=12,
        max_metric_calls=60,
        batch_size=4
    )
    
    optimizer = GepaOptimizer(config=config)
    print("✅ Optimizer created with Google Gemini")
    return optimizer


def gemini_hardcoded_example():
    """Example using Google Gemini with hardcoded API key"""
    
    config = OptimizationConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-1.5-pro",
            api_key="your-google-api-key-here"
        ),
        reflection_model=ModelConfig(
            provider="google",
            model_name="gemini-1.5-flash",
            api_key="your-google-api-key-here"
        ),
        max_iterations=10,
        max_metric_calls=50,
        batch_size=4
    )
    
    optimizer = GepaOptimizer(config=config)
    print("✅ Optimizer created with hardcoded Google API key")
    return optimizer


def all_providers_example():
    """Example showing all supported providers"""
    
    # OpenAI + Anthropic
    config1 = OptimizationConfig(
        model="openai/gpt-4o",
        reflection_model="anthropic/claude-3-opus"
    )
    
    # Google Gemini
    config2 = OptimizationConfig(
        model="google/gemini-1.5-pro",
        reflection_model="google/gemini-1.5-flash"
    )
    
    # Mixed: OpenAI + Gemini
    config3 = OptimizationConfig(
        model="openai/gpt-4o",
        reflection_model="google/gemini-1.5-flash"
    )
    
    print("✅ Configurations created for all supported providers:")
    print("  - OpenAI + Anthropic")
    print("  - Google Gemini")
    print("  - Mixed OpenAI + Gemini")
    
    return [config1, config2, config3]


if __name__ == "__main__":
    print("🚀 GEPA Optimizer Basic Usage Examples")
    print("=" * 50)
    
    # Run basic example
    print("\n1. Basic Optimization Example:")
    asyncio.run(basic_optimization_example())
    
    print("\n2. Hardcoded API Keys Example:")
    hardcoded_api_keys_example()
    
    print("\n3. Mixed Providers Example:")
    mixed_providers_example()
    
    print("\n4. Google Gemini Example:")
    gemini_optimization_example()
    
    print("\n5. Google Gemini Hardcoded Example:")
    gemini_hardcoded_example()
    
    print("\n6. All Providers Example:")
    all_providers_example()
    
    print("\n✅ All examples completed!")
