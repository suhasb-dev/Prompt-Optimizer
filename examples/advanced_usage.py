"""
Advanced Usage Examples for GEPA Optimizer

This example shows advanced features and configurations.
"""

import asyncio
import json
from pathlib import Path
from gepa_optimizer import (
    GepaOptimizer, OptimizationConfig, ModelConfig,
    APIKeyManager, GepaOptimizerError
)


async def advanced_optimization_example():
    """Advanced optimization with custom settings"""
    
    # Advanced configuration
    config = OptimizationConfig(
        model=ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            api_key="your-openai-key",  # Replace with actual key
            temperature=0.7,
            max_tokens=2048
        ),
        reflection_model=ModelConfig(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            api_key="your-anthropic-key",  # Replace with actual key
            temperature=0.5,
            max_tokens=1024
        ),
        max_iterations=50,
        max_metric_calls=300,
        batch_size=8,
        early_stopping=True,
        learning_rate=0.02,
        multi_objective=True,
        objectives=["accuracy", "relevance", "clarity"],
        train_split_ratio=0.85,
        use_cache=True,
        parallel_evaluation=True,
        max_cost_usd=100.0,
        timeout_seconds=3600
    )
    
    # Validate configuration
    print("🔍 Validating configuration...")
    connectivity = config.validate_api_connectivity()
    print(f"API Connectivity: {connectivity}")
    
    cost_estimate = config.get_estimated_cost()
    print(f"Cost Estimate: {cost_estimate}")
    
    # Create optimizer
    optimizer = GepaOptimizer(config=config)
    
    # Load dataset from file (if exists)
    dataset_path = Path("data/ui_dataset.json")
    if dataset_path.exists():
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"📁 Loaded dataset from {dataset_path}")
    else:
        # Use sample dataset
        dataset = create_sample_dataset()
        print("📝 Using sample dataset")
    
    try:
        # Optimize with progress tracking
        print("🚀 Starting advanced optimization...")
        result = await optimizer.train(
            seed_prompt="You are an expert UI element extractor. Analyze screenshots and provide detailed, structured information about all visible UI elements.",
            dataset=dataset
        )
        
        # Detailed results analysis
        print("\n📊 Detailed Results:")
        print(f"Session ID: {result.session_id}")
        print(f"Status: {result.status}")
        print(f"Optimization Time: {result.optimization_time:.2f}s")
        print(f"Dataset Size: {result.dataset_size}")
        print(f"Total Iterations: {result.total_iterations}")
        
        if result.is_successful:
            print(f"\n📝 Prompt Comparison:")
            print(f"Original Length: {len(result.original_prompt)} chars")
            print(f"Optimized Length: {len(result.prompt)} chars")
            
            # Show improvement metrics
            if result.improvement_data:
                print(f"\n📈 Improvement Metrics:")
                for key, value in result.improvement_data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
            
            # Save results
            save_results(result)
        else:
            print(f"\n❌ Optimization failed: {result.error_message}")
            
    except GepaOptimizerError as e:
        print(f"❌ GEPA Optimizer Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def create_sample_dataset():
    """Create a comprehensive sample dataset"""
    return [
        {
            "input": "Extract all UI elements from this login screen",
            "output": "Button: Login (x:100, y:200, w:150, h:30), Input: Username (x:100, y:150, w:200, h:25), Input: Password (x:100, y:180, w:200, h:25)",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "container",
                "children": [
                    {"type": "input", "text": "Username", "bounds": [100, 150, 300, 175]},
                    {"type": "input", "text": "Password", "bounds": [100, 180, 300, 205]},
                    {"type": "button", "text": "Login", "bounds": [100, 200, 250, 230]}
                ]
            }
        },
        {
            "input": "Analyze this dashboard interface",
            "output": "Header: Dashboard (x:0, y:0, w:800, h:60), Sidebar: Navigation (x:0, y:60, w:200, h:600), Main: Content (x:200, y:60, w:600, h:600)",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "container",
                "children": [
                    {"type": "header", "text": "Dashboard", "bounds": [0, 0, 800, 60]},
                    {"type": "sidebar", "text": "Navigation", "bounds": [0, 60, 200, 660]},
                    {"type": "main", "text": "Content", "bounds": [200, 60, 800, 660]}
                ]
            }
        },
        {
            "input": "Identify form elements in this contact page",
            "output": "Input: Name (x:50, y:100, w:300, h:30), Input: Email (x:50, y:140, w:300, h:30), Textarea: Message (x:50, y:180, w:300, h:100), Button: Submit (x:50, y:300, w:100, h:35)",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
            "ui_tree": {
                "type": "form",
                "children": [
                    {"type": "input", "text": "Name", "bounds": [50, 100, 350, 130]},
                    {"type": "input", "text": "Email", "bounds": [50, 140, 350, 170]},
                    {"type": "textarea", "text": "Message", "bounds": [50, 180, 350, 280]},
                    {"type": "button", "text": "Submit", "bounds": [50, 300, 150, 335]}
                ]
            }
        }
    ]


def save_results(result):
    """Save optimization results to files"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_data = {
        "session_id": result.session_id,
        "status": result.status,
        "optimization_time": result.optimization_time,
        "dataset_size": result.dataset_size,
        "total_iterations": result.total_iterations,
        "original_prompt": result.original_prompt,
        "optimized_prompt": result.prompt,
        "improvement_data": result.improvement_data,
        "error_message": result.error_message
    }
    
    results_file = results_dir / f"optimization_{result.session_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Save just the optimized prompt
    prompt_file = results_dir / f"optimized_prompt_{result.session_id}.txt"
    with open(prompt_file, 'w') as f:
        f.write(result.prompt)
    
    print(f"📝 Optimized prompt saved to: {prompt_file}")


def api_key_management_example():
    """Example of API key management"""
    
    # Initialize API key manager
    api_manager = APIKeyManager()
    
    # Check available keys
    print("🔑 API Key Status:")
    providers = ["openai", "anthropic", "huggingface"]
    validation = api_manager.validate_keys(providers)
    
    for provider, has_key in validation.items():
        status = "✅" if has_key else "❌"
        print(f"  {status} {provider.upper()}: {'Available' if has_key else 'Missing'}")
    
    # Set keys programmatically (if needed)
    missing_keys = api_manager.get_missing_keys(providers)
    if missing_keys:
        print(f"\n⚠️  Missing keys for: {', '.join(missing_keys)}")
        print("Set them via environment variables or use api_manager.set_api_key()")
    
    return api_manager


def configuration_examples():
    """Show different configuration examples"""
    
    print("⚙️  Configuration Examples:")
    print("=" * 40)
    
    # Example 1: Simple configuration
    print("\n1. Simple Configuration:")
    simple_config = OptimizationConfig.create_example_config("openai")
    print(simple_config)
    
    # Example 2: Anthropic configuration
    print("\n2. Anthropic Configuration:")
    anthropic_config = OptimizationConfig.create_example_config("anthropic")
    print(anthropic_config)
    
    # Example 3: Mixed providers
    print("\n3. Mixed Providers Configuration:")
    mixed_config = OptimizationConfig.create_example_config("mixed")
    print(mixed_config)


async def main():
    """Main function to run all examples"""
    print("🚀 GEPA Optimizer Advanced Examples")
    print("=" * 50)
    
    # API Key Management
    print("\n1. API Key Management:")
    api_key_management_example()
    
    # Configuration Examples
    print("\n2. Configuration Examples:")
    configuration_examples()
    
    # Advanced Optimization
    print("\n3. Advanced Optimization:")
    await advanced_optimization_example()
    
    print("\n✅ All advanced examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
