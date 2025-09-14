"""
Command Line Interface for GEPA Optimizer
"""

import argparse
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

from .core import GepaOptimizer
from .models import OptimizationConfig, ModelConfig
from .utils import setup_logging, APIKeyManager


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GEPA Universal Prompt Optimizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gepa-optimize --model openai/gpt-4-turbo --prompt "Extract UI elements" --dataset data.json
  gepa-optimize --config config.json --prompt "Analyze interface" --dataset images/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt", 
        required=True,
        help="Initial seed prompt to optimize"
    )
    parser.add_argument(
        "--dataset",
        required=True, 
        help="Path to dataset file or directory"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        help="Model specification (e.g., 'openai/gpt-4-turbo')"
    )
    parser.add_argument(
        "--reflection-model",
        help="Reflection model specification"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )
    
    # Optimization parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations (default: 10)"
    )
    parser.add_argument(
        "--max-metric-calls", 
        type=int,
        default=100,
        help="Maximum metric evaluation calls (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)"
    )
    
    # GEPA-specific parameters
    parser.add_argument(
        "--candidate-selection-strategy",
        type=str,
        default="pareto",
        choices=["pareto", "best"],
        help="Strategy for selecting candidates (default: pareto)"
    )
    parser.add_argument(
        "--skip-perfect-score",
        action="store_true",
        help="Skip updating candidates with perfect scores"
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=None,
        help="Number of examples to use for reflection (default: use batch_size)"
    )
    parser.add_argument(
        "--perfect-score",
        type=float,
        default=1.0,
        help="Perfect score threshold (default: 1.0)"
    )
    parser.add_argument(
        "--module-selector",
        type=str,
        default="round_robin",
        choices=["round_robin", "all"],
        help="Component selection strategy (default: round_robin)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        help="Output file path for results (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    try:
        # Load configuration
        if args.config:
            config = load_config_from_file(args.config)
        else:
            config = create_config_from_args(args)
        
        # Validate API keys
        validate_api_keys(config)
        
        # Create optimizer
        optimizer = GepaOptimizer(config=config)
        
        # Run optimization (async)
        print(f"🚀 Starting optimization with model: {config.model.model_name}")
        result = asyncio.run(optimizer.train(
            seed_prompt=args.prompt,
            dataset=args.dataset
        ))
        
        # Output results
        output_results(result, args.output)
        
        print("✅ Optimization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def load_config_from_file(config_path: str) -> OptimizationConfig:
    """Load configuration from JSON file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_data = json.load(f)
    
    # Convert model configs
    if 'model' in config_data and isinstance(config_data['model'], dict):
        config_data['model'] = ModelConfig(**config_data['model'])
    
    if 'reflection_model' in config_data and isinstance(config_data['reflection_model'], dict):
        config_data['reflection_model'] = ModelConfig(**config_data['reflection_model'])
    
    return OptimizationConfig(**config_data)


def create_config_from_args(args) -> OptimizationConfig:
    """Create configuration from command line arguments"""
    if not args.model:
        raise ValueError("Either --model or --config must be specified")
    
    # Parse model specification
    model_config = ModelConfig.from_string(args.model)
    
    reflection_model_config = None
    if args.reflection_model:
        reflection_model_config = ModelConfig.from_string(args.reflection_model)
    
    return OptimizationConfig(
        model=model_config,
        reflection_model=reflection_model_config,
        max_iterations=args.max_iterations,
        max_metric_calls=args.max_metric_calls,
        batch_size=args.batch_size
    )


def validate_api_keys(config: OptimizationConfig):
    """Validate that required API keys are available"""
    api_manager = APIKeyManager()
    
    providers = [config.model.provider]
    if config.reflection_model:
        providers.append(config.reflection_model.provider)
    
    missing_keys = api_manager.get_missing_keys(providers)
    
    if missing_keys:
        print("❌ Missing API keys for the following providers:")
        for provider in missing_keys:
            print(f"   - {provider.upper()}_API_KEY")
        print("\nPlease set the required environment variables or use a .env file")
        sys.exit(1)

def output_results(result, output_path: Optional[str]):
    """Output optimization results"""
    output_data = {
        "optimized_prompt": result.prompt,
        "original_prompt": result.original_prompt,
        "improvement_metrics": result.improvement_data,
        "optimization_time": result.optimization_time,
        "status": result.status,
        "session_id": result.session_id
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"📄 Results saved to: {output_path}")
    else:
        print("\n📊 Optimization Results:")
        print(f"Session ID: {result.session_id}")
        print(f"Status: {result.status}")
        print(f"Time: {result.optimization_time:.2f}s")
        print(f"\nOriginal Prompt:\n{result.original_prompt}")
        print(f"\nOptimized Prompt:\n{result.prompt}")
        
        if 'improvement_percent' in result.improvement_data:
            print(f"\nImprovement: {result.improvement_data['improvement_percent']:.2f}%")


if __name__ == "__main__":
    main()
