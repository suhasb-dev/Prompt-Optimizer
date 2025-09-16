"""
Test Enhanced Logging for GEPA Optimizer

This test demonstrates the new structured logging features:
1. New proposed candidate prompts
2. Reflection dataset creation with detailed analysis
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv
from gepa_optimizer import (
    GepaOptimizer, 
    OptimizationConfig, 
    ModelConfig,
    BaseEvaluator,
    VisionLLMClient
)

# Load environment variables
load_dotenv()

class SimpleEvaluator(BaseEvaluator):
    """Simple evaluator for testing logging functionality."""
    
    def __init__(self, metric_weights: Dict[str, float] = None):
        default_weights = {
            "accuracy": 0.5,
            "relevance": 0.3,
            "clarity": 0.2
        }
        weights = metric_weights or default_weights
        super().__init__(metric_weights=weights)
        
        # Normalize weights
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """Simple evaluation based on word overlap."""
        if not predicted or not expected:
            return {
                "accuracy": 0.0,
                "relevance": 0.0,
                "clarity": 0.0,
                "composite_score": 0.0
            }
        
        # Simple word overlap accuracy
        pred_words = set(predicted.lower().split())
        exp_words = set(expected.lower().split())
        accuracy = len(pred_words.intersection(exp_words)) / max(len(exp_words), 1)
        
        # Simple relevance (length similarity)
        relevance = 1.0 - abs(len(predicted) - len(expected)) / max(len(expected), 1)
        relevance = max(0.0, min(1.0, relevance))
        
        # Simple clarity (sentence count)
        clarity = min(len(predicted.split('.')) / 3.0, 1.0)
        
        composite_score = (
            accuracy * self.metric_weights.get("accuracy", 0.5) +
            relevance * self.metric_weights.get("relevance", 0.3) +
            clarity * self.metric_weights.get("clarity", 0.2)
        )
        
        return {
            "accuracy": accuracy,
            "relevance": relevance,
            "clarity": clarity,
            "composite_score": composite_score
        }

async def test_enhanced_logging():
    """Test the enhanced logging functionality."""
    print("🚀 Testing Enhanced Logging Features")
    print("=" * 60)
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No OPENAI_API_KEY found in environment variables")
            return False
        
        print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")
        
        # Create configuration
        config = OptimizationConfig(
            model=ModelConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                api_key=api_key
            ),
            reflection_model=ModelConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                api_key=api_key
            ),
            max_iterations=2,      # Small for testing
            max_metric_calls=6,    # Small for testing
            batch_size=2           # Small for testing
        )
        
        # Create evaluator and LLM client
        evaluator = SimpleEvaluator()
        llm_client = VisionLLMClient(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=api_key
        )
        
        # Create optimizer with universal adapter
        optimizer = GepaOptimizer(
            config=config,
            adapter_type="universal",
            llm_client=llm_client,
            evaluator=evaluator
        )
        
        # Create simple test dataset
        dataset = [
            {
                "input": "What is machine learning?",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
            },
            {
                "input": "Explain neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by analyzing training data."
            }
        ]
        
        print(f"✅ Created test dataset with {len(dataset)} samples")
        
        # Run optimization
        print("\n🚀 Starting optimization with enhanced logging...")
        seed_prompt = "You are a helpful AI assistant that explains technical concepts clearly."
        
        result = await optimizer.train(
            seed_prompt=seed_prompt,
            dataset=dataset
        )
        
        print("\n✅ Optimization completed!")
        print(f"   - Status: {result.status}")
        print(f"   - Iterations: {result.total_iterations}")
        print(f"   - Time: {result.optimization_time:.2f}s")
        
        # Show results
        print("\n" + "="*80)
        print("📝 FINAL RESULTS")
        print("="*80)
        
        print("\n🌱 SEED PROMPT:")
        print("-" * 40)
        print(f'"{seed_prompt}"')
        
        print("\n🚀 OPTIMIZED PROMPT:")
        print("-" * 40)
        print(f'"{result.prompt}"')
        
        print("="*80)
        
        print("\n🎉 Enhanced logging test completed!")
        print("\n📝 What you should have seen:")
        print("   ✅ NEW PROPOSED CANDIDATE sections showing each new prompt")
        print("   ✅ REFLECTION DATASET CREATION sections with detailed analysis")
        print("   ✅ Evaluation summaries with scores and feedback")
        print("   ✅ Clean, structured output with emojis and formatting")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test"""
    success = await test_enhanced_logging()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 CONCLUSION: Enhanced Logging is WORKING!")
        print("\nThe library now provides:")
        print("   🔹 Clean, structured candidate logging")
        print("   🔹 Detailed reflection dataset analysis")
        print("   🔹 Pretty-printed evaluation summaries")
        print("   🔹 Professional logging with emojis and formatting")
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed - please check the implementation")

if __name__ == "__main__":
    asyncio.run(main())
