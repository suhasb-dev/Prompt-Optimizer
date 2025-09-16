"""
Test Reflection Logging - Debug Version

This test specifically focuses on testing the reflection dataset logging
to see why it's not showing up in the output.
"""

import asyncio
import os
import sys
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

class DebugEvaluator(BaseEvaluator):
    """Debug evaluator that provides detailed feedback."""
    
    def __init__(self, metric_weights: Dict[str, float] = None):
        default_weights = {
            "accuracy": 0.4,
            "relevance": 0.3,
            "completeness": 0.2,
            "clarity": 0.1
        }
        weights = metric_weights or default_weights
        super().__init__(metric_weights=weights)
        
        # Normalize weights
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """Detailed evaluation with specific feedback."""
        if not predicted or not expected:
            return {
                "accuracy": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "composite_score": 0.0
            }
        
        # Word overlap accuracy
        pred_words = set(predicted.lower().split())
        exp_words = set(expected.lower().split())
        accuracy = len(pred_words.intersection(exp_words)) / max(len(exp_words), 1)
        
        # Relevance (key concept matching)
        key_concepts = ['machine', 'learning', 'artificial', 'intelligence', 'neural', 'network']
        relevance = sum(1 for concept in key_concepts if concept in predicted.lower()) / len(key_concepts)
        
        # Completeness (length ratio)
        completeness = min(len(predicted) / len(expected), 1.0)
        
        # Clarity (sentence structure)
        clarity = min(len(predicted.split('.')) / 2.0, 1.0)
        
        composite_score = (
            accuracy * self.metric_weights.get("accuracy", 0.4) +
            relevance * self.metric_weights.get("relevance", 0.3) +
            completeness * self.metric_weights.get("completeness", 0.2) +
            clarity * self.metric_weights.get("clarity", 0.1)
        )
        
        return {
            "accuracy": accuracy,
            "relevance": relevance,
            "completeness": completeness,
            "clarity": clarity,
            "composite_score": composite_score
        }

async def test_reflection_logging():
    """Test the reflection logging specifically."""
    print("🔍 Testing Reflection Logging Debug")
    print("=" * 60)
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No OPENAI_API_KEY found in environment variables")
            return False
        
        print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")
        
        # Create configuration with very small budget to force reflection
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
            max_iterations=1,      # Very small for testing
            max_metric_calls=4,    # Very small for testing
            batch_size=2,          # Small for testing
            verbose=False          # Disable progress bar to avoid tqdm requirement
        )
        
        # Create evaluator and LLM client
        evaluator = DebugEvaluator()
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
        
        # Force flush output
        sys.stdout.flush()
        
        # Run optimization
        print("\n🚀 Starting optimization with reflection logging debug...")
        print("   (This should show NEW PROPOSED CANDIDATE and REFLECTION DATASET sections)")
        sys.stdout.flush()
        
        seed_prompt = "You are a helpful AI assistant."
        
        result = await optimizer.train(
            seed_prompt=seed_prompt,
            dataset=dataset
        )
        
        print("\n✅ Optimization completed!")
        print(f"   - Status: {result.status}")
        print(f"   - Iterations: {result.total_iterations}")
        print(f"   - Time: {result.optimization_time:.2f}s")
        
        # Force flush output
        sys.stdout.flush()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test"""
    print("🔍 REFLECTION LOGGING DEBUG TEST")
    print("=" * 60)
    print("This test will show:")
    print("   ✅ NEW PROPOSED CANDIDATE sections")
    print("   ✅ REFLECTION DATASET CREATION sections")
    print("   ✅ Detailed analysis with feedback")
    print("=" * 60)
    
    success = await test_reflection_logging()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 REFLECTION LOGGING TEST COMPLETED!")
        print("\nIf you didn't see the reflection sections above,")
        print("there might be an issue with the logging implementation.")
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed - please check the implementation")

if __name__ == "__main__":
    asyncio.run(main())
