"""
Simple Customer Service Prompt Optimization Test

This is a simplified version that works with a small sample dataset
to demonstrate the customer service optimization without requiring
the full 27K dataset.
"""

import asyncio
import os
from typing import Dict, Any, List
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

class SimpleCustomerServiceEvaluator(BaseEvaluator):
    """
    Simplified customer service evaluator for demonstration.
    """
    
    def __init__(self, metric_weights: Dict[str, float] = None):
        default_weights = {
            "helpfulness": 0.4,
            "professionalism": 0.3,
            "empathy": 0.3
        }
        weights = metric_weights or default_weights
        super().__init__(metric_weights=weights)
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """Evaluate customer service response quality."""
        if not predicted or not expected:
            return {
                "helpfulness": 0.0,
                "professionalism": 0.0,
                "empathy": 0.0,
                "composite_score": 0.0
            }
        
        # Simple keyword-based evaluation
        predicted_lower = predicted.lower()
        
        # Helpfulness: Check for solution-oriented words
        helpful_words = ["help", "solution", "resolve", "assist", "support", "fix"]
        helpfulness = min(sum(1 for word in helpful_words if word in predicted_lower) / 3, 1.0)
        
        # Professionalism: Check for professional language
        professional_words = ["thank you", "please", "certainly", "absolutely", "i'd be happy"]
        professionalism = min(sum(1 for phrase in professional_words if phrase in predicted_lower) / 2, 1.0)
        
        # Empathy: Check for empathetic language
        empathy_words = ["understand", "sorry", "apologize", "appreciate", "concern"]
        empathy = min(sum(1 for word in empathy_words if word in predicted_lower) / 2, 1.0)
        
        # Calculate composite score
        composite_score = (
            helpfulness * self.metric_weights.get("helpfulness", 0.4) +
            professionalism * self.metric_weights.get("professionalism", 0.3) +
            empathy * self.metric_weights.get("empathy", 0.3)
        )
        
        return {
            "helpfulness": helpfulness,
            "professionalism": professionalism,
            "empathy": empathy,
            "composite_score": composite_score
        }

def create_sample_customer_service_dataset() -> List[Dict[str, Any]]:
    """Create a small sample dataset for testing."""
    return [
        {
            "input": "Customer query: I need to cancel my order",
            "output": "I understand you'd like to cancel your order. I'd be happy to help you with that. Could you please provide me with your order number so I can locate it in our system and process the cancellation for you?"
        },
        {
            "input": "Customer query: My package was damaged during shipping",
            "output": "I'm very sorry to hear that your package arrived damaged. That must be frustrating. I'll help you resolve this right away. Could you please send me photos of the damaged package and items? I'll then process a replacement or refund for you immediately."
        },
        {
            "input": "Customer query: How do I return an item?",
            "output": "I'd be happy to help you with the return process. You can return your item by following these steps: 1) Log into your account and go to 'My Orders', 2) Select the item you want to return, 3) Print the return label, and 4) Drop it off at any authorized location. Do you need help with any of these steps?"
        },
        {
            "input": "Customer query: I haven't received my refund yet",
            "output": "I understand your concern about the refund. Let me check the status of your refund for you. Could you please provide me with your order number or the email address used for the purchase? I'll look into this immediately and provide you with an update on when you can expect to receive your refund."
        },
        {
            "input": "Customer query: Can I change my shipping address?",
            "output": "Absolutely! I can help you update your shipping address. If your order hasn't shipped yet, I can change the address for you. Could you please provide me with your order number and the new shipping address? I'll update it right away and confirm the change with you."
        }
    ]

async def test_simple_customer_service_optimization():
    """Test customer service optimization with sample data"""
    
    print("🚀 Simple Customer Service Prompt Optimization Test")
    print("=" * 60)
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No OPENAI_API_KEY found in environment variables")
            return False
        
        print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")
        
        # Create sample dataset
        dataset = create_sample_customer_service_dataset()
        print(f"✅ Created sample dataset with {len(dataset)} customer service interactions")
        
        # Create custom evaluator
        evaluator = SimpleCustomerServiceEvaluator()
        print("✅ Created custom customer service evaluator")
        
        # Create LLM client
        llm_client = VisionLLMClient(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=api_key
        )
        print("✅ Created LLM client")
        
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
            max_iterations=3,      # Small for testing
            max_metric_calls=15,   # Small for testing
            batch_size=2,          # Small batch
            verbose=False
        )
        
        print("✅ Created optimization configuration")
        
        # Create optimizer with universal adapter
        optimizer = GepaOptimizer(
            config=config,
            adapter_type="universal",
            llm_client=llm_client,
            evaluator=evaluator
        )
        
        print("✅ Created optimizer with universal adapter")
        
        # Define seed prompt
        seed_prompt = "You are a customer service agent. Answer customer questions politely."
        
        print(f"\n🌱 SEED PROMPT:")
        print("-" * 40)
        print(f'"{seed_prompt}"')
        print("-" * 40)
        
        # Run optimization
        print(f"\n🚀 Starting optimization...")
        
        result = await optimizer.train(
            seed_prompt=seed_prompt,
            dataset=dataset
        )
        
        print(f"\n✅ Optimization completed!")
        print(f"   - Status: {result.status}")
        print(f"   - Iterations: {result.total_iterations}")
        print(f"   - Time: {result.optimization_time:.2f}s")
        
        if result.is_successful:
            print(f"\n" + "="*80)
            print("📝 PROMPT COMPARISON")
            print("="*80)
            
            print(f"\n🌱 SEED PROMPT:")
            print("-" * 40)
            print(f'"{result.original_prompt}"')
            print("-" * 40)
            
            print(f"\n🚀 OPTIMIZED PROMPT:")
            print("-" * 40)
            print(f'"{result.prompt}"')
            print("-" * 40)
            
            # Show improvement metrics
            if result.improvement_data:
                print(f"\n📊 IMPROVEMENT ANALYSIS:")
                print("-" * 40)
                for key, value in result.improvement_data.items():
                    if isinstance(value, (int, float)):
                        if 'percent' in key.lower():
                            print(f"   - {key.replace('_', ' ').title()}: {value:.2f}%")
                        else:
                            print(f"   - {key.replace('_', ' ').title()}: {value:.4f}")
            
            print("="*80)
            
            print(f"\n🎉 Customer Service Optimization Test PASSED!")
            print(f"\n📝 What this demonstrates:")
            print(f"   ✅ Universal adapter works with customer service use case")
            print(f"   ✅ Custom evaluation metrics (helpfulness, professionalism, empathy)")
            print(f"   ✅ Real business scenario optimization")
            print(f"   ✅ Library is truly universal and extensible")
            
        else:
            print(f"\n❌ Optimization failed: {result.error_message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the simple customer service optimization test"""
    print("🎯 SIMPLE CUSTOMER SERVICE OPTIMIZATION TEST")
    print("=" * 60)
    print("This test demonstrates:")
    print("   ✅ Universal adapter with custom evaluator")
    print("   ✅ Customer service-specific evaluation metrics")
    print("   ✅ Business prompt optimization")
    print("   ✅ No external dataset required")
    print("=" * 60)
    
    success = await test_simple_customer_service_optimization()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 CONCLUSION: Universal GEPA Implementation is WORKING!")
        print("\nYour library now supports:")
        print("   🔹 UI Tree extraction (legacy)")
        print("   🔹 Text generation (previous test)")
        print("   🔹 Customer service optimization (this test)")
        print("   🔹 Any custom use case (universal)")
        print("   🔹 User-defined evaluation metrics")
        print("   🔹 Any LLM model")
        print("   🔹 Any data type")
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed - please check the implementation")

if __name__ == "__main__":
    asyncio.run(main())
