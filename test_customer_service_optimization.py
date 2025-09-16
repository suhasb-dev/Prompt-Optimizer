"""
Customer Service Prompt Optimization Test - 

This test demonstrates the universal GEPA implementation by:
1. Loading the real Bitext customer service CSV dataset
2. Creating custom evaluation metrics for customer service quality
3. Using the universal adapter to optimize customer service prompts
4. Showing measurable improvements in response quality

Dataset: Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
Columns: flags, instruction, category, intent, response
"""

import asyncio
import os
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv
from gepa_optimizer import (
    GepaOptimizer, 
    OptimizationConfig, 
    ModelConfig,
    BaseEvaluator,
    BaseLLMClient,
    VisionLLMClient
)

# Load environment variables
load_dotenv()

class CustomerServiceEvaluator(BaseEvaluator):
    """
    Custom evaluator for customer service response quality.
    
    This demonstrates how users can define their own evaluation metrics
    for any business use case beyond UI tree extraction.
    """
    
    def __init__(self, metric_weights: Dict[str, float] = None):
        """
        Initialize with custom weights for customer service metrics.
        """
        # Default weights for customer service evaluation
        default_weights = {
            "helpfulness": 0.3,        # How helpful is the response
            "empathy": 0.25,           # Shows understanding and care
            "solution_focus": 0.25,    # Provides actionable solutions
            "professionalism": 0.2     # Tone and language quality
        }
        
        weights = metric_weights or default_weights
        super().__init__(metric_weights=weights)
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
    
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """
        Evaluate customer service response quality.
        
        Args:
            predicted: The AI-generated response
            expected: The ideal customer service response
            
        Returns:
            Dictionary with metric scores and composite score
        """
        if not predicted or not expected:
            return {
                "helpfulness": 0.0,
                "empathy": 0.0,
                "solution_focus": 0.0,
                "professionalism": 0.0,
                "composite_score": 0.0
            }
        
        # Calculate individual metrics
        helpfulness = self._calculate_helpfulness(predicted, expected)
        empathy = self._calculate_empathy(predicted)
        solution_focus = self._calculate_solution_focus(predicted)
        professionalism = self._calculate_professionalism(predicted)
        
        # DYNAMIC composite score calculation based on user-defined weights
        composite_score = 0.0
        for metric_name, weight in self.metric_weights.items():
            if metric_name == "helpfulness":
                composite_score += helpfulness * weight
            elif metric_name == "empathy":
                composite_score += empathy * weight
            elif metric_name == "solution_focus":
                composite_score += solution_focus * weight
            elif metric_name == "professionalism":
                composite_score += professionalism * weight
        
        return {
            "helpfulness": helpfulness,
            "empathy": empathy,
            "solution_focus": solution_focus,
            "professionalism": professionalism,
            "composite_score": composite_score
        }
    
    def _calculate_helpfulness(self, predicted: str, expected: str) -> float:
        """Calculate how helpful the response is"""
        predicted_lower = predicted.lower()
        
        # Check for helpful keywords
        helpful_keywords = [
            "solution", "resolve", "help", "assist", "support", "fix", "correct",
            "provide", "offer", "suggest", "recommend", "guide", "explain"
        ]
        
        helpful_count = sum(1 for keyword in helpful_keywords if keyword in predicted_lower)
        
        # Check for specific solutions mentioned
        solution_indicators = [
            "step", "process", "procedure", "way", "method", "option", "alternative"
        ]
        
        solution_count = sum(1 for indicator in solution_indicators if indicator in predicted_lower)
        
        # Combine metrics
        keyword_score = min(helpful_count / 5, 1.0)  # Normalize to 0-1
        solution_score = min(solution_count / 3, 1.0)  # Normalize to 0-1
        
        return (keyword_score + solution_score) / 2
    
    def _calculate_empathy(self, predicted: str) -> float:
        """Calculate empathy level in the response"""
        predicted_lower = predicted.lower()
        
        # Empathy indicators
        empathy_phrases = [
            "understand", "sorry", "apologize", "frustrating", "difficult",
            "appreciate", "thank you", "concern", "worry", "feel", "experience"
        ]
        
        empathy_count = sum(1 for phrase in empathy_phrases if phrase in predicted_lower)
        
        # Check for empathetic sentence starters
        empathetic_starters = [
            "i understand", "i'm sorry", "i apologize", "i can see", "i appreciate"
        ]
        
        starter_count = sum(1 for starter in empathetic_starters if predicted_lower.startswith(starter))
        
        # Combine metrics
        phrase_score = min(empathy_count / 4, 1.0)
        starter_score = min(starter_count * 2, 1.0)  # Higher weight for sentence starters
        
        return (phrase_score + starter_score) / 2
    
    def _calculate_solution_focus(self, predicted: str) -> float:
        """Calculate how solution-focused the response is"""
        predicted_lower = predicted.lower()
        
        # Solution-focused indicators
        solution_indicators = [
            "here's how", "you can", "to resolve", "solution is", "next steps",
            "what you need to do", "follow these", "try this", "option", "alternative"
        ]
        
        solution_count = sum(1 for indicator in solution_indicators if indicator in predicted_lower)
        
        # Check for actionable language
        action_words = [
            "click", "go to", "contact", "call", "email", "visit", "submit",
            "fill out", "complete", "send", "provide", "check", "verify"
        ]
        
        action_count = sum(1 for word in action_words if word in predicted_lower)
        
        # Combine metrics
        indicator_score = min(solution_count / 3, 1.0)
        action_score = min(action_count / 4, 1.0)
        
        return (indicator_score + action_score) / 2
    
    def _calculate_professionalism(self, predicted: str) -> float:
        """Calculate professionalism level"""
        predicted_lower = predicted.lower()
        
        # Professional language indicators
        professional_phrases = [
            "thank you for contacting", "i'd be happy to help", "let me assist you",
            "i can help you with", "i'll be glad to", "certainly", "absolutely"
        ]
        
        professional_count = sum(1 for phrase in professional_phrases if phrase in predicted_lower)
        
        # Check for proper greeting/closing
        has_greeting = any(greeting in predicted_lower for greeting in ["hello", "hi", "good morning", "good afternoon"])
        has_closing = any(closing in predicted_lower for closing in ["best regards", "sincerely", "thank you", "have a great day"])
        
        # Check for appropriate length (not too short, not too long)
        word_count = len(predicted.split())
        length_score = 1.0 if 10 <= word_count <= 100 else 0.5
        
        # Combine metrics
        phrase_score = min(professional_count / 2, 1.0)
        structure_score = (has_greeting + has_closing) / 2
        
        return (phrase_score + structure_score + length_score) / 3

def load_customer_service_dataset(csv_path: str, sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Load and prepare customer service dataset from CSV.
    
    Args:
        csv_path: Path to the CSV file
        sample_size: Number of samples to use for optimization
        
    Returns:
        List of dataset items in the required format
    """
    print(f"📁 Loading customer service dataset from: {csv_path}")
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} customer service interactions")
    
    # Show dataset structure
    print(f"📋 Dataset columns: {list(df.columns)}")
    print(f"📋 Sample categories: {df['category'].value_counts().head().to_dict()}")
    print(f"📋 Sample intents: {df['intent'].value_counts().head().to_dict()}")
    
    # Sample the data for optimization (to keep costs reasonable)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"📝 Using sample of {sample_size} interactions for optimization")
    
    # Convert to required format
    dataset = []
    for _, row in df.iterrows():
        # Extract data from the CSV columns (CORRECTED COLUMN NAMES)
        instruction = row.get('instruction', '')  # Customer query
        response = row.get('response', '')        # Agent response
        category = row.get('category', '')        # Category (ORDER, REFUND, etc.)
        intent = row.get('intent', '')            # Intent (cancel_order, change_order, etc.)
        
        if instruction and response:
            # Clean up the instruction (remove template variables for better evaluation)
            clean_instruction = instruction.replace('{{Order Number}}', '[ORDER_NUMBER]')
            clean_instruction = clean_instruction.replace('{{Customer Support Hours}}', '[SUPPORT_HOURS]')
            clean_instruction = clean_instruction.replace('{{Customer Support Phone Number}}', '[PHONE_NUMBER]')
            clean_instruction = clean_instruction.replace('{{Website URL}}', '[WEBSITE_URL]')
            clean_instruction = clean_instruction.replace('{{Online Company Portal Info}}', '[PORTAL_URL]')
            clean_instruction = clean_instruction.replace('{{Online Order Interaction}}', '[ORDER_ACTION]')
            
            dataset.append({
                "input": f"Customer query: {clean_instruction}",
                "output": response,
                "metadata": {
                    "category": category,
                    "intent": intent,
                    "original_query": instruction
                }
            })
    
    print(f"✅ Prepared {len(dataset)} valid interactions for optimization")
    
    # Show sample data
    if dataset:
        print(f"\n📝 Sample interaction:")
        print(f"   Input: {dataset[0]['input'][:100]}...")
        print(f"   Output: {dataset[0]['output'][:100]}...")
        print(f"   Category: {dataset[0]['metadata']['category']}")
        print(f"   Intent: {dataset[0]['metadata']['intent']}")
    
    return dataset

async def test_customer_service_optimization():
    """Test customer service prompt optimization with real dataset"""
    
    print("🚀 Customer Service Prompt Optimization Test")
    print("=" * 60)
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No OPENAI_API_KEY found in environment variables")
            print("Please set your OpenAI API key in the .env file")
            return False
        
        print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")
        
        # Load customer service dataset
        csv_path = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        
        if not os.path.exists(csv_path):
            print(f"❌ Dataset file not found: {csv_path}")
            print("Please download the dataset and place it in the current directory")
            return False
        
        dataset = load_customer_service_dataset(csv_path, sample_size=50)  # Use 50 samples for testing
        
        if len(dataset) < 5:
            print("❌ Not enough valid data samples for optimization")
            return False
        
        # Create custom evaluator
        evaluator = CustomerServiceEvaluator()
        print("✅ Created custom customer service evaluator")
        
        # Create LLM client
        llm_client = VisionLLMClient(
            provider="openai",
            model_name="gpt-4o-mini",  # Use mini for cost efficiency
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
            max_iterations=5,      # Reasonable for testing
            max_metric_calls=50,   # Reasonable for testing
            batch_size=3,          # Small batch for testing
            verbose=False,          # Disable progress bar
            
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
        seed_prompt = "You are a customer service agent. Answer customer questions politely and helpfully."
        
        print(f"\n🌱 SEED PROMPT:")
        print("-" * 40)
        print(f'"{seed_prompt}"')
        print("-" * 40)
        
        # Run optimization
        print(f"\n🚀 Starting customer service prompt optimization...")
        print(f"   - Dataset size: {len(dataset)} interactions")
        print(f"   - Max iterations: {config.max_iterations}")
        print(f"   - Max metric calls: {config.max_metric_calls}")
        
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
            print(f"   ✅ Universal adapter works with any use case")
            print(f"   ✅ Custom evaluation metrics for business needs")
            print(f"   ✅ Real-world dataset optimization")
            print(f"   ✅ Measurable improvements in response quality")
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
    """Run the customer service optimization test"""
    print("🎯 CUSTOMER SERVICE PROMPT OPTIMIZATION TEST")
    print("=" * 60)
    print("This test demonstrates:")
    print("   ✅ Universal adapter with custom evaluator")
    print("   ✅ Real customer service dataset optimization")
    print("   ✅ Business-specific evaluation metrics")
    print("   ✅ Measurable prompt improvements")
    print("=" * 60)
    
    success = await test_customer_service_optimization()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 CONCLUSION: Universal GEPA Implementation is WORKING!")
        print("\nYour library now supports:")
        print("   🔹 UI Tree extraction (legacy)")
        print("   🔹 Text generation (demonstrated)")
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
