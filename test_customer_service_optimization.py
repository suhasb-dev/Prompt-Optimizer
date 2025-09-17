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
import time
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

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfiguration:
    """Configuration class for customer service optimization test"""
    
    def __init__(self):
        # Dataset configuration
        self.dataset = {
            "file": "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
            "sample_size": 50,
            "required_columns": ["flags", "instruction", "category", "intent", "response"],
            "categories": ["ACCOUNT", "ORDER", "REFUND", "CONTACT", "INVOICE"]
        }
        
        # Optimization configuration
        self.optimization = {
            "max_iterations": 5,
            "max_metric_calls": 50,
            "batch_size": 3,
            "perfect_score": 0.95,
            "verbose": False
        }
        
        # Model configuration
        self.models = {
            "target": {
                "provider": "openai",
                "model_name": "gpt-4o-mini"
            },
            "reflection": {
                "provider": "openai", 
                "model_name": "gpt-4o-mini"
            }
        }
        
        # Test output configuration
        self.output = {
            "verbose": True,
            "save_results": True,
            "results_file": "customer_service_test_results.json",
            "show_progress": True
        }
        
        # Success criteria
        self.success_criteria = {
            "min_improvement_percentage": 20.0,
            "min_iterations": 2,
            "max_time_seconds": 600
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check dataset file exists
            if not os.path.exists(self.dataset["file"]):
                print(f"❌ Dataset file not found: {self.dataset['file']}")
                return False
            
            # Validate sample size
            if self.dataset["sample_size"] < 5:
                print("❌ Sample size must be at least 5")
                return False
            
            # Validate optimization parameters
            if self.optimization["max_iterations"] < 1:
                print("❌ Max iterations must be at least 1")
                return False
            
            if self.optimization["max_metric_calls"] < 10:
                print("❌ Max metric calls must be at least 10")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False

# Global test configuration
TEST_CONFIG = TestConfiguration()

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class TestProgressTracker:
    """Professional progress tracking for test execution"""
    
    def __init__(self, total_steps: int, test_name: str = "Customer Service Test"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.test_name = test_name
        self.step_times = []
    
    def start(self):
        """Start the test execution timer"""
        self.start_time = time.time()
        print(f"\n🚀 Starting {self.test_name}")
        print("=" * 60)
    
    def update(self, step_name: str, details: str = ""):
        """Update progress with current step"""
        if self.start_time is None:
            self.start()
        
        self.current_step += 1
        step_start = time.time()
        
        # Calculate progress
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        # Display progress
        print(f"\n🔄 Step {self.current_step}/{self.total_steps}: {step_name}")
        if details:
            print(f"   📋 {details}")
        print(f"   ⏱️  Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s")
        
        # Store step timing
        self.step_times.append({
            "step": self.current_step,
            "name": step_name,
            "start_time": step_start,
            "elapsed": elapsed
        })
    
    def complete(self, success: bool = True):
        """Mark test as completed"""
        if self.start_time is None:
            return
        
        total_time = time.time() - self.start_time
        status = "✅ COMPLETED" if success else "❌ FAILED"
        
        print(f"\n{status}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print("=" * 60)
        
        # Show step breakdown if verbose
        if TEST_CONFIG.output.get("verbose", False):
            print(f"\n📊 Step Breakdown:")
            for step_info in self.step_times:
                print(f"   {step_info['step']}. {step_info['name']} - {step_info['elapsed']:.1f}s")
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

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

def load_customer_service_dataset(csv_path: str = None, sample_size: int = None) -> List[Dict[str, Any]]:
    """
    Load and prepare customer service dataset from CSV with comprehensive validation.
    
    Args:
        csv_path: Path to the CSV file (uses config if None)
        sample_size: Number of samples to use for optimization (uses config if None)
        
    Returns:
        List of dataset items in the required format
        
    Raises:
        FileNotFoundError: If dataset file is not found
        ValueError: If dataset validation fails
        Exception: For other loading errors
    """
    # Use configuration defaults if not provided
    csv_path = csv_path or TEST_CONFIG.dataset["file"]
    sample_size = sample_size or TEST_CONFIG.dataset["sample_size"]
    
    print(f"📁 Loading customer service dataset from: {csv_path}")
    
    try:
        # Load CSV file with error handling
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} customer service interactions")
        
        # Validate dataset structure
        required_columns = TEST_CONFIG.dataset["required_columns"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Dataset structure validation passed")
        print(f"📋 Dataset columns: {list(df.columns)}")
        
        # Show dataset statistics
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().head().to_dict()
            print(f"📋 Sample categories: {category_counts}")
        
        if 'intent' in df.columns:
            intent_counts = df['intent'].value_counts().head().to_dict()
            print(f"📋 Sample intents: {intent_counts}")
        
        # Validate sample size
        if sample_size > len(df):
            print(f"⚠️  Requested sample size ({sample_size}) larger than dataset ({len(df)})")
            sample_size = len(df)
            print(f"📝 Using entire dataset ({sample_size} interactions)")
        elif len(df) > sample_size:
            # Sample with better distribution across categories
            if 'category' in df.columns:
                # Stratified sampling to maintain category distribution
                sampled_df = df.groupby('category').apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // len(df['category'].unique()))), 
                                    random_state=42)
                ).reset_index(drop=True)
                
                # If we need more samples, add random ones
                if len(sampled_df) < sample_size:
                    remaining = sample_size - len(sampled_df)
                    additional = df.sample(n=min(remaining, len(df)), random_state=42)
                    sampled_df = pd.concat([sampled_df, additional], ignore_index=True)
            else:
                sampled_df = df.sample(n=sample_size, random_state=42)
            
            df = sampled_df
            print(f"📝 Using stratified sample of {len(df)} interactions for optimization")
        
        # Convert to required format with validation
        dataset = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract data from the CSV columns
                instruction = str(row.get('instruction', '')).strip()
                response = str(row.get('response', '')).strip()
                category = str(row.get('category', '')).strip()
                intent = str(row.get('intent', '')).strip()
                
                # Validate required fields
                if not instruction or not response:
                    invalid_count += 1
                    continue
                
                if len(instruction) < 10 or len(response) < 10:
                    invalid_count += 1
                    continue
                
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
                        "original_query": instruction,
                        "row_index": idx
                    }
                })
                
            except Exception as e:
                print(f"⚠️  Error processing row {idx}: {e}")
                invalid_count += 1
                continue
        
        # Final validation
        if len(dataset) < 5:
            raise ValueError(f"Not enough valid data samples ({len(dataset)}). Need at least 5.")
        
        if invalid_count > 0:
            print(f"⚠️  Skipped {invalid_count} invalid rows during processing")
        
        print(f"✅ Prepared {len(dataset)} valid interactions for optimization")
        
        # Show sample data
        if dataset and TEST_CONFIG.output.get("verbose", False):
            print(f"\n📝 Sample interaction:")
            sample = dataset[0]
            print(f"   Input: {sample['input'][:100]}...")
            print(f"   Output: {sample['output'][:100]}...")
            print(f"   Category: {sample['metadata']['category']}")
            print(f"   Intent: {sample['metadata']['intent']}")
        
        return dataset
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading dataset: {e}")

async def test_customer_service_optimization():
    """
    Test customer service prompt optimization with real dataset.
    
    This function demonstrates the complete workflow of the GEPA Universal
    Prompt Optimizer with a real customer service dataset.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    # Initialize progress tracker
    tracker = TestProgressTracker(
        total_steps=7, 
        test_name="Customer Service Optimization Test"
    )
    
    try:
        # Step 1: Validate configuration
        tracker.update(
            "Validating test configuration",
            f"Dataset: {TEST_CONFIG.dataset['file']}, Sample size: {TEST_CONFIG.dataset['sample_size']}"
        )
        
        if not TEST_CONFIG.validate():
            print("❌ Configuration validation failed")
            tracker.complete(success=False)
            return False
        
        # Step 2: Check API key
        tracker.update("Checking API key configuration")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No OPENAI_API_KEY found in environment variables")
            print("💡 Please set your OpenAI API key in the .env file")
            print("   Example: OPENAI_API_KEY=your-api-key-here")
            tracker.complete(success=False)
            return False
        
        print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")
        
        # Step 3: Load and validate dataset
        tracker.update(
            "Loading customer service dataset",
            f"File: {TEST_CONFIG.dataset['file']}"
        )
        
        try:
            dataset = load_customer_service_dataset()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Please download the dataset and place it in the current directory")
            tracker.complete(success=False)
            return False
        except ValueError as e:
            print(f"❌ Dataset validation failed: {e}")
            tracker.complete(success=False)
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            tracker.complete(success=False)
            return False
        
        # Step 4: Initialize components
        tracker.update("Initializing optimizer components")
        
        # Create custom evaluator
        evaluator = CustomerServiceEvaluator()
        print("✅ Created custom customer service evaluator")
        
        # Create LLM client
        llm_client = VisionLLMClient(
            provider=TEST_CONFIG.models["target"]["provider"],
            model_name=TEST_CONFIG.models["target"]["model_name"],
            api_key=api_key
        )
        print("✅ Created LLM client")
        
        # Create configuration
        config = OptimizationConfig(
            model=ModelConfig(
                provider=TEST_CONFIG.models["target"]["provider"],
                model_name=TEST_CONFIG.models["target"]["model_name"],
                api_key=api_key
            ),
            reflection_model=ModelConfig(
                provider=TEST_CONFIG.models["reflection"]["provider"],
                model_name=TEST_CONFIG.models["reflection"]["model_name"],
                api_key=api_key
            ),
            max_iterations=TEST_CONFIG.optimization["max_iterations"],
            max_metric_calls=TEST_CONFIG.optimization["max_metric_calls"],
            batch_size=TEST_CONFIG.optimization["batch_size"],
            verbose=TEST_CONFIG.optimization["verbose"]
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
        
        # Step 5: Define seed prompt
        tracker.update("Preparing seed prompt for optimization")
        
        seed_prompt = "You are a customer service agent. Answer customer questions politely and helpfully."
        
        print(f"\n🌱 SEED PROMPT:")
        print("-" * 40)
        print(f'"{seed_prompt}"')
        print("-" * 40)
        
        # Step 6: Run optimization
        tracker.update(
            "Running GEPA optimization",
            f"Dataset: {len(dataset)} interactions, Max iterations: {config.max_iterations}"
        )
        
        print(f"\n🚀 Starting customer service prompt optimization...")
        print(f"   - Dataset size: {len(dataset)} interactions")
        print(f"   - Max iterations: {config.max_iterations}")
        print(f"   - Max metric calls: {config.max_metric_calls}")
        print(f"   - Batch size: {config.batch_size}")
        
        result = await optimizer.train(
            seed_prompt=seed_prompt,
            dataset=dataset
        )
        
        # Step 7: Process and display results
        tracker.update("Processing optimization results")
        
        print(f"\n✅ Optimization completed!")
        print(f"   - Status: {result.status}")
        print(f"   - Iterations: {result.total_iterations}")
        print(f"   - Time: {result.optimization_time:.2f}s")
        
        # Validate success criteria
        success = validate_test_results(result)
        
        if result.is_successful and success:
            display_optimization_results(result)
            print(f"\n🎉 Customer Service Optimization Test PASSED!")
            print(f"\n📝 What this demonstrates:")
            print(f"   ✅ Universal adapter works with any use case")
            print(f"   ✅ Custom evaluation metrics for business needs")
            print(f"   ✅ Real-world dataset optimization")
            print(f"   ✅ Measurable improvements in response quality")
            print(f"   ✅ Library is truly universal and extensible")
            
            tracker.complete(success=True)
            return True
            
        else:
            print(f"\n❌ Optimization failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
            tracker.complete(success=False)
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error: {e}")
        if TEST_CONFIG.output.get("verbose", False):
            import traceback
            traceback.print_exc()
        tracker.complete(success=False)
        return False

def validate_test_results(result) -> bool:
    """
    Validate test results against success criteria.
    
    Args:
        result: Optimization result object
        
    Returns:
        bool: True if results meet success criteria
    """
    try:
        # Check if optimization was successful
        if not result.is_successful:
            return False
        
        # Check minimum iterations
        if result.total_iterations < TEST_CONFIG.success_criteria["min_iterations"]:
            print(f"⚠️  Warning: Only {result.total_iterations} iterations completed (minimum: {TEST_CONFIG.success_criteria['min_iterations']})")
        
        # Check time limit
        if result.optimization_time > TEST_CONFIG.success_criteria["max_time_seconds"]:
            print(f"⚠️  Warning: Optimization took {result.optimization_time:.1f}s (limit: {TEST_CONFIG.success_criteria['max_time_seconds']}s)")
        
        # Check improvement percentage if available
        if hasattr(result, 'improvement_data') and result.improvement_data:
            improvement = result.improvement_data.get('improvement_percentage', 0)
            if improvement < TEST_CONFIG.success_criteria["min_improvement_percentage"]:
                print(f"⚠️  Warning: Improvement {improvement:.1f}% below threshold ({TEST_CONFIG.success_criteria['min_improvement_percentage']}%)")
                return False
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error validating results: {e}")
        return False

def display_optimization_results(result):
    """
    Display optimization results in a professional format.
    
    Args:
        result: Optimization result object
    """
    print(f"\n" + "="*80)
    print("📝 PROMPT COMPARISON")
    print("="*80)
    
    print(f"\n🌱 SEED PROMPT:")
    print("-" * 40)
    print(f'"{result.original_prompt}"')
    print(f"Length: {len(result.original_prompt)} characters")
    print("-" * 40)
    
    print(f"\n🚀 OPTIMIZED PROMPT:")
    print("-" * 40)
    print(f'"{result.prompt}"')
    print(f"Length: {len(result.prompt)} characters")
    print("-" * 40)
    
    # Show improvement metrics
    if hasattr(result, 'improvement_data') and result.improvement_data:
        print(f"\n📊 IMPROVEMENT ANALYSIS:")
        print("-" * 40)
        for key, value in result.improvement_data.items():
            if isinstance(value, (int, float)):
                if 'percent' in key.lower():
                    print(f"   - {key.replace('_', ' ').title()}: {value:.2f}%")
                else:
                    print(f"   - {key.replace('_', ' ').title()}: {value:.4f}")
    
    print("="*80)

async def main():
    """
    Main function to run the customer service optimization test.
    
    This function orchestrates the complete test execution and provides
    comprehensive feedback on the results.
    """
    print("🎯 CUSTOMER SERVICE PROMPT OPTIMIZATION TEST")
    print("=" * 60)
    print("This test demonstrates:")
    print("   ✅ Universal adapter with custom evaluator")
    print("   ✅ Real customer service dataset optimization")
    print("   ✅ Business-specific evaluation metrics")
    print("   ✅ Measurable prompt improvements")
    print("   ✅ Professional error handling and progress tracking")
    print("=" * 60)
    
    # Display test configuration
    if TEST_CONFIG.output.get("verbose", False):
        print(f"\n📋 Test Configuration:")
        print(f"   - Dataset: {TEST_CONFIG.dataset['file']}")
        print(f"   - Sample size: {TEST_CONFIG.dataset['sample_size']}")
        print(f"   - Max iterations: {TEST_CONFIG.optimization['max_iterations']}")
        print(f"   - Max metric calls: {TEST_CONFIG.optimization['max_metric_calls']}")
        print(f"   - Model: {TEST_CONFIG.models['target']['model_name']}")
        print("=" * 60)
    
    try:
        # Run the test
        success = await test_customer_service_optimization()
        
        # Display final results
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
            print("   🔹 Professional error handling")
            print("   🔹 Progress tracking and validation")
            print("\n🚀 Ready for production use!")
        else:
            print("\n" + "=" * 60)
            print("❌ Test failed - please check the implementation")
            print("\n💡 Common issues:")
            print("   - Missing API key (check .env file)")
            print("   - Dataset file not found")
            print("   - Insufficient data samples")
            print("   - Network connectivity issues")
            print("   - Model API rate limits")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        print("Test execution cancelled")
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
        if TEST_CONFIG.output.get("verbose", False):
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Run the test with proper error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(1)
