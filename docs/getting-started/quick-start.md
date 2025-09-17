# ⚡ Quick Start

Get started with GEPA Universal Prompt Optimizer in 5 minutes! This guide will walk you through your first prompt optimization.

## ⚠️ **IMPORTANT: Custom Evaluators Required**

**You MUST define your own evaluation metrics to use this library.** The GEPA Universal Prompt Optimizer requires custom evaluators because:

- 🎯 **Domain-specific metrics**: Each use case needs different success criteria
- 📊 **No one-size-fits-all**: Generic metrics don't work for specialized tasks
- 🔧 **Flexibility**: You define what "good" means for your specific problem
- 🚀 **Optimization target**: The system optimizes based on YOUR metrics

## 🎯 What You'll Build

By the end of this guide, you'll have:
- ✅ Installed and configured the library
- ✅ **Created a custom evaluator** (required!)
- ✅ **Created a custom LLM client** (required!)
- ✅ Run your first prompt optimization
- ✅ Seen measurable improvements in prompt performance
- ✅ Understood the basic workflow

## Step 1: Installation

```bash
pip install gepa-optimizer
```

## Step 2: Set Up API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

## Step 3: Your First Optimization

Create a file called `quick_start.py`:

```python
import asyncio
from gepa_optimizer import GepaOptimizer, OptimizationConfig, ModelConfig
from gepa_optimizer.evaluation import BaseEvaluator
from gepa_optimizer.llms import BaseLLMClient
from typing import Dict

# Simple dataset for demonstration
dataset = [
    {
        "input": "Explain what machine learning is",
        "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
    },
    {
        "input": "What is deep learning?",
        "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
    }
]

# ⚠️ REQUIRED: Custom evaluator - you MUST create this!
class SimpleEvaluator(BaseEvaluator):
    """
    Custom evaluator for measuring prompt performance.
    
    ⚠️ IMPORTANT: You MUST implement the evaluate() method with:
    1. Your custom metrics (accuracy, relevance, etc.)
    2. A composite_score (required for optimization)
    """
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        # Example: Simple evaluation based on word overlap
        predicted_words = set(predicted.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return {"accuracy": 0.0, "composite_score": 0.0}
        
        overlap = len(predicted_words.intersection(expected_words))
        accuracy = overlap / len(expected_words)
        
        # ⚠️ REQUIRED: composite_score is mandatory for optimization
        return {
            "accuracy": accuracy,
            "composite_score": accuracy  # This drives the optimization!
        }

# ⚠️ REQUIRED: Custom LLM client - you MUST create this!
class SimpleLLMClient(BaseLLMClient):
    """
    Custom LLM client for interacting with your chosen model.
    
    ⚠️ IMPORTANT: You MUST implement the generate() method to:
    1. Connect to your LLM provider (OpenAI, Google, Anthropic, etc.)
    2. Handle API calls and responses
    3. Return generated text
    """
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Example: OpenAI API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content

async def main():
    # Create configuration
    config = OptimizationConfig(
        model="openai/gpt-3.5-turbo",
        reflection_model="openai/gpt-3.5-turbo",
        max_iterations=3,
        max_metric_calls=10,
        batch_size=2
    )
    
    # Initialize optimizer
    optimizer = GepaOptimizer(config=config)
    
    # ⚠️ REQUIRED: Create your custom components
    evaluator = SimpleEvaluator()  # Your custom evaluation metrics
    llm_client = SimpleLLMClient()  # Your custom LLM client
    
    print("🚀 Starting prompt optimization...")
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"⚙️ Max iterations: {config.max_iterations}")
    print()
    
    # Run optimization
    result = await optimizer.train(
        dataset=dataset,
        config=config,
        adapter_type="universal",
        evaluator=evaluator,
        llm_client=llm_client
    )
    
    # Display results
    print("✅ Optimization completed!")
    print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    print(f"⏱️ Time taken: {result.total_time:.1f}s")
    print()
    print("📝 PROMPT COMPARISON:")
    print(f"🌱 Original: {result.original_prompt}")
    print(f"🚀 Optimized: {result.optimized_prompt}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 4: Run Your First Optimization

```bash
python quick_start.py
```

## Expected Output

You should see something like this:

```
🚀 Starting prompt optimization...
📊 Dataset size: 2 samples
⚙️ Max iterations: 3

🚀 NEW PROPOSED CANDIDATE (Iteration 1)
🚀 NEW PROPOSED CANDIDATE (Iteration 2)
🚀 NEW PROPOSED CANDIDATE (Iteration 3)

✅ Optimization completed!
📈 Improvement: 45.2%
⏱️ Time taken: 12.3s

📝 PROMPT COMPARISON:
🌱 Original: You are a helpful assistant.
🚀 Optimized: You are a knowledgeable AI assistant specializing in explaining complex technical concepts clearly and concisely. Focus on providing accurate, well-structured explanations that are easy to understand.
```

## 🎉 Congratulations!

You've successfully run your first prompt optimization! Here's what happened:

1. **📊 Dataset Processing**: Your 2 samples were loaded and validated
2. **🔄 Optimization Loop**: The system ran 3 iterations of optimization
3. **📈 Performance Improvement**: Your prompt improved by ~45%
4. **⏱️ Fast Execution**: Completed in about 12 seconds

## 🔍 Understanding the Results

- **Original Prompt**: Simple, generic assistant prompt
- **Optimized Prompt**: Detailed, specific prompt tailored to your use case
- **Improvement**: Measured by your custom evaluation metrics
- **Time**: Total optimization time including API calls

## 🚀 Next Steps

Now that you've seen the basics, explore:

1. **[Basic Usage](basic-usage.md)** - More detailed examples and configurations
2. **[Tutorials](../tutorials/)** - Real-world use cases and advanced techniques
3. **[API Reference](../api-reference/)** - Complete API documentation
4. **[Examples](../examples/)** - Ready-to-run code examples

## 🎯 Key Takeaways

- **⚠️ Custom Components Required**: You MUST create evaluators and LLM clients
- **🎯 Domain-Specific**: Define metrics that matter for YOUR use case
- **🚀 Fast Results**: Get improvements in minutes, not hours
- **📊 Measurable**: See concrete performance improvements based on your metrics
- **🔧 Extensible**: Easy to adapt for any use case with custom components
- **💡 No Generic Solutions**: This library is designed for specialized, custom applications

## 🆘 Need Help?

- Check the [troubleshooting section](../troubleshooting.md)
- Open an issue on [GitHub](https://github.com/suhasb-dev/Prompt-Optimizer/issues)
- Contact us at s8hasgrylls@gmail.com
