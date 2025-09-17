# 📚 Basic Usage

This guide covers the fundamental concepts and patterns for using the GEPA Universal Prompt Optimizer effectively.

## 🎯 Core Concepts

### 1. **OptimizationConfig** - Your Control Center

The `OptimizationConfig` class is your main configuration hub:

```python
from gepa_optimizer import OptimizationConfig, ModelConfig

config = OptimizationConfig(
    # Model configuration
    model="openai/gpt-4o",                    # Target model for optimization
    reflection_model="openai/gpt-4o",         # Model for self-reflection
    
    # Budget controls
    max_iterations=10,                        # Maximum optimization rounds
    max_metric_calls=50,                      # Maximum evaluation calls
    max_cost_usd=5.0,                        # Budget limit in USD
    
    # Performance settings
    batch_size=4,                            # Batch size for evaluation
    early_stopping=True,                     # Stop early if no improvement
    
    # Advanced options
    learning_rate=0.02,                      # Learning rate for optimization
    multi_objective=True,                    # Enable multi-objective optimization
    objectives=["accuracy", "relevance"]     # Optimization objectives
)
```

### 2. **Dataset Format** - How to Structure Your Data

The library supports multiple dataset formats:

#### CSV Format
```csv
input,output
"Explain AI", "AI is artificial intelligence..."
"What is ML?", "Machine learning is..."
```

#### JSON Format
```json
[
  {
    "input": "Explain what machine learning is",
    "output": "Machine learning is a subset of AI...",
    "image": "data:image/jpeg;base64,..."  // Optional for multi-modal
  }
]
```

#### UI Tree Dataset
```json
[
  {
    "input": "Extract UI elements from this screenshot",
    "output": "Button: Login, Text: Welcome",
    "image": "screenshot.jpg",
    "ui_tree": {
      "type": "button",
      "text": "Login",
      "bounds": [100, 200, 150, 230]
    }
  }
]
```

### 3. **Custom Evaluators** - ⚠️ **REQUIRED: Define Your Success Metrics**

**You MUST create custom evaluators - this is not optional!** The library cannot work without your domain-specific evaluation metrics:

```python
from gepa_optimizer.evaluation import BaseEvaluator
from typing import Dict

class MyCustomEvaluator(BaseEvaluator):
    """
    ⚠️ REQUIRED: Custom evaluator for your specific use case.
    
    You MUST implement evaluate() with:
    1. Your domain-specific metrics
    2. A composite_score (drives optimization)
    """
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        # Calculate your custom metrics
        accuracy = self._calculate_accuracy(predicted, expected)
        relevance = self._calculate_relevance(predicted, expected)
        clarity = self._calculate_clarity(predicted)
        
        # ⚠️ REQUIRED: Composite score drives the optimization
        composite_score = (accuracy * 0.4 + relevance * 0.4 + clarity * 0.2)
        
        return {
            "accuracy": accuracy,
            "relevance": relevance,
            "clarity": clarity,
            "composite_score": composite_score  # ⚠️ MANDATORY!
        }
    
    def _calculate_accuracy(self, predicted: str, expected: str) -> float:
        # Your accuracy calculation logic
        pass
    
    def _calculate_relevance(self, predicted: str, expected: str) -> float:
        # Your relevance calculation logic
        pass
    
    def _calculate_clarity(self, predicted: str) -> float:
        # Your clarity calculation logic
        pass
```

### 4. **Custom LLM Clients** - ⚠️ **REQUIRED: Use Any LLM Provider**

**You MUST create custom LLM clients - this is not optional!** The library needs your specific LLM integration:

```python
from gepa_optimizer.llms import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize your LLM client
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Your LLM generation logic
        response = self._call_llm_api(prompt, **kwargs)
        return response
```

## 🔄 Basic Workflow

### Step 1: Prepare Your Data

```python
# Load your dataset
dataset = [
    {"input": "Question 1", "output": "Expected answer 1"},
    {"input": "Question 2", "output": "Expected answer 2"},
    # ... more samples
]
```

### Step 2: Configure Optimization

```python
config = OptimizationConfig(
    model="openai/gpt-3.5-turbo",
    max_iterations=5,
    max_metric_calls=20
)
```

### Step 3: ⚠️ **REQUIRED: Create Custom Components**

**You cannot skip this step - both components are mandatory:**

```python
# ⚠️ REQUIRED: Custom evaluator with your metrics
evaluator = MyCustomEvaluator()

# ⚠️ REQUIRED: Custom LLM client for your provider
llm_client = MyLLMClient(api_key="your-key")
```

### Step 4: Run Optimization

```python
optimizer = GepaOptimizer(config=config)

result = await optimizer.train(
    dataset=dataset,
    config=config,
    adapter_type="universal",
    evaluator=evaluator,
    llm_client=llm_client
)
```

### Step 5: Analyze Results

```python
print(f"Improvement: {result.improvement_percentage:.1f}%")
print(f"Original: {result.original_prompt}")
print(f"Optimized: {result.optimized_prompt}")
```

## 🎯 Common Patterns

### Pattern 1: Text Generation Optimization

```python
# For general text generation tasks
config = OptimizationConfig(
    model="openai/gpt-4o",
    max_iterations=8,
    objectives=["accuracy", "relevance", "clarity"]
)

evaluator = TextGenerationEvaluator()
```

### Pattern 2: Customer Service Optimization

```python
# For customer service applications
config = OptimizationConfig(
    model="openai/gpt-4o",
    max_iterations=10,
    objectives=["helpfulness", "empathy", "solution_focus"]
)

evaluator = CustomerServiceEvaluator()
```

### Pattern 3: Multi-Modal Optimization

```python
# For vision + text tasks
config = OptimizationConfig(
    model="openai/gpt-4o",  # Vision-capable model
    max_iterations=15,
    batch_size=2  # Smaller batch for vision processing
)

evaluator = UITreeEvaluator()
```

## ⚙️ Configuration Best Practices

### 1. **Start Small**
```python
# Begin with conservative settings
config = OptimizationConfig(
    max_iterations=3,
    max_metric_calls=10,
    batch_size=2
)
```

### 2. **Monitor Costs**
```python
# Set budget limits
config = OptimizationConfig(
    max_cost_usd=2.0,  # Budget limit
    max_iterations=5
)
```

### 3. **Use Early Stopping**
```python
# Stop if no improvement
config = OptimizationConfig(
    early_stopping=True,
    patience=2  # Stop after 2 iterations without improvement
)
```

### 4. **Optimize Batch Size**
```python
# Adjust based on your data size
config = OptimizationConfig(
    batch_size=4 if len(dataset) > 20 else 2
)
```

## 🔍 Understanding Results

### Result Object Structure

```python
result = await optimizer.train(...)

# Key properties
result.original_prompt          # Your starting prompt
result.optimized_prompt        # The improved prompt
result.improvement_percentage  # Performance improvement
result.total_time             # Time taken
result.iterations_run         # Number of iterations
result.final_metrics          # Final evaluation scores
result.reflection_history     # Optimization history
```

### Performance Metrics

```python
# Access detailed metrics
metrics = result.final_metrics
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Relevance: {metrics['relevance']:.3f}")
print(f"Composite Score: {metrics['composite_score']:.3f}")
```

## 🚀 Next Steps

Now that you understand the basics:

1. **Explore [Tutorials](../tutorials/)** - Real-world examples and use cases
2. **Check [API Reference](../api-reference/)** - Complete API documentation
3. **Try [Examples](../examples/)** - Ready-to-run code samples
4. **Learn [Architecture](../architecture/)** - Understand the system design

## 🆘 Common Issues

### Issue 1: "No improvement detected"
**Solution**: Check your evaluator's `composite_score` calculation

### Issue 2: "API key not found"
**Solution**: Set environment variable or pass directly in config

### Issue 3: "Dataset format error"
**Solution**: Ensure your dataset has `input` and `output` fields

### Issue 4: "Out of budget"
**Solution**: Increase `max_cost_usd` or reduce `max_iterations`

## 🎯 Key Takeaways

- **⚠️ Custom Components Required**: You MUST create evaluators and LLM clients
- **🎯 Domain-Specific Metrics**: Define what "good" means for YOUR use case
- **⚙️ Configuration is key**: Use `OptimizationConfig` to control everything
- **🚀 Start small**: Begin with conservative settings and scale up
- **💰 Monitor costs**: Set budget limits to avoid surprises
- **🔄 Iterate and improve**: Use results to refine your approach
- **💡 No Generic Solutions**: This library is designed for specialized applications
