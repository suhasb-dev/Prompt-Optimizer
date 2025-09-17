# 📁 Examples

Ready-to-run code examples and use cases for the GEPA Universal Prompt Optimizer.

## 📋 What's in this section

- [Basic Examples](basic-examples.md) - Simple optimization examples
- [Advanced Examples](advanced-examples.md) - Complex use cases and configurations
- [Custom Adapters](custom-adapters.md) - Creating custom adapters and evaluators

## 🎯 Example Categories

### 🌟 **Basic Examples**
Perfect for getting started with the library:

- **Simple Text Optimization** - Basic prompt optimization
- **CSV Dataset Processing** - Working with CSV data
- **Basic Configuration** - Simple configuration setup

### 🚀 **Advanced Examples**
For complex use cases and production scenarios:

- **Multi-Modal Optimization** - Vision + text optimization
- **Custom Evaluators** - Domain-specific evaluation metrics
- **Production Patterns** - Error handling and logging

### 🔧 **Custom Adapters**
For extending the library:

- **Creating Custom Evaluators** - Build your own evaluation metrics
- **Custom LLM Clients** - Integrate new LLM providers
- **Domain-Specific Optimization** - Specialized use cases

## 📊 Example Results

| Example | Use Case | Improvement | Time | Difficulty |
|---------|----------|-------------|------|------------|
| Simple Text | General optimization | 30-50% | 1-3 min | ⭐⭐ |
| Customer Service | Business optimization | 40-70% | 2-5 min | ⭐⭐⭐ |
| UI Tree Extraction | Multi-modal | 20-40% | 5-15 min | ⭐⭐⭐⭐ |

## 🚀 Quick Start

### 1. **Basic Text Optimization**
```python
from gepa_optimizer import GepaOptimizer, OptimizationConfig

config = OptimizationConfig(
    model="openai/gpt-3.5-turbo",
    max_iterations=3
)

optimizer = GepaOptimizer(config=config)
result = await optimizer.train(dataset=my_data)
```

### 2. **Customer Service Optimization**
```python
from gepa_optimizer.evaluation import CustomerServiceEvaluator

evaluator = CustomerServiceEvaluator()
result = await optimizer.train(
    dataset=customer_data,
    evaluator=evaluator
)
```

### 3. **Multi-Modal Optimization**
```python
config = OptimizationConfig(
    model="openai/gpt-4o",  # Vision model
    max_iterations=10
)

result = await optimizer.train(
    dataset=ui_dataset,
    config=config
)
```

## 📁 **Available Examples**

### **From Your Repository**
- `examples/basic_usage.py` - Basic optimization example
- `examples/advanced_usage.py` - Advanced configuration
- `examples/cli_usage.md` - Command-line interface
- `examples/gemini_usage.py` - Google Gemini integration

### **Test Files (Ready to Run)**
- `test_customer_service_optimization.py` - Customer service optimization
- `test_text_generation.py` - Text generation optimization
- `test_ui_optimization.py` - UI tree extraction

## 🎯 **Running Examples**

### **Python Examples**
```bash
# Basic usage
python examples/basic_usage.py

# Advanced usage
python examples/advanced_usage.py

# Customer service optimization
python test_customer_service_optimization.py
```

### **CLI Examples**
```bash
# Simple optimization
gepa-optimize --model openai/gpt-4o --prompt "Extract UI elements" --dataset data/ui_dataset.json

# With configuration file
gepa-optimize --config config.json --prompt "Analyze interface" --dataset data/screenshots/
```

## 🔧 **Customization**

### **Custom Evaluation Metrics**
```python
from gepa_optimizer.evaluation import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        # Your custom evaluation logic
        return {
            "custom_metric": score,
            "composite_score": weighted_score
        }
```

### **Custom LLM Clients**
```python
from gepa_optimizer.llms import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your LLM integration
        return response
```

## 🎯 **Best Practices**

1. **Start Small**: Begin with basic examples
2. **Use Real Data**: Test with your actual datasets
3. **Monitor Costs**: Set budget limits
4. **Save Results**: Always save optimization results
5. **Error Handling**: Implement proper error handling

## 🆘 **Need Help?**

- **Stuck on an example?** Check the troubleshooting section
- **Want to contribute?** See our [Contributing Guide](../contributing/)
- **Found a bug?** Open an issue on [GitHub](https://github.com/suhasb-dev/Prompt-Optimizer/issues)

## 🔗 **Related Resources**

- [Getting Started](../getting-started/) - Installation and setup
- [Tutorials](../tutorials/) - Step-by-step guides
- [API Reference](../api-reference/) - Complete API documentation
- [Architecture](../architecture/) - System design

---

**🎉 Ready to start?** Pick an example that matches your use case and begin optimizing!
