# 📚 GEPA Universal Prompt Optimizer Documentation

Welcome to the comprehensive documentation for the GEPA Universal Prompt Optimizer - a production-ready Python library for universal prompt optimization using the GEPA framework.

## 🎯 What is GEPA Optimizer?

GEPA Optimizer is a sophisticated framework that automatically improves prompts for Large Language Models (LLMs) by iteratively evaluating and refining them based on performance metrics. Think of it as "AutoML for prompts" - it takes your initial prompt and dataset, then uses advanced optimization techniques to create a better-performing prompt.

## ⚠️ **IMPORTANT: Custom Components Required**


**You MUST create custom evaluators and LLM clients to use this library.** This is not optional - the GEPA Universal Prompt Optimizer is designed for specialized, domain-specific applications where you define your own success metrics.

### Why Custom Components?
- 🎯 **Domain-Specific**: Each use case needs different evaluation criteria
- 📊 **No Generic Metrics**: Generic evaluation doesn't work for specialized tasks  
- 🔧 **Your Success Criteria**: You define what "good" means for your problem
- 🚀 **Optimization Target**: The system optimizes based on YOUR metrics

## 📖 Documentation Sections

### 🚀 [Getting Started](getting-started/)
- [Installation](getting-started/installation.md) - How to install the library
- [Quick Start](getting-started/quick-start.md) - Get started in 5 minutes
- [Basic Usage](getting-started/basic-usage.md) - Your first optimization

### 📖 [Tutorials](tutorials/)
- [Text Generation Optimization](tutorials/text-generation-optimization.md) - Optimize prompts for text generation tasks
- [Customer Service Optimization](tutorials/customer-service-optimization.md) - Business-specific customer service optimization
- [UI Tree Extraction](tutorials/ui-tree-extraction.md) - Multi-modal UI automation optimization

### 🔧 [API Reference](api-reference/)
- Complete API documentation for all classes and methods
- Configuration options and parameters
- Advanced usage patterns

### 📁 [Examples](examples/)
- Ready-to-run code examples
- Different use cases and scenarios
- Best practices and patterns

### 🏗️ [Architecture](architecture/)
- [System Overview](architecture/system-overview.md) - Complete system architecture and design patterns
- Component relationships and data flow
- Extension points and customization

### 🤝 [Contributing](contributing/)
- Development setup and guidelines
- Code style and standards
- How to contribute to the project

## 🚀 Quick Start

### Installation

```bash
pip install gepa-optimizer
```

### Basic Usage

```python
import asyncio
from gepa_optimizer import GepaOptimizer, OptimizationConfig
from gepa_optimizer.evaluation import BaseEvaluator
from gepa_optimizer.llms import BaseLLMClient
from typing import Dict

# ⚠️ REQUIRED: Custom evaluator
class MyEvaluator(BaseEvaluator):
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        # Your custom metrics here
        accuracy = self._calculate_accuracy(predicted, expected)
        return {"accuracy": accuracy, "composite_score": accuracy}

# ⚠️ REQUIRED: Custom LLM client
class MyLLMClient(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your LLM integration here
        return "Generated response"

async def optimize_prompt():
    config = OptimizationConfig(
        model="openai/gpt-4o",
        max_iterations=5
    )
    
    optimizer = GepaOptimizer(config=config)
    evaluator = MyEvaluator()
    llm_client = MyLLMClient()
    
    result = await optimizer.train(
        dataset=your_data,
        config=config,
        adapter_type="universal",
        evaluator=evaluator,
        llm_client=llm_client
    )
    
    print(f"Improvement: {result.improvement_percentage:.1f}%")

asyncio.run(optimize_prompt())
```

## 🎯 Key Features

- **🔄 Universal Prompt Optimization**: Works with any LLM provider (OpenAI, Anthropic, Google, Hugging Face)
- **👁️ Multi-Modal Support**: Optimize prompts for vision-capable models (GPT-4V, Claude-3, Gemini)
- **📊 Advanced Evaluation**: Comprehensive metrics for UI tree extraction and general prompt performance
- **🏭 Production Ready**: Enterprise-grade reliability with async support, error handling, and monitoring
- **⚙️ Flexible Configuration**: Easy-to-use configuration system for any optimization scenario
- **💰 Cost Optimization**: Built-in budget controls and cost estimation
- **🎨 UI Tree Extraction**: Specialized for optimizing UI interaction and screen understanding tasks
- **🔧 Extensible Architecture**: Create custom evaluators and adapters for any use case

## 🧪 Testing & Validation

The library includes comprehensive test suites demonstrating real-world usage:

| Test File | Purpose | Key Features | Dataset |
|-----------|---------|--------------|---------|
| [`test_customer_service_optimization.py`](../test_customer_service_optimization.py) | Customer service optimization with business metrics | Real customer service dataset, measurable improvements, robust error handling | 100+ customer service interactions |
| [`test_text_generation.py`](../test_text_generation.py) | General text generation optimization | Custom evaluation metrics, universal adapter usage | Text generation samples |
| [`test_ui_optimization.py`](../test_ui_optimization.py) | Multi-modal UI tree extraction | Vision model integration, screenshot analysis, UI element extraction | UI screenshots + JSON trees |

### Running Tests

```bash
# Customer service optimization
python test_customer_service_optimization.py

# Text generation optimization  
python test_text_generation.py

# UI tree extraction
python test_ui_optimization.py
```

## 🏗️ Architecture Overview

The GEPA Universal Prompt Optimizer is built on a modular, extensible architecture:

### Core Components

1. **GepaOptimizer**: Main entry point for optimization
2. **BaseGepaAdapter**: Abstract base for custom adapters
3. **UniversalGepaAdapter**: Universal adapter for any use case
4. **BaseEvaluator**: Abstract base for custom evaluation metrics
5. **BaseLLMClient**: Abstract base for LLM provider integration

### Data Flow

```
Input Dataset → UniversalConverter → GEPA Framework → Optimization → Results
```

### Extension Points

- **Custom Evaluators**: Define domain-specific success metrics
- **Custom LLM Clients**: Integrate with any LLM provider
- **Custom Adapters**: Specialized optimization strategies
- **Custom Converters**: Handle different data formats

## 🔧 Configuration

The library uses a comprehensive configuration system:

```python
from gepa_optimizer import OptimizationConfig, ModelConfig

config = OptimizationConfig(
    # Model configuration
    model="openai/gpt-4o",
    reflection_model="openai/gpt-4o",
    
    # Budget controls
    max_iterations=10,
    max_metric_calls=50,
    max_cost_usd=5.0,
    
    # Performance settings
    batch_size=4,
    early_stopping=True,
    
    # Advanced options
    learning_rate=0.02,
    multi_objective=True,
    objectives=["accuracy", "relevance"]
)
```

## 📊 Evaluation Metrics

The library supports comprehensive evaluation through custom metrics:

### Built-in Evaluators

- **UITreeEvaluator**: UI element extraction accuracy
- **CustomerServiceEvaluator**: Business-specific customer service metrics
- **TextGenerationEvaluator**: General text generation quality

### Custom Evaluators

Create evaluators tailored to your specific use case:

```python
class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        # Your custom metrics
        accuracy = self._calculate_accuracy(predicted, expected)
        relevance = self._calculate_relevance(predicted, expected)
        
        # Required: composite_score drives optimization
        composite_score = (accuracy * 0.7 + relevance * 0.3)
        
        return {
            "accuracy": accuracy,
            "relevance": relevance,
            "composite_score": composite_score
        }
```

## 🚀 Use Cases

### 1. Customer Service Optimization
- Optimize customer service prompts for better response quality
- Measure improvements in helpfulness, empathy, and solution focus
- Real-world business impact with measurable ROI

### 2. UI Tree Extraction
- Multi-modal optimization for UI automation
- Vision model integration for screenshot analysis
- Specialized metrics for UI element detection accuracy

### 3. Text Generation
- General-purpose text generation optimization
- Custom evaluation metrics for specific domains
- Universal adapter for any text-based use case

### 4. Custom Applications
- Domain-specific optimization for specialized tasks
- Custom evaluators for unique success criteria
- Flexible architecture for any optimization scenario

## 🔒 Security & Best Practices

- **API Key Management**: Secure handling of LLM provider credentials
- **Cost Controls**: Built-in budget limits and cost estimation
- **Error Handling**: Comprehensive error handling and recovery
- **Logging**: Detailed logging for debugging and monitoring
- **Validation**: Input validation and data sanitization

## 📈 Performance

- **Async Support**: Non-blocking operations for better performance
- **Batch Processing**: Efficient batch evaluation
- **Early Stopping**: Stop optimization when no improvement is detected
- **Cost Optimization**: Minimize API calls and costs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing/README.md) for details on:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 📞 Support

- 📧 Email: s8hasgrylls@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/suhasb-dev/Prompt-Optimizer/issues)
- 📖 Documentation: [GitBook](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Made with ❤️ for the AI community**