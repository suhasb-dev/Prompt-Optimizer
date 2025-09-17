# 🚀 GEPA Universal Prompt Optimizer

[![PyPI version](https://badge.fury.io/py/gepa-optimizer.svg)](https://badge.fury.io/py/gepa-optimizer) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://github.com/suhasb-dev/Prompt-Optimizer/workflows/Tests/badge.svg)](https://github.com/suhasb-dev/Prompt-Optimizer/actions)

> **A production-ready Python library for universal prompt optimization using the GEPA  \[** [**https://arxiv.org/abs/2507.19457**](https://arxiv.org/abs/2507.19457) **] Built for developers who need reliable, scalable prompt optimization with comprehensive evaluation metrics.**

## ⚠️ **IMPORTANT: Custom Components Required**

**You MUST create custom evaluators and LLM clients to use this library.** This is not optional - the GEPA Universal Prompt Optimizer is designed for specialized, domain-specific applications where you define your own success metrics.

### Why Custom Components?

* 🎯 **Domain-Specific**: Each use case needs different evaluation criteria
* 📊 **No Generic Metrics**: Generic evaluation doesn't work for specialized tasks
* 🔧 **Your Success Criteria**: You define what "good" means for your problem
* 🚀 **Optimization Target**: The system optimizes based on YOUR metrics

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

## 📚 Documentation

**📖** [**View Full Documentation**](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/)

### Quick Links

* [🚀 Getting Started](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/getting-started/)
* [📖 Tutorials](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/tutorials/)
* [🔧 API Reference](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/api-reference/)
* [📁 Examples](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/examples/)
* [🏗️ Architecture](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/architecture/)

## 🎯 Key Features

* **🔄 Universal Prompt Optimization**: Works with any LLM provider (OpenAI, Anthropic, Google, Hugging Face)
* **👁️ Multi-Modal Support**: Optimize prompts for vision-capable models (GPT-4V, Claude-3, Gemini)
* **📊 Advanced Evaluation**: Comprehensive metrics for UI tree extraction and general prompt performance
* **🏭 Production Ready**: Enterprise-grade reliability with async support, error handling, and monitoring
* **⚙️ Flexible Configuration**: Easy-to-use configuration system for any optimization scenario
* **💰 Cost Optimization**: Built-in budget controls and cost estimation
* **🎨 UI Tree Extraction**: Specialized for optimizing UI interaction and screen understanding tasks
* **🔧 Extensible Architecture**: Create custom evaluators and adapters for any use case

## 🧪 Testing & Validation

The library includes comprehensive test suites demonstrating real-world usage:

| Test File                                                                        | Purpose                                             | Key Features                                                                  | Dataset                            |
| -------------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------- |
| [`test_customer_service_optimization.py`](test_customer_service_optimization.py) | Customer service optimization with business metrics | Real customer service dataset, measurable improvements, robust error handling | 100+ customer service interactions |
| [`test_text_generation.py`](test_text_generation.py)                             | General text generation optimization                | Custom evaluation metrics, universal adapter usage                            | Text generation samples            |
| [`test_ui_optimization.py`](test_ui_optimization.py)                             | Multi-modal UI tree extraction                      | Vision model integration, screenshot analysis, UI element extraction          | UI screenshots + JSON trees        |

### Running Tests

```bash
# Customer service optimization
python test_customer_service_optimization.py

# Text generation optimization  
python test_text_generation.py

# UI tree extraction
python test_ui_optimization.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](broken-reference) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE/) file for details.

## 📞 Support

* 📧 Email: s8hasgrylls@gmail.com
* 🐛 Issues: [GitHub Issues](https://github.com/suhasb-dev/Prompt-Optimizer/issues)
* 📖 Documentation: [GitBook](https://suhasb-dev.gitbook.io/gepa-universal-prompt-optimizer/)

***

**Made with ❤️ for the AI community**
