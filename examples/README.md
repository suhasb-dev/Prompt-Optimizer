# GEPA Optimizer Examples

This directory contains comprehensive examples showing how to use the GEPA Optimizer for prompt optimization.

## Quick Start

1. **Install the package:**
   ```bash
   pip install gepa-optimizer
   ```

2. **Set up your API keys:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Run the basic example:**
   ```bash
   python examples/basic_usage.py
   ```

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)
- Simple prompt optimization
- Environment variable API keys
- Hardcoded API keys
- Mixed provider configuration

### 2. Advanced Usage (`advanced_usage.py`)
- Advanced configuration options
- Custom model parameters
- Result analysis and saving
- API key management
- Error handling

### 3. CLI Usage (`cli_usage.md`)
- Command-line interface examples
- Configuration files
- Batch processing
- Output formats

## Dataset Format

All examples use the following dataset format for UI tree extraction:

```json
[
  {
    "input": "Extract UI elements from this screenshot",
    "output": "Button: Login, Text: Welcome to our app",
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "ui_tree": {
      "type": "button",
      "text": "Login",
      "bounds": [100, 200, 150, 230]
    }
  }
]
```

## Configuration Examples

### Simple Configuration
```python
config = OptimizationConfig(
    model="openai/gpt-4o",
    reflection_model="openai/gpt-4o",
    max_iterations=10,
    max_metric_calls=50,
    batch_size=4
)
```

### Advanced Configuration
```python
config = OptimizationConfig(
    model=ModelConfig(
        provider="openai",
        model_name="gpt-4o",
        api_key="your-key",
        temperature=0.7,
        max_tokens=2048
    ),
    reflection_model=ModelConfig(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        api_key="your-key",
        temperature=0.5
    ),
    max_iterations=50,
    max_metric_calls=300,
    batch_size=8,
    early_stopping=True,
    learning_rate=0.02,
    multi_objective=True,
    objectives=["accuracy", "relevance", "clarity"]
)
```

## Running Examples

### Python Examples
```bash
# Basic usage
python examples/basic_usage.py

# Advanced usage
python examples/advanced_usage.py
```

### CLI Examples
```bash
# Simple optimization
gepa-optimize \
  --model openai/gpt-4o \
  --prompt "Extract UI elements" \
  --dataset data/ui_dataset.json \
  --max-iterations 20

# With configuration file
gepa-optimize \
  --config config.json \
  --prompt "Analyze interface" \
  --dataset data/screenshots/
```

## Customization

### Custom Evaluation Metrics
```python
from gepa_optimizer import UITreeEvaluator

# Custom metric weights
metric_weights = {
    "structural_similarity": 0.5,
    "element_type_accuracy": 0.3,
    "spatial_accuracy": 0.1,
    "text_content_accuracy": 0.1
}

evaluator = UITreeEvaluator(metric_weights=metric_weights)
```

### Custom Data Loading
```python
from gepa_optimizer import DataLoader, UniversalConverter

# Load custom data
loader = DataLoader()
data = loader.load("custom_data.csv")

# Convert to GEPA format
converter = UniversalConverter()
train_data, val_data = converter.convert(data)
```

## Best Practices

1. **Start Small**: Begin with small datasets and low iteration counts
2. **Monitor Costs**: Use `max_cost_usd` to control spending
3. **Validate Data**: Ensure your dataset is properly formatted
4. **Save Results**: Always save optimization results for analysis
5. **Error Handling**: Implement proper error handling for production use

## Troubleshooting

### Common Issues

1. **API Key Errors**: Check environment variables and key validity
2. **Dataset Format**: Ensure JSON structure matches expected format
3. **Memory Issues**: Reduce batch size for large datasets
4. **Timeout Errors**: Increase timeout or reduce complexity

### Debug Mode
```python
import logging
from gepa_optimizer import setup_logging

# Enable debug logging
setup_logging(level="DEBUG")
```

## Support

For more help:
- Check the [CLI Usage Guide](cli_usage.md)
- Review the [main documentation](../README.md)
- Open an issue on GitHub
