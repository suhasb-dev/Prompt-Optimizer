# 🔧 API Reference

Complete API documentation for the GEPA Universal Prompt Optimizer library.

## 📋 What's in this section

- [Core Classes](core-classes.md) - Main classes and their methods
- [Configuration](configuration.md) - Configuration options and parameters
- [Evaluation Metrics](evaluation-metrics.md) - Custom evaluation system

## 🎯 Quick Reference

### Main Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `GepaOptimizer` | Main optimization engine | `train()`, `optimize()` |
| `OptimizationConfig` | Configuration management | Constructor, validation |
| `ModelConfig` | LLM model configuration | Provider setup, API keys |
| `BaseEvaluator` | Evaluation interface | `evaluate()` |
| `BaseLLMClient` | LLM integration | `generate()` |

### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | str | Target LLM model | Required |
| `max_iterations` | int | Max optimization rounds | 10 |
| `max_metric_calls` | int | Max evaluation calls | 50 |
| `batch_size` | int | Evaluation batch size | 4 |
| `early_stopping` | bool | Stop early if no improvement | True |

### Evaluation Metrics

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `composite_score` | float | Overall performance score | 0.0 - 1.0 |
| `accuracy` | float | Response accuracy | 0.0 - 1.0 |
| `relevance` | float | Response relevance | 0.0 - 1.0 |
| `clarity` | float | Response clarity | 0.0 - 1.0 |

## 🚀 Getting Started

1. **Import the library**: `from gepa_optimizer import GepaOptimizer`
2. **Create configuration**: `config = OptimizationConfig(...)`
3. **Run optimization**: `result = await optimizer.train(...)`

## 📚 Detailed Documentation

- **[Core Classes](core-classes.md)** - Complete class documentation
- **[Configuration](configuration.md)** - All configuration options
- **[Evaluation Metrics](evaluation-metrics.md)** - Custom evaluation system

## 🔗 Related Resources

- [Getting Started](../getting-started/) - Installation and basic usage
- [Tutorials](../tutorials/) - Real-world examples
- [Examples](../examples/) - Code examples
- [Architecture](../architecture/) - System design
