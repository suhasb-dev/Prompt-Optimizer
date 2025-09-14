# CLI Usage Guide

The GEPA Optimizer provides a command-line interface for easy prompt optimization.

## Installation

```bash
pip install gepa-optimizer
```

## Basic Usage

### Simple Optimization

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --prompt "Extract UI elements from this screenshot" \
  --dataset data/ui_dataset.json \
  --max-iterations 20 \
  --max-metric-calls 100
```

### Using Configuration File

Create a configuration file `config.json`:

```json
{
  "model": {
    "provider": "openai",
    "model_name": "gpt-4o",
    "api_key": "your-openai-key"
  },
  "reflection_model": {
    "provider": "anthropic", 
    "model_name": "claude-3-opus-20240229",
    "api_key": "your-anthropic-key"
  },
  "max_iterations": 50,
  "max_metric_calls": 300,
  "batch_size": 8,
  "early_stopping": true,
  "learning_rate": 0.02
}
```

Then run:

```bash
gepa-optimize \
  --config config.json \
  --prompt "Analyze this interface" \
  --dataset data/screenshots/
```

### Advanced Options

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --reflection-model anthropic/claude-3-opus \
  --prompt "Extract UI elements" \
  --dataset data/ui_dataset.json \
  --max-iterations 30 \
  --max-metric-calls 200 \
  --batch-size 6 \
  --output results/optimization_results.json \
  --verbose
```

## Command Line Arguments

### Required Arguments

- `--prompt`: Initial seed prompt to optimize
- `--dataset`: Path to dataset file or directory

### Model Configuration

- `--model`: Model specification (e.g., 'openai/gpt-4o')
- `--reflection-model`: Reflection model specification
- `--config`: Path to configuration JSON file

### Optimization Parameters

- `--max-iterations`: Maximum optimization iterations (default: 10)
- `--max-metric-calls`: Maximum metric evaluation calls (default: 100)
- `--batch-size`: Batch size for evaluation (default: 4)

### Output Options

- `--output`: Output file path for results (default: stdout)
- `--verbose`: Enable verbose logging

## Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export HUGGINGFACE_API_KEY="your-hf-key"
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
HUGGINGFACE_API_KEY=your-hf-key
```

## Dataset Format

Your dataset should be a JSON file or directory containing JSON files with the following structure:

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

## Output Format

The CLI outputs results in JSON format:

```json
{
  "optimized_prompt": "You are an expert UI element extractor...",
  "original_prompt": "Extract UI elements from this screenshot",
  "improvement_metrics": {
    "improvement_percent": 23.5,
    "best_score": 0.85,
    "baseline_score": 0.70
  },
  "optimization_time": 45.2,
  "status": "completed",
  "session_id": "opt_1234567890_12345"
}
```

## Examples

### UI Tree Extraction

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --prompt "Extract UI elements from this screenshot" \
  --dataset data/ui_screenshots.json \
  --max-iterations 25 \
  --max-metric-calls 150
```

### Text Analysis

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --prompt "Analyze the sentiment of this text" \
  --dataset data/text_samples.json \
  --max-iterations 15 \
  --max-metric-calls 75
```

### Code Generation

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --reflection-model anthropic/claude-3-opus \
  --prompt "Generate Python code for this task" \
  --dataset data/code_examples.json \
  --max-iterations 30 \
  --max-metric-calls 200
```

### Google Gemini Optimization

```bash
gepa-optimize \
  --model google/gemini-1.5-pro \
  --reflection-model google/gemini-1.5-flash \
  --prompt "Extract UI elements from this screenshot" \
  --dataset data/ui_screenshots.json \
  --max-iterations 20 \
  --max-metric-calls 100
```

### Mixed Provider Optimization

```bash
gepa-optimize \
  --model openai/gpt-4o \
  --reflection-model google/gemini-1.5-flash \
  --prompt "Analyze this interface layout" \
  --dataset data/interface_data.json \
  --max-iterations 15 \
  --max-metric-calls 75
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure your API keys are set in environment variables
2. **Invalid Dataset**: Check that your dataset file exists and has the correct format
3. **Model Not Found**: Verify the model name and provider are correct
4. **Out of Memory**: Reduce batch size or use a smaller dataset

### Debug Mode

Use `--verbose` flag for detailed logging:

```bash
gepa-optimize --verbose --model openai/gpt-4o --prompt "test" --dataset data.json
```

### Help

Get help with:

```bash
gepa-optimize --help
```
