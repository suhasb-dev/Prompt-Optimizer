# 📦 Installation

This guide will help you install the GEPA Universal Prompt Optimizer and set up your development environment.

## Prerequisites

- **Python 3.8 or higher** - The library requires Python 3.8+
- **pip package manager** - For installing Python packages
- **API Key** - For your chosen LLM provider (OpenAI, Anthropic, Google, etc.)

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
pip install gepa-optimizer
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/suhasb-dev/Prompt-Optimizer.git
cd Prompt-Optimizer

# Install in development mode
pip install -e .
```

### Method 3: Install with Development Dependencies

```bash
# Clone the repository
git clone https://github.com/suhasb-dev/Prompt-Optimizer.git
cd Prompt-Optimizer

# Install with development dependencies
pip install -e ".[dev]"
```

## Verify Installation

Test your installation with this simple Python script:

```python
import gepa_optimizer

# Check version
print(f"GEPA Optimizer version: {gepa_optimizer.__version__}")

# Test imports
from gepa_optimizer import GepaOptimizer, OptimizationConfig
print("✅ Installation successful!")
```

## API Key Setup

### Option 1: Environment Variables (Recommended)

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google
export GOOGLE_API_KEY="your-google-api-key"

# For Hugging Face
export HUGGINGFACE_API_KEY="your-hf-api-key"
```

### Option 2: .env File

Create a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
HUGGINGFACE_API_KEY=your-hf-api-key
```

Then load it in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Direct Configuration

```python
from gepa_optimizer import ModelConfig

# Configure with API key directly
model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key-here"
)
```

## Dependencies

The library automatically installs these core dependencies:

- `gepa` - Base GEPA framework
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `google-generativeai` - Google AI client
- `huggingface-hub` - Hugging Face client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pydantic` - Data validation
- `python-dotenv` - Environment variable management

## Development Dependencies

For development and testing, install additional dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `coverage` - Test coverage

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Error
```
ERROR: Package 'gepa-optimizer' requires a different Python: 3.7.9 not in '>=3.8'
```
**Solution**: Upgrade to Python 3.8 or higher.

#### 2. Permission Denied
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```
**Solution**: Use `pip install --user gepa-optimizer` or create a virtual environment.

#### 3. Network Issues
```
ERROR: Could not find a version that satisfies the requirement
```
**Solution**: Check your internet connection and try again.

### Virtual Environment Setup

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv gepa-env

# Activate virtual environment
# On Windows:
gepa-env\Scripts\activate
# On macOS/Linux:
source gepa-env/bin/activate

# Install the library
pip install gepa-optimizer
```

## Next Steps

Once installation is complete:

1. **Set up your API keys** (see above)
2. **Go to [Quick Start](quick-start.md)** to run your first optimization
3. **Explore [Basic Usage](basic-usage.md)** for more detailed examples

## Support

If you encounter any installation issues:

- Check the [troubleshooting section](#troubleshooting) above
- Open an issue on [GitHub](https://github.com/suhasb-dev/Prompt-Optimizer/issues)
- Contact us at s8hasgrylls@gmail.com
