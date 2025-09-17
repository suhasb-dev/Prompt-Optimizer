# 🤝 Contributing

Thank you for considering contributing to the GEPA Universal Prompt Optimizer project! We welcome bug reports, feature requests, and pull requests.

## 📋 What's in this section

- [Development Setup](development-setup.md) - Setting up your development environment
- [Testing](testing.md) - Running tests and validation
- [Code Style](code-style.md) - Coding standards and guidelines

## 🚀 Quick Start

### 1. **Fork the Repository**
- Go to [https://github.com/suhasb-dev/Prompt-Optimizer](https://github.com/suhasb-dev/Prompt-Optimizer)
- Click the "Fork" button
- Clone your fork locally

### 2. **Set Up Development Environment**
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Prompt-Optimizer.git
cd Prompt-Optimizer

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### 3. **Create a Feature Branch**
```bash
git checkout -b feature/amazing-feature
```

### 4. **Make Your Changes**
- Write your code
- Add tests
- Update documentation
- Follow code style guidelines

### 5. **Test Your Changes**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gepa_optimizer

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### 6. **Submit a Pull Request**
- Push your changes to your fork
- Create a pull request with a clear description
- Link any related issues

## 🎯 **Types of Contributions**

### 🐛 **Bug Reports**
- Use the GitHub issue template
- Provide clear reproduction steps
- Include system information
- Add relevant logs

### ✨ **Feature Requests**
- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Consider implementation complexity

### 🔧 **Code Contributions**
- Follow the coding standards
- Add comprehensive tests
- Update documentation
- Ensure backward compatibility

### 📚 **Documentation**
- Fix typos and grammar
- Improve clarity
- Add examples
- Update API documentation

## 🔧 **Development Setup**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git version control

### **Installation**
```bash
# Clone the repository
git clone https://github.com/suhasb-dev/Prompt-Optimizer.git
cd Prompt-Optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### **Development Dependencies**
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `coverage` - Test coverage
- `pre-commit` - Git hooks

## 🧪 **Testing**

### **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gepa_optimizer --cov-report=html

# Run specific test files
pytest tests/unit/test_optimizer.py
pytest tests/integration/test_customer_service.py

# Run with verbose output
pytest -v
```

### **Test Categories**
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### **Writing Tests**
```python
import pytest
from gepa_optimizer import GepaOptimizer, OptimizationConfig

def test_optimizer_initialization():
    """Test optimizer initialization"""
    config = OptimizationConfig(model="openai/gpt-3.5-turbo")
    optimizer = GepaOptimizer(config=config)
    assert optimizer is not None

@pytest.mark.asyncio
async def test_optimization_workflow():
    """Test complete optimization workflow"""
    # Your test implementation
    pass
```

## 📝 **Code Style**

### **Python Style Guide**
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Keep functions small and focused

### **Formatting**
```bash
# Format code with Black
black gepa_optimizer/

# Check formatting
black --check gepa_optimizer/

# Lint with flake8
flake8 gepa_optimizer/

# Type check with mypy
mypy gepa_optimizer/
```

### **Code Examples**
```python
from typing import Dict, List, Optional
from gepa_optimizer.evaluation import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    """Custom evaluator for specific use case.
    
    This evaluator implements custom metrics for domain-specific
    evaluation of prompt optimization results.
    
    Args:
        metric_weights: Dictionary of metric weights
    """
    
    def __init__(self, metric_weights: Optional[Dict[str, float]] = None):
        self.metric_weights = metric_weights or {"default": 1.0}
    
    def evaluate(self, predicted: str, expected: str) -> Dict[str, float]:
        """Evaluate the quality of a predicted response.
        
        Args:
            predicted: The predicted response
            expected: The expected response
            
        Returns:
            Dictionary of metric scores including composite_score
        """
        # Implementation here
        return {"composite_score": 0.8}
```

## 📋 **Pull Request Guidelines**

### **Before Submitting**
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Clear commit messages

### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎯 **Issue Guidelines**

### **Bug Reports**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9]
- Library version: [e.g., 0.1.0]

**Additional context**
Any other context about the problem.
```

### **Feature Requests**
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 🆘 **Getting Help**

- **GitHub Issues**: [Open an issue](https://github.com/suhasb-dev/Prompt-Optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/suhasb-dev/Prompt-Optimizer/discussions)
- **Email**: s8hasgrylls@gmail.com

## 🎉 **Recognition**

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

## 📄 **License**

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GEPA Universal Prompt Optimizer!** 🚀
