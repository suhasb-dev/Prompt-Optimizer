# GEPA Universal Prompt Optimizer - Complete Project Analysis

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure & Locations](#file-structure--locations)
3. [Core Components Analysis](#core-components-analysis)
4. [Data Pipeline Analysis](#data-pipeline-analysis)
5. [Evaluation System Analysis](#evaluation-system-analysis)
6. [Models & Configuration Analysis](#models--configuration-analysis)
7. [Utilities Analysis](#utilities-analysis)
8. [Integration Points](#integration-points)
9. [Data Flow & Dependencies](#data-flow--dependencies)

## 🎯 Project Overview

**GEPA Universal Prompt Optimizer** is a sophisticated Python library that implements the GEPA (Generative Evaluation and Prompt Adaptation) framework for automatic prompt optimization. The project focuses on UI tree extraction optimization using vision-capable LLMs with comprehensive evaluation metrics.

### Key Architecture Principles
- **Modular Design**: Each component has a single responsibility
- **Async-First**: Built for high-performance async operations
- **Provider Agnostic**: Works with any LLM provider
- **Extensible**: Easy to add new evaluation metrics and adapters
- **Production Ready**: Comprehensive error handling and logging

## 📁 File Structure & Locations

### Root Directory Structure
```
gepa-optimizer/
├── gepa_optimizer/              # Main package directory
├── tests/                       # Test suite
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── screenshots/                 # UI screenshots for testing
├── json_tree/                   # Ground truth JSON files
├── setup.py                     # Package installation script
├── pyproject.toml              # Modern Python packaging
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # Project documentation
```

## 🔧 Core Components Analysis

### 1. Main Package Entry Point
**Location**: `gepa_optimizer/__init__.py`

**Purpose**: Defines the public API and exports all user-facing classes and functions.

**Key Exports**:
- `GepaOptimizer`: Main optimization class
- `OptimizationConfig`: Configuration management
- `OptimizedResult`: User-facing result object
- `ModelConfig`: LLM provider configuration
- `VisionLLMClient`: Multi-modal LLM interface
- `UITreeEvaluator`: UI-specific evaluation metrics

**Implementation Details**:
```python
# Key variables and their purposes:
__version__ = "0.1.0"                    # Package version
__all__ = [...]                          # Public API exports
```

**Linked To**: All other modules in the package (imports everything)

---

### 2. Core Optimization Engine
**Location**: `gepa_optimizer/core/optimizer.py`

**Purpose**: Main orchestrator class that provides the primary interface for prompt optimization.

**Key Classes**:
- `GepaOptimizer`: Main optimization class

**Key Methods**:
- `__init__(config, metric_weights)`: Initializes optimizer with configuration
- `train(seed_prompt, dataset, **kwargs)`: Main async optimization method
- `_update_config_from_kwargs(kwargs)`: Updates config with runtime overrides
- `_validate_inputs(seed_prompt)`: Validates input parameters
- `_create_seed_candidate(seed_prompt)`: Creates GEPA-compatible seed candidate
- `_run_gepa_optimization(**gepa_params)`: Executes GEPA optimization

**Key Variables**:
- `self.config: OptimizationConfig`: Required configuration
- `self.converter: UniversalConverter`: Data format converter
- `self.custom_adapter: CustomGepaAdapter`: GEPA framework bridge
- `self.api_manager: APIKeyManager`: API key management
- `self.result_processor: ResultProcessor`: Result processing
- `self.logger: Logger`: Logging instance

**Implementation Details**:
- **Async/Await Pattern**: Uses asyncio for non-blocking operations
- **Error Handling**: Comprehensive exception handling with custom exceptions
- **Configuration Management**: Flexible config system with runtime overrides
- **Result Processing**: Converts raw GEPA results to user-friendly format

**Linked To**: 
- `core/custom_adapter.py` (uses CustomGepaAdapter)
- `data/converters.py` (uses UniversalConverter)
- `models/config.py` (uses OptimizationConfig)
- `core/result.py` (uses ResultProcessor)

---

### 3. GEPA Framework Integration
**Location**: `gepa_optimizer/core/custom_adapter.py`

**Purpose**: Implements the GEPAAdapter protocol to bridge between GEPA optimization engine and UI tree extraction task.

**Key Classes**:
- `CustomGepaAdapter`: Implements GEPAAdapter interface

**Key Methods**:
- `__init__(model_config, metric_weights)`: Initializes adapter with model config
- `evaluate(batch, candidate, capture_traces)`: Evaluates prompt candidates
- `make_reflective_dataset(candidate, eval_batch, components)`: Creates reflection data
- `_parse_json_safely(json_str)`: Robust JSON parsing with repair
- `_repair_json(json_str)`: Attempts to fix malformed JSON
- `_generate_feedback(evaluation_results)`: Generates textual feedback

**Key Variables**:
- `self.vision_llm_client: VisionLLMClient`: Handles LLM API calls
- `self.ui_tree_evaluator: UITreeEvaluator`: Evaluates UI tree quality
- `self.logger: Logger`: Logging instance

**Implementation Details**:
- **JSON Repair System**: Handles malformed LLM outputs
- **Debug Logging**: Comprehensive logging for troubleshooting
- **Error Recovery**: Graceful handling of API failures
- **Feedback Generation**: Intelligent feedback based on evaluation metrics

**Linked To**:
- `llms/vision_llm.py` (uses VisionLLMClient)
- `evaluation/ui_evaluator.py` (uses UITreeEvaluator)
- `models/config.py` (uses ModelConfig)

---

### 4. Result Processing
**Location**: `gepa_optimizer/core/result.py`

**Purpose**: Processes raw GEPA optimization results into clean, usable formats.

**Key Classes**:
- `ResultProcessor`: Processes optimization results
- `OptimizationResult`: Complete optimization result with metadata
- `OptimizedResult`: User-facing result class

**Key Methods**:
- `extract_optimized_prompt(result)`: Extracts optimized prompt from GEPA result
- `extract_metrics(result)`: Extracts performance metrics
- `extract_reflection_history(result)`: Extracts optimization history

**Key Variables**:
- `session_id: UUID`: Unique session identifier
- `original_prompt: str`: Initial seed prompt
- `optimized_prompt: str`: Final optimized prompt
- `improvement_data: Dict`: Performance improvement metrics
- `optimization_time: float`: Time taken for optimization

**Implementation Details**:
- **Result Extraction**: Robust extraction from GEPA result objects
- **Metric Calculation**: Computes improvement percentages and scores
- **History Tracking**: Maintains optimization iteration history
- **Error Handling**: Graceful handling of missing or malformed results

**Linked To**:
- `models/result.py` (defines result models)
- `core/optimizer.py` (used by GepaOptimizer)

---

## 📊 Data Pipeline Analysis

### 1. Universal Data Converter
**Location**: `gepa_optimizer/data/converters.py`

**Purpose**: Converts various dataset formats into standardized GEPA-compatible format.

**Key Classes**:
- `UniversalConverter`: Main conversion class

**Key Methods**:
- `convert(dataset)`: Main conversion method
- `_standardize(data)`: Standardizes data into consistent format
- `_detect_format(source)`: Detects input data format
- `_split_dataset(data, train_ratio)`: Splits data into train/validation

**Key Variables**:
- `self.data_loader: DataLoader`: File loading utility
- `self.logger: Logger`: Logging instance

**Supported Formats**:
- **Structured**: CSV, JSON, JSONL
- **Text**: TXT, MD
- **Images**: PNG, JPG, JPEG
- **DataFrames**: Pandas DataFrame objects
- **UI Tree Datasets**: Special format for UI extraction

**Implementation Details**:
- **Format Detection**: Automatic format detection from file extensions
- **Data Standardization**: Consistent input/output/image format
- **Train/Val Split**: Automatic dataset splitting
- **Error Handling**: Comprehensive error handling for malformed data

**Linked To**:
- `data/loaders.py` (uses DataLoader)
- `core/optimizer.py` (used by GepaOptimizer)

---

### 2. Data Loading Utilities
**Location**: `gepa_optimizer/data/loaders.py`

**Purpose**: Handles loading data from various file formats.

**Key Classes**:
- `DataLoader`: Universal file loading class

**Key Methods**:
- `load(source, format_hint)`: Universal loading method
- `load_image_base64(path)`: Loads and encodes images as Base64
- `load_csv(path)`: Loads CSV files
- `load_json(path)`: Loads JSON files
- `load_jsonl(path)`: Loads JSONL files
- `load_ui_tree_dataset(json_dir, screenshots_dir)`: Special UI dataset loader

**Key Variables**:
- `self.logger: Logger`: Logging instance

**Supported File Types**:
- **CSV**: Loaded as pandas DataFrame
- **JSON**: Loaded as Python objects
- **JSONL**: Loaded as list of dictionaries
- **Text/Markdown**: Loaded as strings
- **Excel**: Loaded as pandas DataFrame
- **Images**: Loaded and encoded as Base64 strings

**Implementation Details**:
- **Base64 Encoding**: Automatic image encoding for LLM consumption
- **File Validation**: Checks file existence and format
- **Error Recovery**: Graceful handling of corrupted files
- **Memory Efficiency**: Streaming for large files

**Linked To**:
- `data/converters.py` (used by UniversalConverter)
- `core/custom_adapter.py` (for UI tree datasets)

---

### 3. Data Validation
**Location**: `gepa_optimizer/data/validators.py`

**Purpose**: Validates data integrity and format compliance.

**Key Classes**:
- `DataValidator`: Data validation utility

**Key Methods**:
- `validate_dataset(dataset)`: Validates dataset structure
- `validate_ui_tree_data(data)`: Validates UI tree specific data
- `validate_image_data(image_data)`: Validates image data

**Implementation Details**:
- **Schema Validation**: Ensures data follows expected format
- **Type Checking**: Validates data types and structures
- **Completeness Check**: Ensures required fields are present

**Linked To**:
- `data/converters.py` (used by UniversalConverter)

---

## 📈 Evaluation System Analysis

### 1. UI Tree Evaluator
**Location**: `gepa_optimizer/evaluation/ui_evaluator.py`

**Purpose**: Comprehensive evaluator for UI tree extraction quality with multiple metrics.

**Key Classes**:
- `UITreeEvaluator`: UI-specific evaluation class

**Key Methods**:
- `evaluate(predicted_json, expected_json)`: Main evaluation method
- `calculate_element_completeness(predicted, expected)`: Element count comparison
- `calculate_element_type_accuracy(predicted, expected)`: Type accuracy
- `calculate_text_content_accuracy(predicted, expected)`: Text similarity
- `calculate_hierarchy_accuracy(predicted, expected)`: Structure accuracy
- `calculate_style_accuracy(predicted, expected)`: Style property accuracy

**Key Variables**:
- `self.metric_weights: Dict[str, float]`: Weights for each metric
- `self.logger: Logger`: Logging instance

**Evaluation Metrics**:
1. **Element Completeness (Weight: 0.3)**: Compares total element counts
2. **Element Type Accuracy (Weight: 0.25)**: Compares element types
3. **Text Content Accuracy (Weight: 0.2)**: Compares text content using difflib
4. **Hierarchy Accuracy (Weight: 0.15)**: Compares parent-child relationships
5. **Style Accuracy (Weight: 0.1)**: Compares styling properties

**Implementation Details**:
- **Weighted Scoring**: Each metric has configurable weights
- **Composite Score**: Final score combines all metrics
- **Debug Logging**: Detailed logging of individual metric scores
- **Error Handling**: Graceful handling of malformed JSON

**Linked To**:
- `core/custom_adapter.py` (used by CustomGepaAdapter)

---

## 🤖 Models & Configuration Analysis

### 1. Configuration Management
**Location**: `gepa_optimizer/models/config.py`

**Purpose**: Manages configuration for optimization process and LLM providers.

**Key Classes**:
- `ModelConfig`: LLM provider configuration
- `OptimizationConfig`: Main optimization configuration

**ModelConfig Key Attributes**:
- `provider: str`: LLM provider name (openai, anthropic, etc.)
- `model_name: str`: Specific model name
- `api_key: str`: API key for the provider
- `base_url: Optional[str]`: Custom endpoint URL
- `temperature: float`: Generation temperature
- `max_tokens: int`: Maximum tokens to generate

**OptimizationConfig Key Attributes**:
- `model: Union[str, ModelConfig]`: Main model configuration
- `reflection_model: Union[str, ModelConfig]`: Reflection model
- `max_iterations: int`: Maximum optimization iterations
- `max_metric_calls: int`: Maximum evaluation calls
- `batch_size: int`: Evaluation batch size
- `early_stopping: bool`: Enable early stopping
- `learning_rate: float`: Optimization learning rate

**Key Methods**:
- `validate_api_connectivity()`: Tests API connectivity
- `get_estimated_cost()`: Estimates optimization cost
- `create_example_config(provider)`: Generates example configs

**Implementation Details**:
- **Flexible Configuration**: Supports both string and object configs
- **Environment Variables**: Automatic API key discovery
- **Validation**: Runtime validation of all parameters
- **Cost Estimation**: Built-in cost calculation

**Linked To**:
- `core/optimizer.py` (used by GepaOptimizer)
- `core/custom_adapter.py` (used by CustomGepaAdapter)
- `utils/api_keys.py` (for API key management)

---

### 2. Dataset Models
**Location**: `gepa_optimizer/models/dataset.py`

**Purpose**: Defines data structures for datasets and samples.

**Key Classes**:
- `DatasetSample`: Individual dataset sample
- `Dataset`: Complete dataset container

**Key Attributes**:
- `input: str`: Input text or prompt
- `output: str`: Expected output
- `image_base64: Optional[str]`: Base64 encoded image
- `metadata: Optional[Dict]`: Additional metadata

**Implementation Details**:
- **Type Safety**: Uses dataclasses for type safety
- **Validation**: Built-in validation for required fields
- **Flexibility**: Supports optional fields for different use cases

**Linked To**:
- `data/converters.py` (uses dataset models)

---

### 3. Result Models
**Location**: `gepa_optimizer/models/result.py`

**Purpose**: Defines result data structures for optimization outcomes.

**Key Classes**:
- `OptimizationResult`: Complete optimization result
- `OptimizedResult`: User-facing result class

**OptimizationResult Key Attributes**:
- `session_id: UUID`: Unique session identifier
- `original_prompt: str`: Initial seed prompt
- `optimized_prompt: str`: Final optimized prompt
- `improvement_data: Dict`: Performance metrics
- `optimization_time: float`: Time taken
- `status: str`: Success/failure status
- `error_message: Optional[str]`: Error details if failed

**OptimizedResult Key Properties**:
- `prompt`: The optimized prompt (read-only)
- `original_prompt`: Original prompt (read-only)
- `improvement_percent`: Percentage improvement
- `is_successful`: Boolean success indicator

**Key Methods**:
- `get_improvement_summary()`: Summary of improvements
- `get_reflection_summary()`: Summary of reflection process
- `get_detailed_result()`: Full detailed result

**Implementation Details**:
- **Immutable Results**: Results are read-only after creation
- **Rich Metadata**: Comprehensive metadata for analysis
- **Error Handling**: Graceful error state representation

**Linked To**:
- `core/result.py` (uses result models)
- `core/optimizer.py` (returns OptimizedResult)

---

## 🛠️ Utilities Analysis

### 1. API Key Management
**Location**: `gepa_optimizer/utils/api_keys.py`

**Purpose**: Secure management of API keys without hardcoding.

**Key Classes**:
- `APIKeyManager`: API key management utility

**Key Methods**:
- `get_openai_key()`: Returns OpenAI API key
- `get_anthropic_key()`: Returns Anthropic API key
- `get_google_key()`: Returns Google API key
- `set_api_key(provider, key)`: Sets API key at runtime
- `has_required_keys()`: Validates required keys
- `validate_keys()`: Tests API key validity

**Key Variables**:
- `_api_keys: Dict[str, str]`: Internal key storage
- `_env_loaded: bool`: Environment loading flag

**Implementation Details**:
- **Environment Variables**: Loads from standard env vars
- **.env File Support**: Automatic .env file loading
- **Runtime Setting**: Ability to set keys programmatically
- **Validation**: Tests key validity with API calls
- **Security**: Keys never logged or stored in plain text

**Linked To**:
- `models/config.py` (used by ModelConfig)
- `llms/vision_llm.py` (used by VisionLLMClient)

---

### 2. Exception Handling
**Location**: `gepa_optimizer/utils/exceptions.py`

**Purpose**: Defines custom exception hierarchy for better error handling.

**Key Classes**:
- `GepaOptimizerError`: Base exception for all library errors
- `GepaDependencyError`: GEPA library dependency issues
- `InvalidInputError`: Invalid user input validation
- `DatasetError`: Dataset-related errors
- `APIError`: API-related errors
- `ConfigurationError`: Configuration-related errors

**Implementation Details**:
- **Hierarchical Structure**: Clear exception hierarchy
- **Rich Error Messages**: Detailed error information
- **Context Preservation**: Maintains error context
- **User-Friendly**: Clear error messages for users

**Linked To**:
- All modules (used throughout the codebase)

---

### 3. Helper Functions
**Location**: `gepa_optimizer/utils/helpers.py`

**Purpose**: Common utility functions used throughout the codebase.

**Key Functions**:
- `sanitize_prompt(prompt)`: Sanitizes and validates prompts
- `calculate_metrics(predicted, expected)`: Calculates basic metrics
- `setup_logging(level)`: Configures logging system
- `validate_model_name(model_name)`: Validates model names

**Implementation Details**:
- **Input Validation**: Robust input validation
- **Default Values**: Sensible defaults for missing inputs
- **Error Handling**: Graceful error handling
- **Reusability**: Functions designed for reuse

**Linked To**:
- `core/optimizer.py` (uses helper functions)
- `models/config.py` (uses validation functions)

---

### 4. Logging Utilities
**Location**: `gepa_optimizer/utils/logging.py`

**Purpose**: Centralized logging configuration and utilities.

**Key Functions**:
- `setup_logging(level)`: Configures logging system
- `get_logger(name)`: Gets logger instance
- `log_optimization_start(config)`: Logs optimization start
- `log_optimization_end(result)`: Logs optimization completion

**Implementation Details**:
- **Structured Logging**: JSON-formatted logs for production
- **Log Levels**: Configurable log levels
- **Performance Tracking**: Logs timing and performance metrics
- **Error Tracking**: Comprehensive error logging

**Linked To**:
- All modules (used throughout the codebase)

---

### 5. Metrics Utilities
**Location**: `gepa_optimizer/utils/metrics.py`

**Purpose**: Common metrics calculation utilities.

**Key Functions**:
- `calculate_similarity(text1, text2)`: Text similarity calculation
- `calculate_accuracy(predicted, expected)`: Basic accuracy calculation
- `calculate_composite_score(metrics, weights)`: Weighted composite scoring

**Implementation Details**:
- **Multiple Algorithms**: Various similarity algorithms
- **Configurable Weights**: Customizable metric weights
- **Performance Optimized**: Efficient implementations
- **Extensible**: Easy to add new metrics

**Linked To**:
- `evaluation/ui_evaluator.py` (uses metrics utilities)

---

## 🔗 Integration Points

### 1. LLM Integration
**Location**: `gepa_optimizer/llms/vision_llm.py`

**Purpose**: Client for interacting with multi-modal Vision LLMs.

**Key Classes**:
- `VisionLLMClient`: Multi-modal LLM client

**Key Methods**:
- `__init__(model_config)`: Initializes client with model config
- `generate(system_prompt, user_prompt, image_base64)`: Generates LLM response
- `_generate_openai(...)`: OpenAI-specific generation
- `_generate_google(...)`: Google Gemini-specific generation
- `_validate_response(response)`: Validates LLM response

**Key Variables**:
- `self.model_config: ModelConfig`: Model configuration
- `self.api_manager: APIKeyManager`: API key management
- `self.logger: Logger`: Logging instance
- `self.timeout: int`: Request timeout

**Supported Providers**:
- **OpenAI**: GPT-4V, GPT-4o, GPT-3.5-turbo
- **Google**: Gemini-1.5-Pro, Gemini-1.5-Flash
- **Anthropic**: Claude-3 (text-only)
- **Hugging Face**: Various models

**Implementation Details**:
- **Multi-Modal Support**: Handles both text and image inputs
- **Provider Abstraction**: Unified interface for different providers
- **Error Handling**: Comprehensive error handling for API failures
- **Response Validation**: Validates LLM responses
- **Timeout Management**: Configurable request timeouts

**Linked To**:
- `core/custom_adapter.py` (used by CustomGepaAdapter)
- `models/config.py` (uses ModelConfig)
- `utils/api_keys.py` (uses APIKeyManager)

---

### 2. CLI Integration
**Location**: `gepa_optimizer/cli.py`

**Purpose**: Command-line interface for the optimizer.

**Key Functions**:
- `main()`: Main CLI entry point
- `validate_api_keys()`: Validates API keys
- `output_results(result)`: Formats and outputs results
- `load_config(config_path)`: Loads configuration from file

**Implementation Details**:
- **Argument Parsing**: Uses argparse for CLI arguments
- **Configuration Loading**: Loads config from files
- **Result Formatting**: Pretty-prints optimization results
- **Error Handling**: User-friendly error messages

**Linked To**:
- `core/optimizer.py` (uses GepaOptimizer)
- `models/config.py` (uses OptimizationConfig)

---

## 🔄 Data Flow & Dependencies

### 1. Optimization Flow
```
User Input → GepaOptimizer → UniversalConverter → CustomGepaAdapter → VisionLLMClient → UITreeEvaluator → ResultProcessor → OptimizedResult
```

### 2. Key Dependencies
- **GEPA Library**: Core optimization framework
- **OpenAI API**: Primary LLM provider
- **Pandas**: Data manipulation
- **Pydantic**: Data validation
- **Python-dotenv**: Environment variable management
- **Requests/Aiohttp**: HTTP client libraries
- **Pillow**: Image processing

### 3. Configuration Flow
```
OptimizationConfig → ModelConfig → APIKeyManager → VisionLLMClient
```

### 4. Data Conversion Flow
```
Raw Dataset → DataLoader → UniversalConverter → GEPA Format → CustomGepaAdapter
```

### 5. Evaluation Flow
```
LLM Output → UITreeEvaluator → Metrics → Feedback → Reflection → New Prompt
```

## 🎯 Implementation Summary

### What Was Implemented
1. **Complete GEPA Integration**: Full implementation of GEPA framework
2. **Multi-Modal Support**: Vision LLM integration with image processing
3. **Comprehensive Evaluation**: UI-specific evaluation metrics
4. **Universal Data Pipeline**: Support for multiple data formats
5. **Production-Ready Architecture**: Error handling, logging, configuration
6. **CLI Interface**: Command-line tool for easy usage
7. **Extensible Design**: Easy to add new providers and metrics

### Why This Architecture
1. **Modularity**: Each component has a single responsibility
2. **Scalability**: Async-first design for high performance
3. **Flexibility**: Provider-agnostic design
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new features
6. **Production Ready**: Comprehensive error handling and logging

### How It Works
1. **User provides**: Seed prompt, dataset, configuration
2. **System converts**: Dataset to GEPA-compatible format
3. **GEPA optimizes**: Iteratively improves the prompt
4. **System evaluates**: Uses custom metrics to score prompts
5. **System reflects**: Generates feedback for improvement
6. **System returns**: Optimized prompt with performance metrics

This architecture provides a solid foundation for universal prompt optimization while maintaining the flexibility to extend to new use cases and providers.
