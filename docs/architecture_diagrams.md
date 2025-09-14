# GEPA Universal Prompt Optimizer - Architecture & Workflow Diagrams

## 1. High-Level Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[GepaOptimizer API]
        CONV[optimize_prompt Function]
    end
    
    subgraph "Core Processing Layer"
        ADAPTER[CustomGepaAdapter]
        PROCESSOR[ResultProcessor]
        CONVERTER[UniversalConverter]
    end
    
    subgraph "LLM & Evaluation Layer"
        LLM[VisionLLMClient]
        EVAL[UITreeEvaluator]
    end
    
    subgraph "Data Layer"
        LOADER[DataLoader]
        VALIDATOR[DataValidator]
    end
    
    subgraph "Configuration & Utils"
        CONFIG[ModelConfig/OptimizationConfig]
        KEYS[APIKeyManager]
        EXCEPT[Custom Exceptions]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        HF[HuggingFace API]
        VLLM[vLLM API]
    end
    
    UI --> ADAPTER
    CONV --> UI
    ADAPTER --> LLM
    ADAPTER --> EVAL
    ADAPTER --> PROCESSOR
    CONVERTER --> LOADER
    LOADER --> VALIDATOR
    LLM --> OPENAI
    LLM --> ANTHROPIC
    LLM --> HF
    LLM --> VLLM
    CONFIG --> LLM
    KEYS --> LLM
    EXCEPT --> LLM
    PROCESSOR --> EVAL
```

## 2. Detailed Workflow - Prompt Optimization Process

```mermaid
flowchart TD
    START([User Initiates Optimization]) --> VALIDATE_CONFIG{Validate Configuration}
    VALIDATE_CONFIG -->|Invalid| ERROR_CONFIG[Throw Configuration Error]
    VALIDATE_CONFIG -->|Valid| LOAD_DATA[Load Dataset]
    
    LOAD_DATA --> DETECT_FORMAT{Detect Data Format}
    DETECT_FORMAT --> CSV[Process CSV]
    DETECT_FORMAT --> JSON[Process JSON]
    DETECT_FORMAT --> JSONL[Process JSONL]
    DETECT_FORMAT --> TXT[Process Text]
    DETECT_FORMAT --> IMG[Process Images]
    
    CSV --> CONVERT[Universal Conversion]
    JSON --> CONVERT
    JSONL --> CONVERT
    TXT --> CONVERT
    IMG --> CONVERT
    
    CONVERT --> VALIDATE_DATA{Validate Data Structure}
    VALIDATE_DATA -->|Invalid| ERROR_DATA[Throw Data Error]
    VALIDATE_DATA -->|Valid| INIT_GEPA[Initialize GEPA Framework]
    
    INIT_GEPA --> CREATE_ADAPTER[Create CustomGepaAdapter]
    CREATE_ADAPTER --> SETUP_LLM[Setup VisionLLMClient]
    SETUP_LLM --> SETUP_EVAL[Setup UITreeEvaluator]
    
    SETUP_EVAL --> START_OPT[Start GEPA Optimization]
    START_OPT --> GEN_CANDIDATES[Generate Prompt Candidates]
    
    GEN_CANDIDATES --> EVAL_LOOP{Evaluation Loop}
    EVAL_LOOP --> CALL_LLM[Call Vision LLM]
    CALL_LLM --> RETRY_LOGIC{API Call Success?}
    RETRY_LOGIC -->|Fail| HANDLE_ERROR[Handle Error & Retry]
    HANDLE_ERROR --> CALL_LLM
    RETRY_LOGIC -->|Success| EXTRACT_RESPONSE[Extract LLM Response]
    
    EXTRACT_RESPONSE --> EVAL_METRICS[Calculate Evaluation Metrics]
    EVAL_METRICS --> STRUCT_SIM[Structural Similarity]
    EVAL_METRICS --> ELEM_ACC[Element Type Accuracy]
    EVAL_METRICS --> SPATIAL_ACC[Spatial Accuracy]
    EVAL_METRICS --> TEXT_ACC[Text Content Accuracy]
    EVAL_METRICS --> COMPLETE[Completeness Score]
    
    STRUCT_SIM --> AGGREGATE[Aggregate Scores]
    ELEM_ACC --> AGGREGATE
    SPATIAL_ACC --> AGGREGATE
    TEXT_ACC --> AGGREGATE
    COMPLETE --> AGGREGATE
    
    AGGREGATE --> CHECK_CONVERGENCE{Convergence Check}
    CHECK_CONVERGENCE -->|Not Converged| GEN_CANDIDATES
    CHECK_CONVERGENCE -->|Converged| PROCESS_RESULTS[Process Final Results]
    
    PROCESS_RESULTS --> FORMAT_OUTPUT[Format Output]
    FORMAT_OUTPUT --> RETURN_RESULT[Return OptimizedResult]
    RETURN_RESULT --> END([End])
    
    ERROR_CONFIG --> END
    ERROR_DATA --> END
```

## 3. Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Data Sources"
        CSV_FILE[CSV Files]
        JSON_FILE[JSON Files]
        JSONL_FILE[JSONL Files]
        TXT_FILE[Text Files]
        IMG_FILE[Image Files]
    end
    
    subgraph "Data Processing Pipeline"
        LOADER[DataLoader]
        CONVERTER[UniversalConverter]
        VALIDATOR[DataValidator]
    end
    
    subgraph "Core Processing"
        GEPA_ADAPTER[CustomGepaAdapter]
        OPTIMIZATION[GEPA Optimization Engine]
    end
    
    subgraph "LLM Processing"
        VISION_CLIENT[VisionLLMClient]
        API_CALLS[API Calls with Retry Logic]
    end
    
    subgraph "Evaluation Engine"
        UI_EVALUATOR[UITreeEvaluator]
        METRICS[Evaluation Metrics]
    end
    
    subgraph "Output Processing"
        RESULT_PROCESSOR[ResultProcessor]
        FORMATTED_OUTPUT[Formatted Results]
    end
    
    CSV_FILE --> LOADER
    JSON_FILE --> LOADER
    JSONL_FILE --> LOADER
    TXT_FILE --> LOADER
    IMG_FILE --> LOADER
    
    LOADER --> CONVERTER
    CONVERTER --> VALIDATOR
    VALIDATOR --> GEPA_ADAPTER
    
    GEPA_ADAPTER --> OPTIMIZATION
    OPTIMIZATION --> VISION_CLIENT
    VISION_CLIENT --> API_CALLS
    API_CALLS --> UI_EVALUATOR
    
    UI_EVALUATOR --> METRICS
    METRICS --> RESULT_PROCESSOR
    RESULT_PROCESSOR --> FORMATTED_OUTPUT
```

## 4. Class Relationship Diagram

```mermaid
classDiagram
    class GepaOptimizer {
        +optimize_async(config, dataset)
        +optimize_sync(config, dataset)
        -_validate_config()
        -_setup_components()
    }
    
    class CustomGepaAdapter {
        +evaluate_batch(batch)
        +get_model_response(prompt, data)
        -_process_evaluation()
    }
    
    class VisionLLMClient {
        +generate(system_prompt, user_prompt, image)
        +from_config(config)
        -_generate_openai()
        -_handle_error()
    }
    
    class UITreeEvaluator {
        +evaluate_ui_tree(predicted, reference)
        +calculate_structural_similarity()
        +calculate_element_accuracy()
        +calculate_spatial_accuracy()
    }
    
    class UniversalConverter {
        +convert_to_standard_format(data)
        -_detect_format()
        -_convert_csv()
        -_convert_json()
    }
    
    class ModelConfig {
        +provider: str
        +model_name: str
        +api_key: str
        +temperature: float
    }
    
    class OptimizationConfig {
        +model: ModelConfig
        +reflection_model: ModelConfig
        +dataset_path: str
        +max_iterations: int
    }
    
    class OptimizedResult {
        +optimized_prompt: str
        +evaluation_scores: dict
        +optimization_history: list
        +metadata: dict
    }
    
    GepaOptimizer --> CustomGepaAdapter
    GepaOptimizer --> OptimizationConfig
    CustomGepaAdapter --> VisionLLMClient
    CustomGepaAdapter --> UITreeEvaluator
    VisionLLMClient --> ModelConfig
    GepaOptimizer --> UniversalConverter
    GepaOptimizer --> OptimizedResult
    OptimizationConfig --> ModelConfig
```

## 5. Error Handling Flow

```mermaid
flowchart TD
    API_CALL[API Call to LLM] --> SUCCESS{Success?}
    SUCCESS -->|Yes| PROCESS[Process Response]
    SUCCESS -->|No| ERROR_TYPE{Error Type?}
    
    ERROR_TYPE -->|Rate Limit| RATE_LIMIT[Rate Limit Handler]
    ERROR_TYPE -->|Network| NETWORK[Network Error Handler]
    ERROR_TYPE -->|Validation| VALIDATION[Validation Error Handler]
    ERROR_TYPE -->|API Error| API_ERROR[API Error Handler]
    
    RATE_LIMIT --> BACKOFF[Exponential Backoff]
    NETWORK --> RETRY_LOGIC[Retry Logic]
    VALIDATION --> LOG_ERROR[Log Error]
    API_ERROR --> LOG_ERROR
    
    BACKOFF --> WAIT[Wait Period]
    WAIT --> CHECK_RETRIES{Max Retries?}
    RETRY_LOGIC --> CHECK_RETRIES
    
    CHECK_RETRIES -->|No| API_CALL
    CHECK_RETRIES -->|Yes| FAIL[Fail with Error]
    LOG_ERROR --> FAIL
    
    PROCESS --> SUCCESS_END[Success]
    FAIL --> ERROR_END[Error End]
```

## 6. Configuration Management Flow

```mermaid
graph TD
    START[Initialize Configuration] --> CHECK_ENV{Environment Variables?}
    CHECK_ENV -->|Found| LOAD_ENV[Load from Environment]
    CHECK_ENV -->|Not Found| USER_CONFIG[Use User Configuration]
    
    LOAD_ENV --> VALIDATE_CONFIG[Validate Configuration]
    USER_CONFIG --> VALIDATE_CONFIG
    
    VALIDATE_CONFIG --> CHECK_PROVIDER{Valid Provider?}
    CHECK_PROVIDER -->|No| ERROR[Configuration Error]
    CHECK_PROVIDER -->|Yes| CHECK_API_KEY{API Key Present?}
    
    CHECK_API_KEY -->|No| ERROR
    CHECK_API_KEY -->|Yes| CHECK_MODEL{Valid Model?}
    
    CHECK_MODEL -->|No| ERROR
    CHECK_MODEL -->|Yes| CREATE_CLIENT[Create LLM Client]
    
    CREATE_CLIENT --> READY[Configuration Ready]
    ERROR --> END[End with Error]
    READY --> END[End Successfully]
```
