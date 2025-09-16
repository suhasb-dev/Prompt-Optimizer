# Customer Service Prompt Optimization Test Guide

## 🎯 Overview

This test case demonstrates the **universal capabilities** of the GEPA Optimizer library by optimizing customer service prompts using real business data and custom evaluation metrics.

## 📊 What This Test Demonstrates

### 1. **Universal Adapter in Action**
- Uses the universal adapter (not UI-specific)
- Works with text-only data (no images)
- Custom evaluation metrics for business needs

### 2. **Real Business Application**
- Customer service is a critical business function
- Measurable impact on customer satisfaction
- Clear ROI demonstration for prompt optimization

### 3. **Custom Evaluation Metrics**
- **Helpfulness**: How helpful is the response
- **Empathy**: Shows understanding and care
- **Solution Focus**: Provides actionable solutions
- **Professionalism**: Tone and language quality

## 🚀 Test Options

### Option 1: Full Dataset Test (Recommended)
**File**: `test_customer_service_optimization.py`

**Requirements**:
- Download the CSV file: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- Place it in the project root directory
- Set `OPENAI_API_KEY` in your `.env` file

**Features**:
- Uses real 27K customer service interactions
- Comprehensive evaluation metrics
- Full optimization with reflection analysis
- Demonstrates production-ready capabilities

### Option 2: Simple Test (Quick Demo)
**File**: `test_customer_service_simple.py`

**Requirements**:
- Only needs `OPENAI_API_KEY` in your `.env` file
- No external dataset required

**Features**:
- Uses built-in sample data (5 interactions)
- Simplified evaluation metrics
- Quick demonstration of universal capabilities
- Perfect for testing and development

## 📋 Running the Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install gepa-optimizer pandas python-dotenv
   ```

2. **Set API Key**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```

### Run Simple Test (Recommended First)

```bash
python test_customer_service_simple.py
```

**Expected Output**:
```
🚀 Simple Customer Service Prompt Optimization Test
============================================================
🔑 API key loaded: sk-proj-...8IAA
✅ Created sample dataset with 5 customer service interactions
✅ Created custom customer service evaluator
✅ Created LLM client
✅ Created optimization configuration
✅ Created optimizer with universal adapter

🌱 SEED PROMPT:
----------------------------------------
"You are a customer service agent. Answer customer questions politely."
----------------------------------------

🚀 Starting optimization...

✅ Optimization completed!
   - Status: completed
   - Iterations: 3
   - Time: 45.23s

================================================================================
📝 PROMPT COMPARISON
================================================================================

🌱 SEED PROMPT:
----------------------------------------
"You are a customer service agent. Answer customer questions politely."
----------------------------------------

🚀 OPTIMIZED PROMPT:
----------------------------------------
"You are a professional customer service representative dedicated to providing exceptional support. When responding to customer inquiries, follow these guidelines:

1. **Show Empathy**: Acknowledge the customer's concern and express understanding of their situation.

2. **Be Solution-Focused**: Provide clear, actionable steps to resolve their issue. Offer specific solutions and alternatives when possible.

3. **Maintain Professionalism**: Use polite, respectful language and maintain a helpful tone throughout the interaction.

4. **Be Helpful**: Ensure your response directly addresses the customer's question or concern with relevant information and next steps.

5. **Take Ownership**: Use phrases like 'I'll help you with that' and 'Let me assist you' to show you're taking responsibility for resolving their issue.

Remember to always end with a question or next step to keep the conversation moving forward and ensure the customer's needs are fully addressed."
----------------------------------------

📊 IMPROVEMENT ANALYSIS:
----------------------------------------
   - Baseline Score: 0.3245
   - Final Score: 0.6789
   - Improvement Percent: 109.2%
================================================================================

🎉 Customer Service Optimization Test PASSED!

📝 What this demonstrates:
   ✅ Universal adapter works with customer service use case
   ✅ Custom evaluation metrics (helpfulness, professionalism, empathy)
   ✅ Real business scenario optimization
   ✅ Library is truly universal and extensible
```

### Run Full Dataset Test

1. **Download Dataset**:
   - Get the CSV file from the provided source
   - Place it in the project root as `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

2. **Run Test**:
   ```bash
   python test_customer_service_optimization.py
   ```

## 🔍 What You'll See

### 1. **Enhanced Logging**
The universal adapter provides detailed logging:

```
🚀 NEW PROPOSED CANDIDATE (Iteration 1)
================================================================================
📝 PROPOSED PROMPT:
----------------------------------------
"You are a helpful customer service agent..."
----------------------------------------
📊 Prompt Length: 45 characters
📊 Word Count: 8 words
================================================================================

🔍 REFLECTION DATASET CREATION
================================================================================
📋 CURRENT PROMPT BEING ANALYZED:
----------------------------------------
"You are a helpful customer service agent..."
----------------------------------------

📊 EVALUATION SUMMARY:
----------------------------------------
   • Average Score: 0.3245
   • Min Score: 0.2100
   • Max Score: 0.4500
   • Total Samples: 5

🎯 COMPONENTS TO UPDATE:
----------------------------------------
   1. system_prompt

🔍 DETAILED ANALYSIS:
----------------------------------------
   📝 Sample 1 (Score: 0.3200):
      Input: "Customer query: I need to cancel my order"
      Output: "I understand you'd like to cancel your order..."
      Detailed Scores:
        • Helpfulness: 0.4000
        • Professionalism: 0.3000
        • Empathy: 0.2600
      Feedback: "The overall quality is moderate. Professionalism is low. Focus on improving this aspect."
```

### 2. **Measurable Improvements**
- **Baseline Score**: Initial prompt performance
- **Final Score**: Optimized prompt performance
- **Improvement Percentage**: Quantified improvement
- **Detailed Metrics**: Breakdown by evaluation criteria

### 3. **Business-Ready Prompts**
The optimized prompts will be:
- More detailed and specific
- Include actionable guidelines
- Focus on customer satisfaction metrics
- Professional and empathetic tone

## 🎯 Expected Results

### Before Optimization
```
Seed Prompt: "You are a customer service agent. Answer customer questions politely."
```

### After Optimization
```
Optimized Prompt: "You are a professional customer service representative dedicated to providing exceptional support. When responding to customer inquiries, follow these guidelines:

1. Show Empathy: Acknowledge the customer's concern and express understanding...
2. Be Solution-Focused: Provide clear, actionable steps to resolve their issue...
3. Maintain Professionalism: Use polite, respectful language...
4. Be Helpful: Ensure your response directly addresses the customer's question...
5. Take Ownership: Use phrases like 'I'll help you with that'...

Remember to always end with a question or next step to keep the conversation moving forward..."
```

## 🔧 Customization Options

### 1. **Custom Evaluation Metrics**
Modify the evaluator to focus on your specific business needs:

```python
class CustomCustomerServiceEvaluator(BaseEvaluator):
    def __init__(self, metric_weights: Dict[str, float] = None):
        # Define your custom metrics
        default_weights = {
            "response_time": 0.2,      # Speed of resolution
            "customer_satisfaction": 0.3,  # Customer happiness
            "first_call_resolution": 0.3,  # Resolve in one interaction
            "upselling": 0.2           # Cross-sell opportunities
        }
        super().__init__(metric_weights or default_weights)
```

### 2. **Different LLM Models**
Test with different models for cost/quality trade-offs:

```python
# Use GPT-3.5-turbo for cost efficiency
config = OptimizationConfig(
    model="openai/gpt-3.5-turbo",
    reflection_model="openai/gpt-3.5-turbo",
    max_iterations=10
)

# Use GPT-4o for higher quality
config = OptimizationConfig(
    model="openai/gpt-4o",
    reflection_model="openai/gpt-4o",
    max_iterations=5
)
```

### 3. **Different Dataset Sizes**
Adjust the sample size based on your needs:

```python
# Small test (fast, cheap)
dataset = load_customer_service_dataset(csv_path, sample_size=20)

# Medium test (balanced)
dataset = load_customer_service_dataset(csv_path, sample_size=100)

# Large test (comprehensive, expensive)
dataset = load_customer_service_dataset(csv_path, sample_size=500)
```

## 🎉 Success Criteria

The test is successful if you see:

1. ✅ **Universal Adapter Working**: No UI-specific code, works with text data
2. ✅ **Custom Evaluation**: Your business-specific metrics are being used
3. ✅ **Measurable Improvement**: Clear improvement in evaluation scores
4. ✅ **Enhanced Logging**: Detailed reflection analysis and candidate tracking
5. ✅ **Business-Ready Output**: Professional, detailed optimized prompts

## 🚀 Next Steps

After running this test successfully:

1. **Try Different Use Cases**: Apply the same pattern to other business scenarios
2. **Customize Metrics**: Define evaluation metrics for your specific needs
3. **Scale Up**: Use larger datasets for production optimization
4. **Integrate**: Use the optimized prompts in your customer service system

This test case perfectly demonstrates that your GEPA Optimizer library is now **truly universal** and can handle any business use case, not just UI tree extraction!
