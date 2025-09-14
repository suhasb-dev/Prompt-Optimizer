"""
Helper functions for GEPA Optimizer
"""

def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize and validate prompt string
    
    Args:
        prompt: Input prompt string to sanitize
        
    Returns:
        str: Cleaned and validated prompt
    """
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    prompt = prompt.strip()
    
    if not prompt:
        prompt = "You are a helpful assistant."
    
    return prompt
