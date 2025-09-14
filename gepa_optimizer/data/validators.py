"""
Data validation utilities for GEPA optimizer
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates datasets for completeness and GEPA compatibility
    """
    
    def __init__(self):
        self.required_fields = ['input', 'output']
        self.optional_fields = ['metadata', 'id', 'tags']
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate entire dataset
        
        Args:
            dataset: List of data items to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic dataset checks
        if not dataset:
            errors.append("Dataset is empty")
            return False, errors
        
        if not isinstance(dataset, list):
            errors.append("Dataset must be a list")
            return False, errors
        
        # Validate each item
        for idx, item in enumerate(dataset):
            item_errors = self.validate_item(item, idx)
            errors.extend(item_errors)
        
        # Check for minimum dataset size
        if len(dataset) < 2:
            errors.append("Dataset should have at least 2 items for proper train/val split")
        
        # Log validation results
        if errors:
            logger.warning(f"Dataset validation failed with {len(errors)} errors")
        else:
            logger.info(f"Dataset validation passed for {len(dataset)} items")
        
        return len(errors) == 0, errors
    
    def validate_item(self, item: Dict[str, Any], index: Optional[int] = None) -> List[str]:
        """
        Validate a single dataset item
        
        Args:
            item: Single data item to validate
            index: Optional item index for error reporting
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        item_ref = f"item {index}" if index is not None else "item"
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            errors.append(f"{item_ref}: Must be a dictionary")
            return errors
        
        # Check for required fields
        if 'input' not in item:
            errors.append(f"{item_ref}: Missing required 'input' field")
        elif not isinstance(item['input'], str):
            errors.append(f"{item_ref}: 'input' field must be a string")
        elif not item['input'].strip():
            errors.append(f"{item_ref}: 'input' field cannot be empty")
        
        # Check output field (can be empty but should exist for supervised learning)
        if 'output' in item:
            if not isinstance(item['output'], str):
                errors.append(f"{item_ref}: 'output' field must be a string")
        
        # Validate metadata if present
        if 'metadata' in item and not isinstance(item['metadata'], dict):
            errors.append(f"{item_ref}: 'metadata' field must be a dictionary")
        
        return errors
    
    def validate_gepa_format(self, gepa_data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate data in GEPA format
        
        Args:
            gepa_data: Data in GEPA format
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if not gepa_data:
            errors.append("GEPA dataset is empty")
            return False, errors
        
        for idx, item in enumerate(gepa_data):
            if 'input' not in item:
                errors.append(f"GEPA item {idx}: Missing 'input' field")
            
            if 'expected_output' not in item:
                errors.append(f"GEPA item {idx}: Missing 'expected_output' field")
            
            if 'metadata' not in item:
                errors.append(f"GEPA item {idx}: Missing 'metadata' field")
            elif not isinstance(item['metadata'], dict):
                errors.append(f"GEPA item {idx}: 'metadata' must be a dictionary")
        
        return len(errors) == 0, errors
    
    def validate_split(self, trainset: List[Dict], valset: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate train/validation split
        
        Args:
            trainset: Training data
            valset: Validation data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if not trainset:
            errors.append("Training set is empty")
        
        if not valset:
            errors.append("Validation set is empty")
        
        # Check proportions
        total_size = len(trainset) + len(valset)
        if total_size > 0:
            train_ratio = len(trainset) / total_size
            if train_ratio < 0.5:
                errors.append(f"Training set too small: {train_ratio:.2%} of total data")
            elif train_ratio > 0.95:
                errors.append(f"Validation set too small: {1-train_ratio:.2%} of total data")
        
        return len(errors) == 0, errors
    
    def get_dataset_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the dataset
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dict[str, Any]: Dataset statistics
        """
        if not dataset:
            return {'total_items': 0, 'valid': False}
        
        stats = {
            'total_items': len(dataset),
            'has_output': sum(1 for item in dataset if item.get('output')),
            'avg_input_length': 0,
            'avg_output_length': 0,
            'empty_inputs': 0,
            'empty_outputs': 0
        }
        
        input_lengths = []
        output_lengths = []
        
        for item in dataset:
            if isinstance(item, dict):
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                if isinstance(input_text, str):
                    input_lengths.append(len(input_text))
                    if not input_text.strip():
                        stats['empty_inputs'] += 1
                
                if isinstance(output_text, str):
                    output_lengths.append(len(output_text))
                    if not output_text.strip():
                        stats['empty_outputs'] += 1
        
        if input_lengths:
            stats['avg_input_length'] = sum(input_lengths) / len(input_lengths)
        
        if output_lengths:
            stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
        
        # Determine if dataset looks valid
        stats['valid'] = (
            stats['total_items'] > 0 and
            stats['empty_inputs'] < stats['total_items'] * 0.5  # Less than 50% empty inputs
        )
        
        return stats
