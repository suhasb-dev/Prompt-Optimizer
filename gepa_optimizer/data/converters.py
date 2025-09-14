"""
Universal converter for dataset to GEPA format
"""

import os
import json
from typing import Any, List, Tuple, Union , Dict
from pathlib import Path
import pandas as pd
import logging

from .loaders import DataLoader
from ..utils.exceptions import DatasetError
from typing import Dict

logger = logging.getLogger(__name__)

class UniversalConverter:
    def __init__(self):
        self.supported_extensions = [
            '.csv', '.json', '.jsonl', '.txt', '.md',
            '.png', '.jpg', '.jpeg'
        ]
        self.loader = DataLoader()

    def convert(self, dataset: Union[List[Any], str, Any, Dict[str, Any]]) -> Tuple[List[dict], List[dict]]:
        """Convert any dataset to GEPA format with train/val split"""
        try:
            # Handle UI tree dataset format
            if isinstance(dataset, dict) and 'type' in dataset and dataset['type'] == 'ui_tree_dataset':
                return self.convert_ui_tree_dataset(
                    dataset.get('json_dir', 'json_tree'),
                    dataset.get('screenshots_dir', 'screenshots')
                )
            elif isinstance(dataset, str):
                data = self._load_from_path(dataset)
            elif hasattr(dataset, 'to_dict'):  # pandas DataFrame
                data = dataset.to_dict(orient='records')
            elif isinstance(dataset, list):
                data = dataset
            else:
                data = [dataset]

            logger.info(f"Normalized data length: {len(data)}")
            standardized = self._standardize(data)
            train, val = self._split(standardized)
            return train, val
        except (FileNotFoundError, ValueError, TypeError) as e:
            raise DatasetError(f"Failed to convert dataset: {str(e)}")

    def _load_from_path(self, path: str) -> List[Any]:
        """Load data from file path"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        ext = p.suffix.lower()
        if ext in self.supported_extensions:
            return [self.loader.load(p)]
        else:
            raise DatasetError(f"Unsupported file extension: {ext}")

    def _standardize(self, data: List[Any]) -> List[dict]:
        """Standardize data to input/output format
        
        Handles both UI tree JSON format and simple text inputs.
        UI tree format should have: {'screenshot': str, 'ui_tree': dict, 'expected_output': str}
        Simple format can be: {'input': str, 'output': str} or {'question': str, 'answer': str} etc.
        """
        out = []
        for item in data:
            if not isinstance(item, dict):
                item = {'input': str(item)}
                
            # Handle UI tree JSON format
            if 'ui_tree' in item and 'screenshot' in item:
                ui_tree = item['ui_tree']
                input_text = ui_tree.get('text', '')
                output_text = item.get('expected_output', '')
                image = item.get('screenshot', '')
                out.append({'input': input_text, 'output': output_text, 'image': image})
            # Handle simple text format
            else:
                inp = self._extract(item, ['input', 'question', 'text', 'prompt']) or ''
                outp = self._extract(item, ['output', 'result', 'response', 'answer', 'expected_output']) or ''
                image = self._extract(item, ['image', 'image_base64', 'screenshot']) or ''
                out.append({'input': inp, 'output': outp, 'image': image})
                
        return out

    def _extract(self, d: dict, keys: List[str]) -> Union[str, None]:
        """Extract value by trying multiple keys"""
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _split(self, data: List[dict], ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
        """Split data into train and validation sets"""
        split = max(1, int(len(data) * ratio))
        train = data[:split]
        val = data[split:] or data[-1:]  # Ensure val is not empty
        return train, val

    def convert_ui_tree_dataset(self, json_dir: str, screenshots_dir: str) -> Tuple[List[dict], List[dict]]:
        """
        Convert UI tree dataset (JSON + screenshots) to GEPA format
        
        Args:
            json_dir: Directory containing JSON files
            screenshots_dir: Directory containing screenshot images
            
        Returns:
            Tuple of (train_data, val_data) in GEPA format
        """
        try:
            # Load paired dataset
            dataset = self.loader.load_ui_tree_dataset(json_dir, screenshots_dir)
            
            if not dataset:
                raise DatasetError("No valid image-JSON pairs found")
            
            logger.info(f"Loaded {len(dataset)} UI tree samples")
            
            # Split into train/val
            train, val = self._split(dataset)
            
            logger.info(f"Split dataset: {len(train)} train, {len(val)} validation")
            return train, val
            
        except Exception as e:
            raise DatasetError(f"Failed to convert UI tree dataset: {str(e)}")
