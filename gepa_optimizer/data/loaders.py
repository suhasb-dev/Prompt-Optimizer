"""
Data loading utilities for various file formats
"""

import json
import base64
import pandas as pd
from typing import Any, Optional, Union, List , Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility class for loading data from various sources
    """
    
    def __init__(self):
        self.supported_formats = [
            '.csv', '.json', '.jsonl', '.txt', '.md', '.xlsx',
            '.png', '.jpg', '.jpeg'
        ]
    
    def load(self, source: Union[str, Path], format_hint: Optional[str] = None) -> Optional[Any]:
        """
        Load data from any supported source
        
        Args:
            source: File path or data source
            format_hint: Optional format hint to override auto-detection
            
        Returns:
            Loaded data or None if failed
        """
        try:
            path = Path(source)
            
            if not path.exists():
                logger.error(f"File not found: {source}")
                return None
            
            # Use format hint or detect from extension
            file_format = format_hint or path.suffix.lower()
            
            if file_format == '.csv':
                return self.load_csv(path)
            elif file_format == '.json':
                return self.load_json(path)
            elif file_format == '.jsonl':
                return self.load_jsonl(path)
            elif file_format in ['.txt', '.md']:
                return self.load_text(path)
            elif file_format == '.xlsx':
                return self.load_excel(path)
            elif file_format in ['.png', '.jpg', '.jpeg']:
                return self.load_image_base64(path)
            else:
                logger.warning(f"Unsupported format: {file_format}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data from {source}: {str(e)}")
            return None
    
    def load_csv(self, path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load CSV file as pandas DataFrame"""
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {path}: {str(e)}")
            return None
    
    def load_json(self, path: Union[str, Path]) -> Optional[Any]:
        """Load JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                logger.info(f"Loaded JSON with {len(data)} items")
            else:
                logger.info("Loaded JSON object")
            
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {str(e)}")
            return None
    
    def load_jsonl(self, path: Union[str, Path]) -> Optional[List[Dict]]:
        """Load JSONL (JSON Lines) file"""
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {str(e)}")
            
            logger.info(f"Loaded JSONL with {len(data)} items")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSONL {path}: {str(e)}")
            return None
    
    def load_text(self, path: Union[str, Path]) -> Optional[str]:
        """Load plain text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded text file with {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"Failed to load text {path}: {str(e)}")
            return None
    
    def load_excel(self, path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load Excel file as pandas DataFrame"""
        try:
            df = pd.read_excel(path)
            logger.info(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel {path}: {str(e)}")
            return None
            
    def load_image_base64(self, path: Union[str, Path]) -> Optional[str]:
        """Load image file and encode as Base64 string"""
        try:
            with open(path, 'rb') as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            logger.info(f"Loaded image {path} and encoded to Base64")
            return encoded_string
        except Exception as e:
            logger.error(f"Failed to load image {path}: {str(e)}")
            return None

    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_formats
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file"""
        path = Path(file_path)
        
        if not path.exists():
            return {'exists': False}
        
        return {
            'exists': True,
            'size': path.stat().st_size,
            'format': path.suffix.lower(),
            'supported': self.is_supported_format(path),
            'name': path.name,
            'stem': path.stem,
            'parent': str(path.parent)
        }

    def load_ui_tree_dataset(self, json_dir: str, screenshots_dir: str) -> List[Dict[str, Any]]:
        """
        Load UI tree dataset by pairing JSON files with corresponding screenshots
        
        Args:
            json_dir: Directory containing JSON files (e.g., "json_tree")
            screenshots_dir: Directory containing screenshot images (e.g., "screenshots")
            
        Returns:
            List of dictionaries with 'input', 'output', and 'image' keys
        """
        json_path = Path(json_dir)
        screenshots_path = Path(screenshots_dir)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        if not screenshots_path.exists():
            raise FileNotFoundError(f"Screenshots directory not found: {screenshots_dir}")
        
        dataset = []
        
        # Get all JSON files
        json_files = list(json_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {json_dir}")
        
        for json_file in json_files:
            # Extract filename without extension (e.g., "2" from "2.json")
            file_stem = json_file.stem
            
            # Look for corresponding image file
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_file = None
            
            for ext in image_extensions:
                potential_image = screenshots_path / f"{file_stem}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            if not image_file:
                logger.warning(f"No corresponding image found for {json_file.name}")
                continue
                
            try:
                # Load JSON content
                json_data = self.load_json(json_file)
                if not json_data:
                    logger.warning(f"Failed to load JSON: {json_file}")
                    continue
                    
                # Load image as base64
                image_base64 = self.load_image_base64(image_file)
                if not image_base64:
                    logger.warning(f"Failed to load image: {image_file}")
                    continue
                
                # Create dataset entry
                dataset_entry = {
                    'input': 'Extract UI elements from this screenshot and provide the complete UI tree structure',
                    'output': json.dumps(json_data, indent=2),  # Convert JSON to string
                    'image': image_base64
                }
                
                dataset.append(dataset_entry)
                logger.debug(f"Loaded pair: {json_file.name} + {image_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(dataset)} image-JSON pairs")
        return dataset
