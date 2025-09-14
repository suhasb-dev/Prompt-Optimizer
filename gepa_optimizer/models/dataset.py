"""
Dataset models for GEPA Optimizer
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

@dataclass
class DatasetItem:
    """Single item in a dataset"""
    
    # Identifiers
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core data
    input_data: Any = ""
    expected_output: Optional[str] = None
    image_base64: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # File references
    file_paths: List[str] = field(default_factory=list)
    
    # Quality indicators
    quality_score: float = 1.0
    is_validated: bool = False
    validation_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate item after initialization"""
        if self.quality_score < 0 or self.quality_score > 1:
            raise ValueError("quality_score must be between 0 and 1")
    
    def add_tag(self, tag: str):
        """Add a tag to this item"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def mark_validated(self, notes: Optional[List[str]] = None):
        """Mark item as validated"""
        self.is_validated = True
        if notes:
            self.validation_notes.extend(notes)

@dataclass 
class ProcessedDataset:
    """Dataset after processing for GEPA optimization"""
    
    # Identifiers
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Dataset"
    
    # Data
    items: List[DatasetItem] = field(default_factory=list)
    train_split: List[DatasetItem] = field(default_factory=list)
    val_split: List[DatasetItem] = field(default_factory=list)
    
    # Metadata
    source_info: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    total_items: int = 0
    validated_items: int = 0
    avg_quality_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.total_items = len(self.items)
        
        if self.items:
            self.validated_items = sum(1 for item in self.items if item.is_validated)
            self.avg_quality_score = sum(item.quality_score for item in self.items) / len(self.items)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'total_items': self.total_items,
            'validated_items': self.validated_items,
            'validation_rate': self.validated_items / self.total_items if self.total_items > 0 else 0,
            'avg_quality_score': self.avg_quality_score,
            'train_size': len(self.train_split),
            'val_size': len(self.val_split),
            'has_expected_outputs': sum(1 for item in self.items if item.expected_output),
        }
