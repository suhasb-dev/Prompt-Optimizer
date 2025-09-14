"""
Custom exceptions for GEPA Optimizer
"""

class GepaOptimizerError(Exception):
    """Base class for all GEPA Optimizer exceptions"""
    pass

class GepaDependencyError(GepaOptimizerError):
    """Exception raised for errors related to the GEPA library dependency"""
    pass

class InvalidInputError(GepaOptimizerError):
    """Exception raised for invalid user inputs"""
    pass

class DatasetError(GepaOptimizerError):
    """Exception raised for errors related to the dataset"""
    pass
