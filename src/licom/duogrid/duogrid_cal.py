"""
File: duogrid_cal.py
Description: Calculation functions for Duogrid class.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-19
Updated: 2025-09-19
"""

from backend.calculation.field import Field

def duogrid_cal(cls):
    """
    Decorator to add calculation methods to Duogrid class.
    
    This decorator adds a general init method for calculations.
    You can easily add new calculation functions here.
    
    Args:
        cls: The class to be decorated
        
    Returns:
        The decorated class with calculation methods
    """
    
    @classmethod
    def init_calculations(cls):
        """
        Initialize all calculation fields.
        This is the main entry point for all calculations.
        """
        cls._init_inner_outer_fields()
        # other calculations here ...
        # cls._init_other_calculations()
    
    @classmethod
    def _init_inner_outer_fields(cls):
        """Initialize the inner and outer fields."""
        
        ng = cls.mp.ng
        # inner field: 0s on boundary, 1s in interior
        cls.inner = Field.new('2d')
        cls.inner = cls.inner.at[:,:].set(0)
        sizex = cls.inner.shape[0]
        sizey = cls.inner.shape[1]
        cls.inner = cls.inner.at[ng:sizex-ng,ng:sizey-ng].set(1)

        # outer field: 1s on boundary, 0s in interior  
        cls.outer = Field.new('2d')
        cls.outer = cls.outer.at[:,:].set(1)
        sizex = cls.outer.shape[0]
        sizey = cls.outer.shape[1]
        cls.outer = cls.outer.at[ng:sizex-ng,ng:sizey-ng].set(0)
    
    # Add the methods to the class
    cls.init_calculations = init_calculations
    cls._init_inner_outer_fields = _init_inner_outer_fields
    
    return cls