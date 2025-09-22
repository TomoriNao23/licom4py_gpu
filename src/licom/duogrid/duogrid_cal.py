"""
File: duogrid_cal.py
Description: Calculation functions for Duogrid class.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-19
Updated: 2025-09-19
"""

from backend.calculation.field import Field
from typing import Optional

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
        cls._ocean_depth()
        cls._coriolis_parameter()
        # other calculations here ...
        # cls._init_other_calculations()
    
    @classmethod
    def _init_inner_outer_fields(cls):
        """Initialize the inner and outer fields.
        inner field: 0s on halo, 1s in interior
        outer field: 1s on halo, 0s in interior
        """
        
        ng = cls.mp.ng
        # inner field: 0s on halo, 1s in interior
        cls.inner = Field.new('2d')
        cls.inner = cls.inner.at[:,:].set(0)
        sizex = cls.inner.shape[0]
        sizey = cls.inner.shape[1]
        cls.inner = cls.inner.at[ng:sizex-ng,ng:sizey-ng].set(1)

        # outer field: 1s on halo, 0s in interior  
        cls.outer = Field.new('2d')
        cls.outer = cls.outer.at[:,:].set(1)
        sizex = cls.outer.shape[0]
        sizey = cls.outer.shape[1]
        cls.outer = cls.outer.at[ng:sizex-ng,ng:sizey-ng].set(0)

    @classmethod
    def _ocean_depth(cls):

        cls.dzph = Field.new('2d')
        cls.dzph_x = Field.new('2d')
        cls.dzph_y = Field.new('2d')
        cls.kmt = Field.new('2d')

        cls.vit = Field.new('3d')

        cls.dzph = cls.dzph.at[:,:].set(5600.0)
        cls.dzph_x = cls.dzph_x.at[:,:].set(5600.0)
        cls.dzph_y = cls.dzph_y.at[:,:].set(5600.0)
        cls.kmt = cls.kmt.at[:,:].set(30)

        cls.vit = cls.vit.at[:,:,:].set(1)

    @classmethod
    def _coriolis_parameter(cls, alpha: Optional[float] = None):
        """
        Calculate Coriolis parameter using matrix operations.
        Converted from Fortran code:
        do j = jsd,jed
            do i = isd,ied
                lon = dg%a_pt(1,i,j)
                lat = dg%a_pt(2,i,j)
                dg%a_f(i,j) = 2.*OMEGA*(-1.*cos(lon)*cos(lat)*sin(alpha) + sin(lat)*cos(alpha))
            enddo
        enddo
        """
        import jax.numpy as jnp
        
        # Constants
        OMEGA = 7.292e-5  # Earth's angular velocity
        
        # Get longitude and latitude from a_pt field
        # a_pt has shape (2, ied-isd+1, jed-jsd+1) where first dimension is [lon, lat]
        lon = cls.a_pt[:, :, 0]  # longitude values
        lat = cls.a_pt[:, :, 1]  # latitude values

        #print(jnp.sum(cls.a_pt[:, :, 0]), jnp.sum(cls.a_pt[:, :, 1])) if (cls.mp.pe == 6) else None
        
        # Calculate alpha (assuming it's defined elsewhere, if not, set to 0)
        # You may need to define alpha based on your specific requirements
        alpha_ = alpha if alpha is not None else 0.0 
        
        # Matrix calculation of Coriolis parameter
        # f = 2*OMEGA*(-cos(lon)*cos(lat)*sin(alpha) + sin(lat)*cos(alpha))
        cls.a_f= Field.new('2d').at[:,:].set(
            2.0 * OMEGA * (
                -jnp.cos(lon) * jnp.cos(lat) * jnp.sin(alpha_) + 
                jnp.sin(lat) * jnp.cos(alpha_)
            )
        )
    
    # Add the methods to the class
    cls.init_calculations = init_calculations
    cls._init_inner_outer_fields = _init_inner_outer_fields
    cls._ocean_depth = _ocean_depth
    cls._coriolis_parameter = _coriolis_parameter
    
    return cls