"""
File: velocity_transform.py
Description: Velocity field transformation functions for converting between
    spherical (lat-lon) coordinates and cubed-sphere grid coordinates.

Author: Chtholly <mengleshan@mail.iap.ac.cn>  
Created: 2025-09-22
"""

from typing import Optional
# Third-party imports
import jax
import jax.numpy as jnp
import functools
from typing import Tuple

# Local application imports
from backend.calculation.field import Field
from duogrid.duogrid import Duogrid as Dg


@functools.partial(jax.jit, static_argnums=())
def spherical_to_cubed_velocity_field(ubar: float, alpha:Optional[float] = 0.0) -> Tuple[Field.datatype, Field.datatype]:
    """
    Convert uniform spherical velocity field to cubed-sphere grid velocity field.
    
    This function implements the transformation:
    lon = ocn%dg%a_pt(1,i,j)
    lat = ocn%dg%a_pt(2,i,j)  
    u_rll(1) = ubar * (cos(alpha)*cos(lat) + sin(alpha)*cos(lon)*sin(lat))
    u_rll(2) = -ubar * sin(alpha)*sin(lon)
    u_co2 = matmul(ocn%dg%a_l2c(:,:,i,j), u_rll)
    
    Args:
        ubar: Magnitude of uniform velocity field
        alpha: Angle of velocity field direction (in radians)
        
    Returns:
        tuple: (u_cubed, v_cubed) - Velocity components in cubed-sphere coordinates
    """
    
    # Extract longitude and latitude from duogrid
    # a_pt has shape (ni, nj, 2) where [:, :, 0] is lon, [:, :, 1] is lat
    lon = Dg.a_pt[:, :, 0]  # longitude
    lat = Dg.a_pt[:, :, 1]  # latitude
    
    # Calculate spherical velocity components (lat-lon coordinates)
    # u_rll[0]: longitudinal component
    # u_rll[1]: latitudinal component
    u_rll_lon = ubar * (jnp.cos(alpha) * jnp.cos(lat) + 
                        jnp.sin(alpha) * jnp.cos(lon) * jnp.sin(lat))
    u_rll_lat = -ubar * jnp.sin(alpha) * jnp.sin(lon)
    
    # Transform from lat-lon to cubed-sphere coordinates using transformation matrix
    # a_l2c has shape (ni, nj, 2, 2) - transformation matrix from lat-lon to cubed
    u_cubed = (Dg.a_l2c[:, :, 0, 0] * u_rll_lon + 
               Dg.a_l2c[:, :, 0, 1] * u_rll_lat)
    v_cubed = (Dg.a_l2c[:, :, 1, 0] * u_rll_lon + 
               Dg.a_l2c[:, :, 1, 1] * u_rll_lat)
    
    return u_cubed, v_cubed


@functools.partial(jax.jit, static_argnums=())
def cubed_to_spherical_velocity_field(u_cubed: Field.datatype, 
                                     v_cubed: Field.datatype) -> Tuple[Field.datatype, Field.datatype]:
    """
    Convert cubed-sphere velocity field to spherical (lat-lon) velocity field.
    
    This is the inverse transformation of spherical_to_cubed_velocity_field.
    
    Args:
        u_cubed: U-component velocity in cubed-sphere coordinates
        v_cubed: V-component velocity in cubed-sphere coordinates
        
    Returns:
        tuple: (u_lon, u_lat) - Velocity components in spherical coordinates
    """
    
    # Transform from cubed-sphere to lat-lon coordinates using inverse transformation matrix
    # a_c2l has shape (ni, nj, 2, 2) - transformation matrix from cubed to lat-lon
    u_lon = (Dg.a_c2l[:, :, 0, 0] * u_cubed + 
             Dg.a_c2l[:, :, 0, 1] * v_cubed)
    u_lat = (Dg.a_c2l[:, :, 1, 0] * u_cubed + 
             Dg.a_c2l[:, :, 1, 1] * v_cubed)
    
    return u_lon, u_lat


def initialize_test_velocity_field(test_case: str = 'w92case2', momentum = None) -> None:
    """
    Initialize test velocity fields for validation and testing.
    
    Args:
        test_case: Type of test case to initialize
                  
    Returns:
        tuple: (u_field, v_field) - Initialized velocity fields
    """
    
        
    if test_case == 'w92case2' and momentum is not None:
        # Uniform eastward flow
        ubar = 1.0  # m/s
        alpha = 0.0  # Pure zonal (eastward)
        ub, vb = spherical_to_cubed_velocity_field(ubar, alpha)
        momentum.ub = momentum.ub.at[:].set(ub)
        momentum.vb = momentum.vb.at[:].set(vb)

        # ssh field
        radius = 6.371e6
        omega = 7.292e-5
        grav = 9.80

        lon = Dg.a_pt[:, :, 0]  # longitude
        lat = Dg.a_pt[:, :, 1]  # latitude
        momentum.h0 = momentum.h0.at[:].set(
            -(radius*omega*ubar+ubar**2/2.) * 
            (
                (
                    -jnp.cos(lon)*jnp.cos(lat)*jnp.sin(alpha)+jnp.sin(lat)*jnp.cos(alpha)
                )**2 - 1.0/3.0
            )/grav
        )

        # ubp, vbp, h0p
        momentum.ubp = momentum.ubp.at[:].set(momentum.ub)
        momentum.vbp = momentum.vbp.at[:].set(momentum.vb)
        momentum.h0p = momentum.h0p.at[:].set(momentum.h0)

    else:
        raise ValueError(f"Unknown test case: {test_case}")


@functools.partial(jax.jit, static_argnums=())
def spherical_to_cubed_velocity_field_from_components(u_spherical: Field.datatype, 
                                                     v_spherical: Field.datatype) -> Tuple[Field.datatype, Field.datatype]:
    """
    Convert spherical velocity components to cubed-sphere coordinates.
    
    This is a helper function when you already have the spherical components
    rather than computing them from ubar and alpha.
    
    Args:
        u_spherical: Longitudinal velocity component
        v_spherical: Latitudinal velocity component
        
    Returns:
        tuple: (u_cubed, v_cubed) - Velocity components in cubed coordinates
    """
    
    # Transform using the lat-lon to cubed transformation matrix
    u_cubed = (Dg.a_l2c[:, :, 0, 0] * u_spherical + 
               Dg.a_l2c[:, :, 0, 1] * v_spherical)
    v_cubed = (Dg.a_l2c[:, :, 1, 0] * u_spherical + 
               Dg.a_l2c[:, :, 1, 1] * v_spherical)
    
    return u_cubed, v_cubed

