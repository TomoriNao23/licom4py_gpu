"""
File: velocity_transform.py
Description: Velocity field transformation functions for converting between
    spherical (lat-lon) coordinates and cubed-sphere grid coordinates.

Author: Chtholly <mengleshan@mail.iap.ac.cn>  
Created: 2025-09-22
"""

# Standard library imports
from typing import Any
from typing import Optional

# Third-party imports
import jax
import jax.numpy as jnp
import functools
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from typing import Tuple

# Local application imports

from mesh.communication import Communication
from mesh.cube import Cube
from mesh.gpu_mesh import GPU_Mesh
from duogrid import Dg
from operators.remap import vector_trans_2d

def spherical_to_cubed_velocity_field(ubar: float, alpha:Optional[float] = 0.0) -> Tuple[Any, Any]:
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
    # a_pt has shape (ntile, ni, nj, 2) where [..., 0] is lon, [..., 1] is lat
    lon = Dg.a_pt[..., 0]  # longitude
    lat = Dg.a_pt[..., 1]  # latitude
    
    # Calculate spherical velocity components (lat-lon coordinates)
    u_rll_lon = ubar * (jnp.cos(alpha) * jnp.cos(lat) + 
                        jnp.sin(alpha) * jnp.cos(lon) * jnp.sin(lat))
    u_rll_lat = -ubar * jnp.sin(alpha) * jnp.sin(lon)
    
    # Transform from lat-lon to cubed-sphere coordinates using transformation matrix
    # a_l2c has shape (ntile, ni, nj, 2, 2)
    u_cubed = (Dg.a_l2c[..., 0, 0] * u_rll_lon + 
               Dg.a_l2c[..., 0, 1] * u_rll_lat)
    v_cubed = (Dg.a_l2c[..., 1, 0] * u_rll_lon + 
               Dg.a_l2c[..., 1, 1] * u_rll_lat)
    
    return u_cubed, v_cubed

@functools.partial(pjit, 
                   static_argnames=("test_case",),
                   in_shardings=(P('tile', 'x', 'y'), P('tile', 'x', 'y'), P('tile', 'x', 'y')),
                   out_shardings=(P('tile', 'x', 'y'), P('tile', 'x', 'y'), P('tile', 'x', 'y')))
def initialize_test_velocity_field_jit(ub_in: Any, vb_in: Any, h0_in: Any, test_case: str = 'w92case2') -> Tuple[Any, Any, Any]:
    """JIT version of velocity initialization for sharding safety."""
    if test_case == 'w92case2':
        ubar = 1.0
        alpha = 0.0
        ub, vb = spherical_to_cubed_velocity_field(ubar, alpha)
        
        radius = 6.371e6
        omega = 7.292e-5
        grav = 9.80

        lon = Dg.a_pt[..., 0]
        lat = Dg.a_pt[..., 1]
        h0 = -(radius*omega*ubar+ubar**2/2.) * (
            (-jnp.cos(lon)*jnp.cos(lat)*jnp.sin(alpha)+jnp.sin(lat)*jnp.cos(alpha))**2 - 1.0/3.0
        )/grav
        
        return ub, vb, h0
    else:
        return ub_in, vb_in, h0_in

def initialize_test_velocity_field(momentum = None, test_case: str = 'w92case2') -> None:
    """Initialize test velocity fields using JIT for sharding safety."""
    if momentum is not None:
        with GPU_Mesh.mesh:
            ub, vb, h0 = initialize_test_velocity_field_jit(momentum.ub, momentum.vb, momentum.h0, test_case)
        # 直接用返回结果替换字段，避免形状 / sharding 广播问题
        momentum.ub = ub
        momentum.vb = vb
        momentum.h0 = h0

        momentum.h0 = Cube.ext_scalar(momentum.h0)
        momentum.ub, momentum.vb = Cube.ext_vector(momentum.ub, momentum.vb)

        # ubp, vbp, h0p
        momentum.ubp = momentum.ub
        momentum.vbp = momentum.vb
        momentum.h0p = momentum.h0

    else:
        raise ValueError(f"Unknown test case: {test_case}")
