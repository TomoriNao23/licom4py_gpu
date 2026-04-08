"""
File: w92_field.py
Description: Velocity field transformation functions for converting between
    spherical (lat-lon) coordinates and cubed-sphere grid coordinates.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-03-19

REVISION HISTORY:
    22/09/2025 - Initial implementation of velocity transformation
    19/03/2026 - Consolidate typing imports; refactor imports to package-level paths
"""

# Standard library imports
import functools
from typing import Any, Optional, Tuple

# Third-party imports
import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.mesh import Communication, Cube, GPU_Mesh
from licom.mesh.g2l import Global2Local
from licom.duogrid import Dg
from licom.operators import Remap
from jax.sharding import NamedSharding
from jax.experimental.shard_map import shard_map

@functools.partial(jit, 
                   static_argnames=("test_case",))
def initialize_test_velocity_field_jit(ub_in: Any, vb_in: Any, h0_in: Any, test_case: str = 'w92case2') -> Tuple[Any, Any, Any]:
    """JIT version of velocity initialization for sharding safety."""
    if test_case == 'w92case2':
        ubar = 1.0
        alpha = 0.0
        radius = 6.371e6
        omega = 7.292e-5
        grav = 9.80

        def map_fn(ub_loc, vb_loc, h0_loc, a_pt_loc, a_l2c_loc):
            lon = a_pt_loc[..., 3:-3, 3:-3, 0]
            lat = a_pt_loc[..., 3:-3, 3:-3, 1]
            
            u_rll_lon = ubar * (jnp.cos(alpha) * jnp.cos(lat) + 
                                jnp.sin(alpha) * jnp.cos(lon) * jnp.sin(lat))
            u_rll_lat = -ubar * jnp.sin(alpha) * jnp.sin(lon)
            
            a_l2c_int = a_l2c_loc[..., 3:-3, 3:-3, :, :]
            u_cubed = (a_l2c_int[..., 0, 0] * u_rll_lon + 
                       a_l2c_int[..., 0, 1] * u_rll_lat)
            v_cubed = (a_l2c_int[..., 1, 0] * u_rll_lon + 
                       a_l2c_int[..., 1, 1] * u_rll_lat)
            
            h0_int = -(radius*omega*ubar+ubar**2/2.) * (
                (-jnp.cos(lon)*jnp.cos(lat)*jnp.sin(alpha)+jnp.sin(lat)*jnp.cos(alpha))**2 - 1.0/3.0
            )/grav

            ub_out = ub_loc.at[..., 3:-3, 3:-3].set(u_cubed)
            vb_out = vb_loc.at[..., 3:-3, 3:-3].set(v_cubed)
            h0_out = h0_loc.at[..., 3:-3, 3:-3].set(h0_int)
            return ub_out, vb_out, h0_out

        spec_2d = Global2Local.get_spec(ub_in.shape)
        spec_pt = Global2Local.get_spec(Dg.a_pt.shape)
        spec_l2c = Global2Local.get_spec(Dg.a_l2c.shape)

        sharded_fn = shard_map(
            map_fn,
            mesh=GPU_Mesh.mesh,
            in_specs=(spec_2d, spec_2d, spec_2d, spec_pt, spec_l2c),
            out_specs=(spec_2d, spec_2d, spec_2d),
            check_rep=False
        )
        return sharded_fn(ub_in, vb_in, h0_in, Dg.a_pt, Dg.a_l2c)
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

        sharding = NamedSharding(GPU_Mesh.mesh, Global2Local.get_spec(momentum.h0.shape))
        
        momentum.h0 = jax.device_put(Cube.ext_scalar(momentum.h0), sharding)
        
        ub_ext, vb_ext = Cube.ext_vector(momentum.ub, momentum.vb)
        momentum.ub = jax.device_put(ub_ext, sharding)
        momentum.vb = jax.device_put(vb_ext, sharding)

        # ubp, vbp, h0p
        momentum.ubp = momentum.ub
        momentum.vbp = momentum.vb
        momentum.h0p = momentum.h0

        ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy = Remap.vector_trans_2d(ub, vb)
        print(ub[0,2,3],vb[0,2,3],ub_ct[0,2,3],vb_ct[0,2,3],ub_cx[0,2,3],ub_cy[0,2,3],vb_cx[0,2,3],vb_cy[0,2,3])
        # 手动重算 ub_ct[0,2,3]：(a_gct[0,0]*ub + a_gct[0,1]*vb) * a_sina

    else:
        raise ValueError(f"Unknown test case: {test_case}")
