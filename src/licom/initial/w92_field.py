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
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.duogrid import Dg
from licom.kernel import Communication, GPU_Mesh
from licom.kernel.g2l import Global2Local
from licom.operators import Remap


@functools.partial(jit, static_argnames=("test_case",))
def initialize_test_velocity_field_jit(
    ub_in: Any, vb_in: Any, h0_in: Any, test_case: str = "w92case2"
) -> Tuple[Any, Any, Any]:
    """JIT version of velocity initialization for sharding safety."""
    if test_case == "w92case2":
        ubar = 1.0
        alpha = 0.0
        radius = 6.371e6
        omega = 7.292e-5
        grav = 9.80

        def map_fn(ub_loc, vb_loc, h0_loc, a_pt_loc, a_l2c_loc):
            """Calculate spherical coordinates, map velocities to local cubed-sphere projection, and compute steady-state surface elevation."""
            lon = a_pt_loc[..., 3:-3, 3:-3, 0]
            lat = a_pt_loc[..., 3:-3, 3:-3, 1]

            u_rll_lon = ubar * (
                jnp.cos(alpha) * jnp.cos(lat)
                + jnp.sin(alpha) * jnp.cos(lon) * jnp.sin(lat)
            )
            u_rll_lat = -ubar * jnp.sin(alpha) * jnp.sin(lon)

            a_l2c_int = a_l2c_loc[..., 3:-3, 3:-3, :, :]
            u_cubed = (
                a_l2c_int[..., 0, 0] * u_rll_lon + a_l2c_int[..., 0, 1] * u_rll_lat
            )
            v_cubed = (
                a_l2c_int[..., 1, 0] * u_rll_lon + a_l2c_int[..., 1, 1] * u_rll_lat
            )

            h0_int = (
                -(radius * omega * ubar + ubar**2 / 2.0)
                * (
                    (
                        -jnp.cos(lon) * jnp.cos(lat) * jnp.sin(alpha)
                        + jnp.sin(lat) * jnp.cos(alpha)
                    )
                    ** 2
                    - 1.0 / 3.0
                )
                / grav
            )

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
            check_rep=False,
        )
        return sharded_fn(ub_in, vb_in, h0_in, Dg.a_pt, Dg.a_l2c)
    else:
        return ub_in, vb_in, h0_in


def initialize_test_velocity_field(momentum=None, test_case: str = "w92case2") -> None:
    """
    Initialize test velocity fields using fully localized SPMD mapping for sharding safety.
    Resolves analytical steady-state conditions and synchronizes corner padding globally.
    """
    with GPU_Mesh.mesh:
        momentum.ub, momentum.vb, momentum.h0 = initialize_test_velocity_field_jit(
            momentum.ub, momentum.vb, momentum.h0, test_case
        )

        # Local application imports
        from licom.kernel.cube import ext_scalar, ext_vector
        from licom.kernel.spmd import make_spmd_jit

        px = GPU_Mesh.mesh.shape["x"]
        py = GPU_Mesh.mesh.shape["y"]

        def _scalar_core(state, consts):
            (h0,) = state
            coef, loc_i, loc_j = consts
            return (ext_scalar(h0, coef, loc_i, loc_j, px, py),)

        def _vector_core(state, consts):
            u, v = state
            coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer = consts
            return ext_vector(u, v, coef, loc_i, loc_j, a_c2l, a_l2c, inner, outer, px, py)

        scalar_consts = (Dg.k2e_coef, Dg.loc_i_local, Dg.loc_j_local)
        vector_consts = (Dg.k2e_coef, Dg.loc_i_local, Dg.loc_j_local,
                         Dg.a_c2l, Dg.a_l2c, Dg.inner, Dg.outer)

        ext_scalar_spmd = make_spmd_jit(
            _scalar_core, (momentum.h0,), scalar_consts, static_argnums=()
        )
        ext_vector_spmd = make_spmd_jit(
            _vector_core, (momentum.ub, momentum.vb), vector_consts, static_argnums=()
        )

        (momentum.h0,) = ext_scalar_spmd((momentum.h0,), scalar_consts)
        momentum.ub, momentum.vb = ext_vector_spmd(
            (momentum.ub, momentum.vb), vector_consts
        )

    # ubp, vbp, h0p
    momentum.ubp = momentum.ub
    momentum.vbp = momentum.vb
    momentum.h0p = momentum.h0

    # print("debug h0 4 56 98:", momentum.h0[4, 56, 98])
    # print(momentum.h0[0,2,48],momentum.h0[0,2,49], momentum.h0[0,2,50])
    # print(momentum.h0[0,2,48+3], momentum.h0[0,2,49+3],momentum.h0[0,2,50+3])
    # print(momentum.h0[0,2,48+6], momentum.h0[0,2,49+6],momentum.h0[0,2,50+6])
    # print(momentum.h0[0,2,48+9], momentum.h0[0,2,49+9],momentum.h0[0,2,50+9])
    # print(momentum.h0[0,2,0:6], momentum.h0[0,2,-6:])
    # print(momentum.h0[0,2,48:51], momentum.h0[0,2,51:54])
    # print(momentum.h0[0,2,54:57], momentum.h0[0,2,57:60])
