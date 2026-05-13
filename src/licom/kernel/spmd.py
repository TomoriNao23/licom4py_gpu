"""
File: spmd.py
Description: SPMD compilation wrapper for LICOM.
    Provides the translation layer between global distributed arrays
    and local shard execution via shard_map + JIT.

    All data is passed explicitly through (state, consts) — no monkey-patching.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-04-07
Updated: 2026-05-14

REVISION HISTORY:
    07/04/2026 - Initial implementation with monkey-patching
    13/04/2026 - Eliminated monkey-patching; all data through explicit parameters
    14/05/2026 - Updated comments
"""

# Third-party imports
import jax
from jax import jit
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.kernel import GPU_Mesh


def _sh(x):
    return getattr(x, "sharding", None)


def _spec(x):
    sh = _sh(x)
    return sh.spec if sh is not None else P()


def make_spmd_jit(core_fn, state, consts, static_argnums=(2, 3)):
    """
    JIT parallel compilation wrapper for distributed processing.
    All grid data is passed explicitly through state/consts — no singleton patching.

    Args:
        core_fn: Pure function, signature (state, consts, *static_args)
        state: Tuple of mutable state arrays (with Sharding layouts)
        consts: Tuple of read-only parameter arrays
        static_argnums: Which wrapper args are static constants (default: args 2, 3)
    """
    # Extract shard specs for shard_map in/out
    state_specs = jax.tree_util.tree_map(_spec, state)
    consts_specs = jax.tree_util.tree_map(_spec, consts)

    # Extract shardings for JIT in/out
    state_sh = jax.tree_util.tree_map(_sh, state)
    consts_sh = jax.tree_util.tree_map(_sh, consts)

    def wrapper(s, c, *static_args):
        def map_fn(s_inner, c_inner):
            return core_fn(s_inner, c_inner, *static_args)

        sharded_fn = shard_map(
            map_fn,
            mesh=GPU_Mesh.mesh,
            in_specs=(state_specs, consts_specs),
            out_specs=state_specs,
            check_rep=False,
        )
        return sharded_fn(s, c)

    return jit(
        wrapper,
        static_argnums=static_argnums,
        in_shardings=(state_sh, consts_sh),
        out_shardings=state_sh,
    )


# =====================================================================
# Generic Packing and Unpacking Logic
# =====================================================================


def auto_pack(instance, state_keys, const_sources):
    """
    Generic packing function: dynamically bundles required fields from a class 
    instance into state and consts tuples for SPMD JIT consumption.
    """
    state = tuple(getattr(instance, k) for k in state_keys)
    consts = tuple(
        getattr(src, k) if src is not None else getattr(instance, k)
        for src, k in const_sources
    )
    return state, consts


def auto_unpack(instance, state_keys, state):
    """
    Generic unpacking function: systematically writes back the JIT-returned
    state tuple into the caller instance variables.
    """
    for k, v in zip(state_keys, state):
        setattr(instance, k, v)
