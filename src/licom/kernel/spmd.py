"""
File: spmd.py
Description: Generalized Single Program Multiple Data (SPMD) compilation wrapper for LICOM.
             Provides the core translation layer between global tracer tensors and local block execution.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-04-07
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
    Generalized JIT parallel compilation wrapper for distributed processing.
    Performs static Monkey Patching during Tracing, and executes zero-overhead Local Shard boundary calls during real computation.

    Args:
        core_fn: Pure function to be parallelized, signature must be (state, consts, *static_args)
        state: Tuple of state inputs composed of JAX Distributed Arrays (with Sharding layouts)
        consts: Tuple of static, read-only parameter inputs
        static_argnums: Specifies which arguments in wrapper are static constants (starts from 2 by default)
    """
    # Local application imports
    from licom.duogrid import Dg
    from licom.kernel import Cube

    # Extract shard specs for state and consts to explicitly prevent UnspecifiedValue JIT crashes
    state_specs = jax.tree_util.tree_map(_spec, state)
    consts_specs = jax.tree_util.tree_map(_spec, consts)

    # Extract intrinsic sharding definitions to be bound to the final JIT decorator
    state_sh = jax.tree_util.tree_map(_sh, state)
    consts_sh = jax.tree_util.tree_map(_sh, consts)

    # Statically extract the patch mapping table from singleton topological objects (Dg, Cube)
    patch_keys = []
    patch_specs = []
    for obj in (Dg, Cube):
        for k, v in vars(obj).items():
            if hasattr(v, "shape") and hasattr(v, "sharding"):
                patch_keys.append((obj, k))
                patch_specs.append(_spec(v))
    patch_specs = tuple(patch_specs)

    # Define the external model wrapper, utilizing variadic arguments to support distinct physical step constants
    def wrapper(s, c, *static_args):
        # 4.1 Extract local tracer references of the targeted global arrays
        patch_vals = tuple(getattr(obj, k) for obj, k in patch_keys)

        # 4.2 Define the shard_map parallel computational core kernel
        def map_fn(s_inner, c_inner, patch_inner):
            # Cache original global singleton attributes
            old_vals = [getattr(obj, k) for obj, k in patch_keys]

            # Dynamically attach incoming local sliced tracers to module-level singleton objects
            for (obj, k), mapped_v in zip(patch_keys, patch_inner):
                setattr(obj, k, mapped_v)

            try:
                # 4.3 CORE: Execute the JIT mathematical computation
                return core_fn(s_inner, c_inner, *static_args)
            finally:
                # 4.4 Detach local references from singletons, fully restoring original global attributes
                for (obj, k), orig_v in zip(patch_keys, old_vals):
                    setattr(obj, k, orig_v)

        # 4.5 Launch shard_map submitting to the target execution Mesh
        sharded_fn = shard_map(
            map_fn,
            mesh=GPU_Mesh.mesh,
            in_specs=(state_specs, consts_specs, patch_specs),
            out_specs=state_specs,
            check_rep=False,
        )
        return sharded_fn(s, c, patch_vals)

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
    Generic packing function: dynamically bundles required fields from a class instance into state and consts tuples for SPMD JIT consumption.
    """
    state = tuple(getattr(instance, k) for k in state_keys)
    consts = tuple(
        getattr(src, k) if src is not None else getattr(instance, k)
        for src, k in const_sources
    )
    return state, consts


def auto_unpack(instance, state_keys, state):
    """
    Generic unpacking function: systematically writes back the JIT-returned state tuple into the caller instance variables.
    """
    for k, v in zip(state_keys, state):
        setattr(instance, k, v)
