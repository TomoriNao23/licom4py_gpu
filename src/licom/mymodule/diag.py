"""
File: diag.py
Description: Diagnostic methods for LICOM model using native JAX sharding.
             Correctly aggregates per-shard statistics to avoid double-counting
             halo regions at internal partition boundaries.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15 (Refactored from MPI to JAX)
Updated: 2026-04-13
"""

# Third-party imports
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.kernel import GPU_Mesh


def _build_diag_fn():
    """
    Build a shard_map-based diagnostic function that correctly computes
    per-shard statistics on the halo-free interior, then aggregates via
    collective reductions (max-of-maxes, sum-of-sums for proper global mean).
    """
    h = GPU_Mesh.halo
    mesh = GPU_Mesh.mesh
    # Total number of interior cells per shard (excluding halo on all 4 sides)
    nx_local = GPU_Mesh.nx_local
    ny_local = GPU_Mesh.ny_local

    in_spec = P("tile", "y", "x")

    def _shard_stats(h0, ub, vb):
        """
        Per-shard kernel: compute max, min, sum on the interior region only.
        The interior is [h:-h, h:-h], which strips the halo from all edges
        of each local shard — no double-counting across partition boundaries.
        """
        h0_int = h0[:, h:-h, h:-h]
        ub_int = ub[:, h:-h, h:-h]
        vb_int = vb[:, h:-h, h:-h]

        # Local per-shard statistics
        h0_max = jnp.max(h0_int)
        h0_min = jnp.min(h0_int)
        h0_sum = jnp.sum(h0_int)

        ub_max = jnp.max(ub_int)
        ub_min = jnp.min(ub_int)
        ub_sum = jnp.sum(ub_int)

        vb_max = jnp.max(vb_int)
        vb_min = jnp.min(vb_int)
        vb_sum = jnp.sum(vb_int)

        # All-reduce across all mesh axes to get global statistics
        g_h0_max = lax.pmax(h0_max, ("tile", "x", "y"))
        g_h0_min = lax.pmin(h0_min, ("tile", "x", "y"))
        g_h0_sum = lax.psum(h0_sum, ("tile", "x", "y"))

        g_ub_max = lax.pmax(ub_max, ("tile", "x", "y"))
        g_ub_min = lax.pmin(ub_min, ("tile", "x", "y"))
        g_ub_sum = lax.psum(ub_sum, ("tile", "x", "y"))

        g_vb_max = lax.pmax(vb_max, ("tile", "x", "y"))
        g_vb_min = lax.pmin(vb_min, ("tile", "x", "y"))
        g_vb_sum = lax.psum(vb_sum, ("tile", "x", "y"))

        return (
            g_h0_max, g_h0_min, g_h0_sum,
            g_ub_max, g_ub_min, g_ub_sum,
            g_vb_max, g_vb_min, g_vb_sum,
        )

    out_spec = P()  # scalar outputs, replicated across all shards

    sharded_fn = shard_map(
        _shard_stats,
        mesh=mesh,
        in_specs=(in_spec, in_spec, in_spec),
        out_specs=out_spec,
        check_rep=False,
    )

    return jax.jit(sharded_fn), nx_local, ny_local


def add_diag_methods(cls):
    """
    Add diagnostic methods to the class.

    Refactored to use shard_map with pmax/pmin/psum for correct global
    aggregation that excludes internal halo overlap regions.
    """

    # Deferred build flag (constructed on first call after GPU_Mesh is configured)
    _diag_state = {"fn": None, "total_cells": 0}

    def _ensure_built():
        if _diag_state["fn"] is None:
            fn, nx_local, ny_local = _build_diag_fn()
            _diag_state["fn"] = fn
            # Total interior cells across all shards:
            # pdev*px*py shards, each with (ntile/pdev * nx_local * ny_local) interior cells
            # Simplifies to: ntile * px * py * nx_local * ny_local
            ntile = GPU_Mesh.ntile
            px = GPU_Mesh.px
            py = GPU_Mesh.py
            _diag_state["total_cells"] = ntile * px * py * nx_local * ny_local

    def get_global_stats(self):
        """
        Get global statistics using shard_map-based reductions.
        Returns (max, min, mean) for h0, ub, vb — 9 values total.
        """
        _ensure_built()
        stats = _diag_state["fn"](self.h0, self.ub, self.vb)
        total = _diag_state["total_cells"]

        # stats: (h0_max, h0_min, h0_sum, ub_max, ub_min, ub_sum, vb_max, vb_min, vb_sum)
        vals = tuple(float(s) for s in stats)
        return (
            vals[0], vals[1], vals[2] / total,   # h0 max, min, mean
            vals[3], vals[4], vals[5] / total,   # ub max, min, mean
            vals[6], vals[7], vals[8] / total,   # vb max, min, mean
        )

    def print_global_diag(self):
        """
        Print global diagnostic information.
        In JAX-native mode, we just print once (usually from host process).
        """
        stats = self.get_global_stats()
        h0max, h0min, h0mean, ubmax, ubmin, ubmean, vbmax, vbmin, vbmean = stats

        # We can just print directly; JAX will handle the host-to-device synchronization.
        print("                  max                  min                 mean")
        print(f"Global h0: {h0max:18.10e}  {h0min:18.10e}  {h0mean:18.10e}")
        print(f"Global ub: {ubmax:18.10e}  {ubmin:18.10e}  {ubmean:18.10e}")
        print(f"Global vb: {vbmax:18.10e}  {vbmin:18.10e}  {vbmean:18.10e}")
        print("------------------------------------------------")

    cls.get_global_stats = get_global_stats
    cls.print_global_diag = print_global_diag
    return cls
