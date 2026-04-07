"""
File: diag.py
Description: Diagnostic methods for LICOM model using native JAX sharding.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15 (Refactored from MPI to JAX)
"""

import jax
import jax.numpy as jnp

@jax.jit
def _get_global_stats_jit(h0, ub, vb):
    """
    Fused JIT kernel for computing global statistics.
    Reduces the number of XLA dispatches from 9 down to 1.
    """
    # Leading dimension is ntile, spatial dimensions are [nx+2h, ny+2h]
    h0_interior = h0[:, 3:-3, 3:-3]
    ub_interior = ub[:, 3:-3, 3:-3]
    vb_interior = vb[:, 3:-3, 3:-3]

    return (
        jnp.max(h0_interior), jnp.min(h0_interior), jnp.mean(h0_interior),
        jnp.max(ub_interior), jnp.min(ub_interior), jnp.mean(ub_interior),
        jnp.max(vb_interior), jnp.min(vb_interior), jnp.mean(vb_interior)
    )

def add_diag_methods(cls):
    """
    Add diagnostic methods to the class.
    
    Refactored to native JAX: JAX arrays with NamedSharding will
    automatically perform global reductions across devices.
    """

    def get_global_stats(self):
        """
        Get global statistics using JAX's automatic reduction on sharded arrays.
        Uses a fused JIT kernel for maximum performance.
        """
        stats = _get_global_stats_jit(self.h0, self.ub, self.vb)
        # 转换回 Python 标量执行同步（此步阻塞，等待计算图结果并返回宿主环境）
        return tuple(float(s) for s in stats)
            
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