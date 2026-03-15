"""
File: g2l.py
Description: Distributes global arrays to local devices with zero-padded halo boundaries.
    Also serves as the central factory for sharded JAX arrays.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-03-15 (Merged Field abstraction functionality)

REVISION HISTORY:
    14/03/2026 - Initial implementation
    15/03/2026 - Added zeros(), array(), and allocate() to replace Field class
"""
# Third-party imports
from typing import Any, Dict
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

# =====================================
# Global2Local Class
# =====================================
class Global2Local:
    """
    Distributes global arrays to individual devices and handles sharded array creation.
    
    This class now replaces the 'Field' abstraction. All created arrays are 
    automatically sharded according to the global 'sharding' attribute.
    """

    # Class attributes (Global configuration)
    sharding: NamedSharding | None = None
    halo: int = 0
    nx_local: int = 0
    ny_local: int = 0
    npz: int = 30
    dtype: jnp.dtype = jnp.float64
    
    # Store pre-calculated shapes for convenience (ntile, nx_h, ny_h, ...)
    _shape: dict = {}
    
    # JIT-compiled vmap function for adding halos
    _add_halo_vmap = None

    @classmethod
    def configure(
        cls,
        sharding: NamedSharding,
        halo: int,
        nx_local: int,
        ny_local: int,
        npz: int = 30,
        ntile: int = 6,
        dtype: jnp.dtype = jnp.float64
    ) -> None:
        """Configures the global parameters and predefines shapes."""
        cls.sharding = sharding
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.npz = npz
        cls.dtype = dtype
        
        # Number of logical tiles (cubed-sphere faces), kept independent of pdev
        # so that leading dimension of all fields is always ntile (e.g. 6),
        # regardless of how many devices we actually use.
        h = halo
        nx_h = nx_local + 2 * h
        ny_h = ny_local + 2 * h
        
        cls._shape = {
            '2d':       (ntile, nx_h, ny_h),
            '3d':       (ntile, npz, nx_h, ny_h),
            '3d1':      (ntile, npz + 1, nx_h, ny_h),
            '3d_agrid': (ntile, nx_h, ny_h, 2),
            '3d_bgrid': (ntile, nx_h + 1, ny_h + 1, 2),
            '3d_cgrid': (ntile, nx_h + 1, ny_h, 2),
            '3d_dgrid': (ntile, nx_h, ny_h + 1, 2),
            '4d_agrid': (ntile, nx_h, ny_h, 2, 2),
            '4d_bgrid': (ntile, nx_h + 1, ny_h + 1, 2, 2),
            '4d_cgrid': (ntile, nx_h + 1, ny_h, 2, 2),
            '4d_dgrid': (ntile, nx_h, ny_h + 1, 2, 2),
        }
        
        cls._add_halo_vmap = jax.jit(jax.vmap(cls._add_halo_single))

    @classmethod
    def _add_halo_single(cls, data_in: jnp.ndarray) -> jnp.ndarray:
        """
        Pads a single tile's data with a zero-filled halo region.
        Only pads the first two dimensions (nx, ny). Trailing dimensions are preserved.
        """
        h = cls.halo
        # Dynamically determine the input local spatial shape
        # This handles cases where B/C/D grids might have nx_local+1 or ny_local+1
        nx_in, ny_in = data_in.shape[0], data_in.shape[1]
        
        # Target spatial shape preserved trailing dimensions
        new_shape = (nx_in + 2 * h, ny_in + 2 * h) + data_in.shape[2:]
        
        u = jnp.zeros(new_shape, dtype=data_in.dtype)
        # Inject the original data into the center spatial region
        return u.at[h:-h, h:-h, ...].set(data_in)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """Shards global data onto devices and adds halo padding."""
        # Safer way to put data on sharding that avoids internal AssertionError:
        # 1. Put on first device as a regular JAX array
        data_jax = jax.device_put(global_data, jax.devices()[0])
        # 2. Reshard it using jax.device_put with sharding (which is generally safer on existing JAX arrays)
        data_dist = jax.device_put(data_jax, cls.sharding)
        return cls._add_halo_vmap(data_dist)

    @classmethod
    def zeros(cls, shape_key: str) -> jnp.ndarray:
        """Create a sharded zero array with specified shape key."""
        if shape_key not in cls._shape:
            raise KeyError(f"Unknown shape key '{shape_key}'")
        # Use device_put for compatibility across JAX versions
        return jax.device_put(jnp.zeros(cls._shape[shape_key], dtype=cls.dtype), cls.sharding)

    @classmethod
    def array(cls, arr: Any) -> jnp.ndarray:
        """Convert input to a sharded jnp.array."""
        # Note: If it's already sharded (e.g. from distribute), this is a no-op
        return jax.device_put(jnp.array(arr, dtype=cls.dtype), cls.sharding)

    @classmethod
    def allocate(cls, owner: Any, field_groups: Dict[str, Any]):
        """Batch allocate fields to the specified object."""
        for shape_key, names in field_groups.items():
            for name in names:
                setattr(owner, name, cls.zeros(shape_key))