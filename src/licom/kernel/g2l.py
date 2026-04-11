"""
File: g2l.py
Description: Refactored Global-to-Local distribution logic with dynamic
             dimension detection and halo handling for different grid structures.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
"""

# Standard library imports
from typing import Any, Dict, Tuple

# Third-party imports
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


class Global2Local:
    # Global configuration parameters
    mesh: Mesh = None
    halo: int = 3
    nx_local: int = 0
    ny_local: int = 0
    nx: int = 0
    ny: int = 0
    npz: int = 30
    ntile: int = 6
    dtype: jnp.dtype = jnp.float64

    @classmethod
    def configure(
        cls,
        mesh,
        halo: int,
        nx_local: int,
        ny_local: int,
        nx: int,
        ny: int,
        npz: int,
        ntile: int = 6,
    ) -> None:
        """Retrieve base configuration from GPU_Mesh"""
        cls.mesh = mesh
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.nx = nx
        cls.ny = ny
        cls.npz = npz
        cls.ntile = ntile

    @classmethod
    def get_spec(cls, shape: Tuple[int, ...]) -> P:
        """
        Core logic: Dynamically detect dimensions and return PartitionSpec.
        Rules:
          1. Dimension 0 (size == ntile) is constantly mapped to 'tile'
          2. Skipping dimension 0, the first two dimensions with size > nx_local are mapped to 'x' and 'y' respectively
          3. Remaining dimensions are mapped to None (no sharding)
        """
        spec_list = [None] * len(shape)
        if len(shape) > 0 and shape[0] == cls.ntile:
            spec_list[0] = "tile"

        found_spatial = 0
        for i in range(1, len(shape)):
            # nx_local is the local width without halo points;
            # arrays with halo will inherently feature a spatial dimension strictly greater than nx_local.
            if shape[i] >= cls.nx_local and found_spatial < 2:
                spec_list[i] = "y" if found_spatial == 0 else "x"
                found_spatial += 1

        return P(*spec_list)

    @classmethod
    def distribute_pre_padded(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        Distribute a global array that [already incorporates the outermost Halo regions] across JAX devices.
        Every device will exactly receive a partitioned chunk of (nx_local + 2h) × (ny_local + 2h) data.
        """
        h = cls.halo
        spec = cls.get_spec(global_data.shape)
        spatial_axes = [i for i, s in enumerate(spec) if s in ("x", "y")]

        if len(spatial_axes) != 2:
            return jax.device_put(global_data, NamedSharding(cls.mesh, spec))

        axis_1, axis_2 = spatial_axes
        nx_h = cls.nx_local + 2 * h
        ny_h = cls.ny_local + 2 * h
        px = cls.mesh.shape["x"]
        py = cls.mesh.shape["y"]

        target_shape = list(global_data.shape)
        target_shape[axis_1] = py * nx_h
        target_shape[axis_2] = px * ny_h
        target_shape = tuple(target_shape)

        sharding = NamedSharding(cls.mesh, spec)

        def slice_fn(idx):
            slc = list(idx)
            start_1 = 0 if slc[axis_1].start is None else slc[axis_1].start
            start_2 = 0 if slc[axis_2].start is None else slc[axis_2].start

            iy = start_1 // nx_h
            ix = start_2 // ny_h

            slc[axis_1] = slice(iy * cls.nx_local, iy * cls.nx_local + nx_h)
            slc[axis_2] = slice(ix * cls.ny_local, ix * cls.ny_local + ny_h)
            return global_data[tuple(slc)]

        return jax.make_array_from_callback(target_shape, sharding, slice_fn)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        Globally pad a raw array [without Halo regions] on the CPU native buffer,
        then fallback to distribute_pre_padded for sharding.
        """
        h = cls.halo
        spec = cls.get_spec(global_data.shape)
        spatial_axes = [i for i, s in enumerate(spec) if s in ("x", "y")]

        padding = [(0, 0)] * global_data.ndim
        for axis in spatial_axes:
            padding[axis] = (h, h)
        padded = jnp.pad(global_data, padding, mode="constant", constant_values=0)

        return cls.distribute_pre_padded(padded)

    # ---- Convenience Factory Methods ----

    @classmethod
    def zeros(cls, shape_key: str) -> jnp.ndarray:
        """
        Create a sharded zeros array with Halo padding initialized based on shape key.
        Uses the [halo-free] global shape to construct the native zeros array, then dispatches to `distribute`.
        """
        # Global shapes (without halo)
        shapes = {
            "2d": (cls.ntile, cls.nx, cls.ny),
            "3d": (cls.ntile, cls.npz, cls.nx, cls.ny),
            "4d": (cls.ntile, cls.nx, cls.ny, 2, 2),
            "3d1": (cls.ntile, cls.npz + 1, cls.nx, cls.ny),
            "3d_agrid": (cls.ntile, cls.nx, cls.ny, 2),
            "4d_agrid": (cls.ntile, cls.nx, cls.ny, 2, 2),
        }

        return cls.distribute(jnp.zeros(shapes[shape_key], dtype=cls.dtype))

    @classmethod
    def allocate(cls, owner: Any, field_groups: Dict[str, Any]) -> None:
        """Batch allocate generic fields"""
        for shape_key, names in field_groups.items():
            for name in names:
                setattr(owner, name, cls.zeros(shape_key))
