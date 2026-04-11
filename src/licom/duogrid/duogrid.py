"""
File: duogrid.py
Description: Duogrid data structure for grid management.
    Now uses the refactored Global2Local for intelligent auto-sharding
    of pre-padded arrays.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15
Updated: 2026-03-19

REVISION HISTORY:
    15/03/2026 - Initial implementation with Global2Local-based sharding
    16/03/2026 - Refine field distribution logic
    19/03/2026 - Refactor imports to package-level paths
"""

# Standard library imports
import os

# Third-party imports
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.sharding import NamedSharding

# Local application imports
from licom.kernel import Global2Local


class Duogrid:
    """
    Duogrid data structure for grid management in LICOM ocean model.
    """

    # Grid parameters
    ntile: int = 6
    halo: int = 3
    nx_local: int = 0
    ny_local: int = 0
    px: int = 1
    py: int = 1

    @classmethod
    def configure(cls, namelist, gpu_mesh) -> "Duogrid":
        cls.ntile = gpu_mesh.ntile
        cls.halo = gpu_mesh.halo
        cls.nx_local = gpu_mesh.nx_local
        cls.ny_local = gpu_mesh.ny_local
        cls.px = gpu_mesh.px
        cls.py = gpu_mesh.py

        # Retrieve the NPZ path
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        npz_path = os.path.join(project_root, "field", f"duogrid_C{namelist.nx}.npz")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Duogrid NPZ file not found: {npz_path}")

        data = np.load(npz_path)
        print(f"Loading pre-padded duogrid from: {npz_path}")

        # Use the refactored distribution function
        cls._distribute_fields(data)

        # Initialize the generated mask and depth fields
        # These functions internally call Global2Local.zeros/array maintaining sharding semantics
        cls._init_calculations()

        # Local application imports
        from licom.kernel import Cube

        Cube.configure(
            cls.k2e_coef,
            cls.k2e_loc_i,
            cls.k2e_loc_j,
            cls.a_c2l,
            cls.a_l2c,
            cls.inner,
            cls.outer,
            cls.nx_local,
            cls.ny_local,
            cls.px,
            cls.py,
        )

        print("Duogrid initialized successfully.")
        return cls

    @classmethod
    def _distribute_fields(cls, data) -> None:
        """
        Distribute the mesh fields that already include Halo regions.
        Note: Unified calling of distribute_pre_padded, which automatically infers
        the 'x' and 'y' axes based on the array layout (6, [npz], nx+h, ny+h, [2,2]).
        """
        dist_fixed = Global2Local.distribute_pre_padded

        # ---- Automatically shard all grid fields ----
        # Includes all internal a, b, c, and d grid components from the original data list
        fields_to_load = [
            "a_x",
            "a_y",
            "a_kik_x",
            "a_kik_y",
            "a_sina",
            "a_cosa",
            "a_dx",
            "a_dy",
            "a_da",
            "rda",
            "rdx",
            "rdy",
            "a_f",
            "ub",
            "vb",
            "a_pt",
            "k2e_coef",
            "a_gco",
            "a_gct",
            "a_c2l",
            "a_l2c",
            "k2e_loc_i",
            "k2e_loc_j",
            "b_pt",
            "c_gco",
            "c_gct",
            "c_ct2ort_x",
            "c_ort2ct_x",
            "c_sina",
            "c_cosa",
            "c_dy",
            "c_dx",
            "d_gco",
            "d_gct",
            "d_ct2ort_y",
            "d_ort2ct_y",
            "d_sina",
            "d_cosa",
            "d_dx",
            "d_dy",
        ]

        for name in fields_to_load:
            if name in data:
                # Explicitly assign to jnp array and partition matching original memory layout
                setattr(
                    cls,
                    name,
                    dist_fixed(jnp.array(data[name], dtype=Global2Local.dtype)),
                )

        # Special handling: integer-based index fields
        if "k2e_loc" in data:
            cls.k2e_loc = dist_fixed(jnp.array(data["k2e_loc"], dtype=jnp.int32))

    @classmethod
    def _init_calculations(cls) -> None:
        """Initialize auto-generated derivative topological fields"""
        cls._init_inner_outer_fields()
        cls._ocean_depth()

    @classmethod
    def _init_inner_outer_fields(cls) -> None:
        """Populate Mask fields dynamically via logical boolean bounds"""
        # Global2Local.zeros builds the correctly distributed array using a '2d' template shape (6, nx_h, ny_h)
        inner = Global2Local.zeros("2d")

        # Third-party imports
        from jax.experimental.shard_map import shard_map

        # Local application imports
        from licom.kernel import GPU_Mesh

        with GPU_Mesh.mesh:
            ones_full = jnp.ones_like(inner)

            spec = Global2Local.get_spec(inner.shape)

            def _set_mask_logic(in_arr, out_arr):
                h = cls.halo
                in_arr = in_arr.at[..., h:-h, h:-h].set(1.0)
                out_arr = out_arr.at[..., h:-h, h:-h].set(0.0)
                return in_arr, out_arr

            sharded_fn = shard_map(
                _set_mask_logic,
                mesh=GPU_Mesh.mesh,
                in_specs=(spec, spec),
                out_specs=(spec, spec),
                check_rep=False,
            )

            cls.inner, cls.outer = jit(sharded_fn)(inner, ones_full)

    @classmethod
    def _ocean_depth(cls) -> None:
        """Initialize 2D/3D fields related to Ocean topology and depth"""

        # Preallocate correctly sharded GPU memory space natively
        dzph_init = Global2Local.zeros("2d")
        kmt_init = Global2Local.zeros("2d")
        vit_init = Global2Local.zeros("3d")

        # Third-party imports
        from jax.experimental.shard_map import shard_map

        # Local application imports
        from licom.kernel import GPU_Mesh

        with GPU_Mesh.mesh:
            spec_2d = Global2Local.get_spec(dzph_init.shape)
            spec_3d = Global2Local.get_spec(vit_init.shape)

            def _init_depth_logic(dzph, kmt, vit):
                # Initialize all bounds/depths to standard constants (dzph=5600, kmt=30, vit=1.0)
                dzph = dzph.at[:].set(5600.0)
                kmt = kmt.at[:].set(30.0)
                vit = vit.at[:].set(1.0)
                return dzph, kmt, vit

            sharded_fn = shard_map(
                _init_depth_logic,
                mesh=GPU_Mesh.mesh,
                in_specs=(spec_2d, spec_2d, spec_3d),
                out_specs=(spec_2d, spec_2d, spec_3d),
                check_rep=False,
            )

            cls.dzph, cls.kmt, cls.vit = jit(sharded_fn)(dzph_init, kmt_init, vit_init)
        # Fields corresponding to B/C/D grids might be derived via interpolation over stencils, mapped dynamically
        cls.dzph_x = cls.dzph
        cls.dzph_y = cls.dzph
