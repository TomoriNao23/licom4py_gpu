"""
File: duogrid.py
Description: Duogrid data structure for grid management in LICOM ocean model.
    Loads pre-computed grid data from NPZ file and distributes via Global2Local.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-04
Updated: 2026-03-15 (Refactored: remove Field abstraction, use Global2Local)
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import jax.numpy as jnp

# Local application imports
from mesh.g2l import Global2Local


class Duogrid:
    """
    Duogrid data structure for grid management in LICOM ocean model.
    """

    # Grid parameters
    ntile: int = 6
    halo: int = 3
    nx: int = 0
    ny: int = 0
    nx_local: int = 0
    ny_local: int = 0

    @classmethod
    def configure(cls, namelist, gpu_mesh) -> 'Duogrid':
        """
        Initialize Duogrid by loading from NPZ and distributing via Global2Local.
        """
        cls.ntile = gpu_mesh.ntile
        cls.halo = gpu_mesh.halo
        cls.nx = namelist.nx
        cls.ny = namelist.ny
        cls.nx_local = gpu_mesh.nx_local
        cls.ny_local = gpu_mesh.ny_local

        # Resolve NPZ file path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        npz_path = os.path.join(project_root, 'field', f'duogrid_C{namelist.nx}.npz')

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Duogrid NPZ file not found: {npz_path}")

        # Load all arrays from NPZ
        data = np.load(npz_path)
        print(f"Loading duogrid from: {npz_path}")

        # Distribute each field via Global2Local
        cls._distribute_fields(data)

        # Initialize calculation fields (inner/outer masks, ocean depth)
        cls._init_calculations()

        print("Duogrid initialized successfully.")
        return cls

    @classmethod
    def _distribute_fields(cls, data) -> None:
        """Distribute grid fields from NPZ data via Global2Local."""
        distribute = Global2Local.distribute

        # ---- 2D A-grid fields ----
        for name in ['a_x', 'a_y', 'a_kik_x', 'a_kik_y',
                      'a_sina', 'a_cosa', 'a_dx', 'a_dy',
                      'a_da', 'rda', 'rdx', 'rdy',
                      'a_f', 'ub', 'vb']:
            if name in data:
                setattr(cls, name, distribute(jnp.array(data[name])))

        # k2e_loc (integer field)
        if 'k2e_loc' in data:
            cls.k2e_loc = distribute(jnp.array(data['k2e_loc'], dtype=jnp.int32))

        # ---- 3D A-grid fields ----
        for name in ['a_pt', 'k2e_coef']:
            if name in data:
                setattr(cls, name, distribute(jnp.array(data[name])))

        # ---- 4D A-grid fields ----
        for name in ['a_gco', 'a_gct', 'a_c2l', 'a_l2c']:
            if name in data:
                setattr(cls, name, distribute(jnp.array(data[name])))

        # ---- B-grid (Strip halos 3:-3 in both x and y) ----
        if 'b_pt' in data:
            cls.b_pt = distribute(jnp.array(data['b_pt']))

        # ---- C-grid (Strip halos 3:-3 in x only) ----
        for name in ['c_gco', 'c_gct', 'c_ct2ort_x', 'c_ort2ct_x',
                     'c_sina', 'c_cosa', 'c_dy', 'c_dx']:
            if name in data:
                val = jnp.array(data[name])
                setattr(cls, name, distribute(val))

        # ---- D-grid (Strip halos 3:-3 in y only) ----
        for name in ['d_gco', 'd_gct', 'd_ct2ort_y', 'd_ort2ct_y',
                     'd_sina', 'd_cosa', 'd_dx', 'd_dy']:
            if name in data:
                setattr(cls, name, distribute(jnp.array(data[name])))

    @classmethod
    def _init_calculations(cls) -> None:
        """Initialize derived calculation fields."""
        cls._init_inner_outer_fields()
        cls._ocean_depth()

    @classmethod
    def _init_inner_outer_fields(cls) -> None:
        """Initialize inner/outer masks using Global2Local.zeros."""
        h = cls.halo
        
        # inner: zeros everywhere, ones in interior
        cls.inner = Global2Local.zeros('2d')
        cls.inner = cls.inner.at[:, h:-h, h:-h].set(1.0)

        # outer: ones everywhere, zeros in interior
        cls.outer = jnp.ones_like(cls.inner)
        cls.outer = cls.outer.at[:, h:-h, h:-h].set(0.0)

    @classmethod
    def _ocean_depth(cls) -> None:
        """Initialize ocean depth fields using Global2Local.zeros."""
        cls.dzph = Global2Local.zeros('2d').at[:,:,:].set(5600.0)
        cls.dzph_x = Global2Local.zeros('2d').at[:,:,:].set(5600.0)
        cls.dzph_y = Global2Local.zeros('2d').at[:,:,:].set(5600.0)
        cls.kmt = Global2Local.zeros('2d').at[:,:,:].set(30.0)
        cls.vit = Global2Local.zeros('3d').at[:,:,:,:].set(1.0)