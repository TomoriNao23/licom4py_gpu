"""
File: field.py
Description: Field class for creating arrays with different backends (JAX, NumPy)
    in the LICOM ocean model backend calculation system.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-04 (Chtholly: add classmethod allocate and deallocate)
Updated: 2025-09-16 (Chtholly: optimization and backend separation)
       : 2025-09-21 (Chtholly: add comment for classmethod allocate and deallocate)
       : 2026-01-31 (Chtholly: gather data for device_put_sharded)
    
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any, Dict

# Local application imports
from datatype import MpDate
from . import jax as jax_backend
from . import numpy as numpy_backend

@dataclass(slots=True)
class Field:
    """
    Field class for creating arrays with different backends (JAX, NumPy).
    Class need to be initialized with .init method.

    Public Attributes:
        datatype: The datatype of the field.

    Public Methods:
        init: Initialize the Field class with the given MP configuration.
        new: Create a new zero array with specified shape.
        array: Convert input to array with specified backend (no error checking).
        allocate: Batch allocate fields to the specified object.
        deallocate: Batch deallocate fields from the specified object.

    Private Attributes:
        _create_inf: The creation information of the field.
        _shape: The shape of the field.
    """
    
    datatype: Any # Union[jax.numpy.ndarray, numpy.ndarray]
    _create_inf: dict
    _shape: dict

    @classmethod
    def configure(cls, mp: MpDate) -> None:
        """
        Initialize the Field class with the given MP configuration.
        """
        # Initialize shape configurations
        cls._shape = {
            # A-grid
            '2d': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1),                       # (isd:ied, jsd:jed)
            '3d': ( mp.npz, mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1,),             # (npz, isd:ied, jsd:jed)
            '3d1': (mp.npz + 1,mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1),           # (npz+1, isd:ied, jsd:jed)

            # B/C/D-grid
            '2d_bgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1 + 1),         # (isd:ied+1, jsd:jed+1)
            '2d_cgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1),             # (isd:ied+1, jsd:jed)
            '2d_dgrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1 + 1),             # (isd:ied, jsd:jed+1)
            
            '3d_agrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1, 2),              # (isd:ied, jsd:jed, 2)
            '3d_bgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1 + 1, 2),      # (isd:ied+1, jsd:jed+1, 2)
            '3d_cgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1, 2),          # (isd:ied+1, jsd:jed, 2)
            '3d_dgrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1 + 1, 2),          # (isd:ied, jsd:jed+1, 2) 

            '4d_agrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1, 2, 2),           # (isd:ied, jsd:jed, 2, 2)
            '4d_bgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1 + 1, 2, 2),   # (isd:ied+1, jsd:jed+1, 2, 2)
            '4d_cgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1, 2, 2),       # (isd:ied+1, jsd:jed, 2, 2)
            '4d_dgrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1 + 1, 2, 2),       # (isd:ied, jsd:jed+1, 2, 2)
    
        }

        # Initialize creation configuration
        cls._create_inf = {}

        (lambda cfg: jax_backend.setup_jax_backend(cls, cfg) if cfg.lib == 'jax' else \
            numpy_backend.setup_numpy_backend(cls, cfg))(mp)
        return None


    @classmethod
    def new(cls, shape: int):
        """Create a new zero array with specified shape."""
        return cls._new(cls, shape)
    
    @classmethod
    def array(cls, arr):
        """Convert input to array with specified backend (no error checking)."""
        return cls._array(cls, arr)

    # High-level Encapsulation
    # Allocate fields
    @classmethod
    def allocate(cls, owner: Any, field_groups: Dict[str, Any]):
        """
        Batch allocate fields to the specified object.
        
        This method creates arrays with corresponding shapes for each field based on 
        field group configuration and sets them as attributes of the target object.
        
        Args:
            cls: Field class
            owner: Target object where fields will be allocated
            field_groups (dict): Field group configuration dictionary with format {shape_key: [field_names]}
                - shape_key (str): Shape key corresponding to shapes defined in Field._shape
                - field_names (list): List of field names for the corresponding shape
                
        Returns:
            list: List containing results of all setattr operations
            
        Usage Examples:
            # Example : Allocate fields for momentum module
            momentum_fields = {
                '2d': ['u', 'v'],           # 2D velocity fields
                '3d': ['temp', 'salt'],     # 3D temperature and salinity fields
                '2d_bgrid': ['h']           # Thickness field on B-grid
            }
            
        Note:
            - If field_names is an empty list, that shape group will be skipped
            - All fields are initialized with zeros
            - Field names cannot be duplicated, otherwise later ones will overwrite earlier ones
            - Shape keys must be predefined in Field._shape
        """
        return [setattr(owner, name, cls.new(shape_key)) \
            for shape_key, names in field_groups.items() \
                if names for name in names] 

    # Deallocate fields
    @classmethod
    def deallocate(cls, owner, field_groups):
        """
        Batch deallocate fields from the specified object.
        """
        return [delattr(owner, name) \
            for _, names in field_groups.items() \
                if names for name in names if hasattr(owner, name)]
    
    #
    @staticmethod
    def gather_tiles_with_id(data, tile):

        from mpi4py import MPI
        import numpy as np

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # 1. 收 tile id
        tile_ids = None
        if rank == 0:
            tile_ids = np.empty(size, dtype=np.int32)

        comm.Gather(
            sendbuf=np.array(tile, dtype=np.int32),
            recvbuf=tile_ids,
            root=0
        )

        # 2. 收数据
        recvbuf = None
        if rank == 0:
            recvbuf = np.empty((size, *data.shape), dtype=data.dtype)

        comm.Gather(
            sendbuf=data,
            recvbuf=recvbuf,
            root=0
        )
        # 3. rank 0 按 tile 重排
        if rank == 0:
            tiles = {}
            for r in range(size):
                tiles[tile_ids[r]] = recvbuf[r]

            # 保证 tile 顺序
            tiles_all = np.stack(
                [tiles[t] for t in sorted(tiles.keys())],
                axis=0
            )
            #print(tiles_all.shape)
            return tiles_all

        return None
