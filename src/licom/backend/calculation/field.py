"""
File: field.py
Description: Field class for creating arrays with different backends (JAX, NumPy)
    in the LICOM ocean model backend calculation system.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-04 (Chtholly: add classmethod allocate and deallocate)
Updated: 2025-09-16 (Chtholly: optimization)
    
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any

# Local application imports
from datatype import MpDate

@dataclass(slots=True)
class Field:
    """
    Field class for creating arrays with different backends (JAX, NumPy).
    """
    
    _create_inf: dict
    _shape: dict
    dtype: Any

    @classmethod
    def init(cls, mp: MpDate) -> None:
        """
        Initialize the Field class with the given MP configuration.
        """
        # Initialize shape configurations
        cls._shape = {
            '2d': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1),                       # (isd:ied, jsd:jed)
            '2d_bgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1 + 1),         # (isd:ied+1, jsd:jed+1)
            '2d_cgrid': (mp.ied - mp.isd + 1 + 1, mp.jed - mp.jsd + 1),             # (isd:ied+1, jsd:jed)
            '2d_dgrid': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1 + 1),             # (isd:ied, jsd:jed+1)

            '3d': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1, mp.npz),               # (isd:ied, jsd:jed, 2)
            '3d1': (mp.ied - mp.isd + 1, mp.jed - mp.jsd + 1, mp.npz + 1),          # (isd:ied, jsd:jed, 2)
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

        (lambda cfg: cls._setup_jax_backend(cfg) if cfg.lib == 'jax' else \
            cls._setup_numpy_backend(cfg))(mp)
        return None

    @classmethod
    def _setup_jax_backend(cls, mp: MpDate) -> None:
        """Setup JAX backend with appropriate device configuration."""
        import jax

        # Utilize precision
        (lambda precision: cls._create_inf.update({'dtype': jax.numpy.float64 \
            if precision == 'double' else jax.numpy.float32}))(mp.precision)

        # Utilize platform
        (lambda: cls._create_inf.update({'device': jax.devices(mp.platform)[0]}))()
        
        # Use lambda functions for array creation with JIT compilation
        cls._new = lambda c, shape: jax.numpy.zeros(c._shape[shape], **c._create_inf)
        cls._array = lambda c, arr: jax.numpy.array(arr, **c._create_inf)
        cls._set = jax.jit(lambda arr, idx, value: arr.at[idx].set(value))
        return None
        
    @classmethod
    def _setup_numpy_backend(cls, mp: MpDate) -> None:
        """Setup NumPy backend for array creation."""
        import numpy
            
        # Utilize precision
        (lambda precision: cls._create_inf.update({'dtype': numpy.float64 \
            if precision == 'double' else numpy.float32}))(mp.precision)

        # Use lambda functions for array creation
        cls._new = lambda c, shape: numpy.zeros(c._shape[shape], **c._create_inf)
        cls._array = lambda c, arr: numpy.array(arr, **c._create_inf)
        cls._set = lambda arr, idx, value: (arr.__setitem__(idx, value), arr)[1]
        return None

    @classmethod
    def new(cls, shape: int):
        """Create a new zero array with specified shape."""
        return cls._new(cls, shape)
    
    @classmethod
    def array(cls, arr):
        """Convert input to array with specified backend (no error checking)."""
        return cls._array(cls, arr)

    @classmethod
    def set_(cls, arr, idx, value):
        """Set value at specified index."""
        return cls._set(arr, idx, value)


    # High-level Encapsulation

    # Allocate fields
    allocate = lambda cls, owner, field_groups: \
        [setattr(owner, name, cls.new(shape_key)) \
            for shape_key, names in field_groups.items() \
                if names for name in names] 

    # Deallocate fields
    deallocate = lambda cls, owner, field_groups: \
        [delattr(owner, name) \
            for _, names in field_groups.items() \
                if names for name in names if hasattr(owner, name)]