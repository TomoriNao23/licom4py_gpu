"""
File: field.py
Description: Field class for creating arrays with different backends (JAX, NumPy)
    in the LICOM ocean model backend calculation system.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-04 (Chtholly: add classmethod allocate and deallocate)
    
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any
from typing import NamedTuple

# Local application imports
from datatype import MpDomain

# BackendConfig
BackendConfig = NamedTuple(
    'BackendConfig',
    [
        ('platform', str), # Target platform: 'cpu' or 'gpu'
        ('lib', str),      # Backend library: 'jax' or 'numpy'
        ('precision', str) # Data precision for arrays: 'single' or 'double'
    ]
)

@dataclass(slots=True)
class Field:
    """
    Field class for creating arrays with different backends (JAX, NumPy).
    """
    
    _create_inf: dict
    _shape: dict
    dtype: Any

    @classmethod
    def init(cls, config: BackendConfig, mp: MpDomain) -> None:
        """
        Initialize the Field class with the given configuration and MP configuration.
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

        cls._setup_backend = lambda cfg: cls._setup_jax_backend(cfg) if cfg.lib == 'jax' else cls._setup_numpy_backend(cfg)
        cls._setup_backend(config)


    @classmethod
    def _setup_jax_backend(cls, config: BackendConfig) -> None:
        """Setup JAX backend with appropriate device configuration."""
        import jax

        cls.dtype = jax.numpy.float64 if config.precision == 'double' else jax.numpy.float32
        cls._create_inf = {'dtype': cls.dtype}
        if config.platform == 'cpu':
            cls._create_inf.update({'device': jax.devices("cpu")[0]})
        elif config.platform == 'gpu':
            cls._create_inf.update({'device': jax.devices("gpu")[0]})
        
        # Apply JIT compilation
        cls._set = jax.jit(lambda arr, idx, value: arr.at[idx].set(value))
        
        # Use lambda functions for array creation
        cls._new = lambda c, shape: jax.numpy.zeros(c._shape[shape], **c._create_inf)
        cls._array = lambda c, arr: jax.numpy.array(arr, **c._create_inf)

    @classmethod
    def _setup_numpy_backend(cls, config: BackendConfig) -> None:
        """Setup NumPy backend for array creation."""
        import numpy
        
        cls.dtype = numpy.float64 if config.precision == 'double' else numpy.float32
        cls._create_inf = {'dtype': cls.dtype}

        # Use lambda functions for array creation
        cls._new = lambda c, shape: numpy.zeros(c._shape[shape], **c._create_inf)
        cls._array = lambda c, arr: numpy.array(arr, **c._create_inf)
        cls._set = lambda arr, idx, value: (arr.__setitem__(idx, value), arr)[1]

    @classmethod
    def new(cls, shape: int):
        """
        Create a new zero array with specified shape.
        """
        return cls._new(cls, shape)
    
    @classmethod
    def array(cls, arr):
        """
        Convert input to array with specified backend.
        """
        return cls._array(cls, arr)

    @classmethod
    def set_(cls, arr, idx, value):
        """
        Set value at specified index.
        """
        return cls._set(arr, idx, value)


    # High-level Encapsulation
    @classmethod
    def allocate(cls, owner: Any, field_groups: dict) -> None:
        """
        Allocate and attach arrays to an owner object based on a mapping of
        shape-key -> list[field_names]. The shape-key must exist in Field._shape.
        """
        for shape_key, names in field_groups.items():
            if not names:
                continue
            for name in names:
                setattr(owner, name, cls.new(shape_key))

    @classmethod
    def deallocate(cls, owner: Any, field_groups: dict) -> None:
        """
        Delete previously attached arrays from an owner object using the same
        mapping that was passed to allocate().
        """
        for _, names in field_groups.items():
            if not names:
                continue
            for name in names:
                if hasattr(owner, name):
                    delattr(owner, name)