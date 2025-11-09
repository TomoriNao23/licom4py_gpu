"""
File: jax.py
Description: JAX backend implementation for Field class.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-19
Updated: 2025-09-19
"""

import jax
from datatype import MpDate

def setup_jax_backend(cls, mp: MpDate) -> None:
    """Setup JAX backend with appropriate device configuration."""
    # Utilize precision
    (lambda precision: cls._create_inf.update({'dtype': jax.numpy.float64 \
        if precision == 'double' else jax.numpy.float32}))(mp.precision)

    # Utilize platform
    (lambda: cls._create_inf.update({'device': jax.devices(mp.platform)[0]}))()
    
    # Use lambda functions for array creation with JIT compilation
    cls._new = lambda c, shape: jax.numpy.zeros(c._shape[shape], **c._create_inf)
    cls._array = lambda c, arr: jax.numpy.array(arr, **c._create_inf)
    cls.datatype = jax.numpy.ndarray
    return None
