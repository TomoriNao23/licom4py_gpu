"""
File: numpy.py
Description: NumPy backend implementation for Field class with JAX-like .at[].set() syntax support.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-19
Updated: 2025-09-19

"""

import numpy
from datatype import MpDate

def setup_numpy_backend(cls, mp: MpDate) -> None:
    """Setup NumPy backend for array creation."""
    # Utilize precision
    (lambda precision: cls._create_inf.update({'dtype': numpy.float64 \
        if precision == 'double' else numpy.float32}))(mp.precision)

    # Use lambda functions for array creation with wrapper
    cls._new = lambda c, shape: NumpyArrayWrapper(numpy.zeros(c._shape[shape], **c._create_inf))
    cls._array = lambda c, arr: NumpyArrayWrapper(numpy.array(arr, **c._create_inf))
    cls.datatype = numpy.ndarray
    return None

class NumpyArrayWrapper:
    """Wrapper for NumPy arrays to support JAX-like .at[].set() syntax"""
    
    def __init__(self, array):
        self._array = array
    
    def __getattr__(self, name):
        return getattr(self._array, name)
    
    def __getitem__(self, key):
        return self._array[key]
    
    def __setitem__(self, key, value):
        self._array[key] = value
    
    def __array__(self):
        return self._array
    
    @property
    def at(self):
        return NumpyAtWrapper(self._array)

class NumpyAtWrapper:
    """Wrapper for NumPy arrays to support .at[].set() syntax"""
    
    def __init__(self, array):
        self._array = array
    
    def __getitem__(self, key):
        return NumpySetWrapper(self._array, key)

class NumpySetWrapper:
    """Wrapper for NumPy arrays to support .set() method"""
    
    def __init__(self, array, key):
        self._array = array
        self._key = key
    
    def set(self, value):
        """Set value at the specified key and return the array"""
        self._array[self._key] = value
        return NumpyArrayWrapper(self._array)