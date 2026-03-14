"""
File: use_mpp.py
Description: Use the Chtholly FMS library to get the MPP configuration

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-16
Updated: 2025-09-28

REVISION HISTORY:
    16/09/2025 - Initial Python wrapper for Chtholly FMS library
    19/09/2025 - Added communication2d and ext_vector methods
    28/10/2025 - debug
"""


# Standard library imports
import ctypes
import os
from typing import Callable

import jax.numpy as jnp
import jax
from jax.experimental import io_callback

# Local application imports
from mp_data import MpDate

class FMS_chtholly:
    """Class to use the Chtholly FMS library"""

    # library of chtholly_fms
    _lib: ctypes.CDLL
    # mp class
    mp: MpDate
    # rmp flags
    rmp_w: bool
    rmp_e: bool
    rmp_s: bool
    rmp_n: bool
    rmp_sw: bool
    rmp_se: bool
    rmp_nw: bool
    rmp_ne: bool
    # public functions
    ext_scalar: Callable
    ext_vector: Callable
    communication2d: Callable

    @classmethod
    def configure(cls, size:int) -> None:

       cls._load_chtholly_library()
       cls._lib.chtholly_init(ctypes.c_int(size), ctypes.c_int(size), ctypes.c_int(1), ctypes.c_int(1))
       cls._mp_init(size)

    @classmethod
    def _load_chtholly_library(cls) -> None:
        """Load the Chtholly FMS library"""
        # Get the library path automatically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..")
        # Automatically support both .so and .dylib extensions
        lib_dir = os.path.join(project_root, "lib", "lib")
        so_path = os.path.join(lib_dir, "libfms_unified.so")
        dylib_path = os.path.join(lib_dir, "libfms_unified.dylib")
        if os.path.exists(so_path):
            lib_path = so_path
        elif os.path.exists(dylib_path):
            lib_path = dylib_path
        else:
            raise FileNotFoundError(f"Cannot find shared library: {so_path} or {dylib_path}")
        #lib_path = os.path.join("/data/yyq/data01/mls/licom4py/licom4py/lib/lib/libfms_unified.so")
        cls._lib = ctypes.CDLL(lib_path)
        cls._define_c_size()

    @classmethod
    def end(cls) -> None:
        """End the Chtholly FMS library"""
        if hasattr(cls, '_lib'):
            cls._lib.chtholly_end()

    @classmethod
    def _mp_init(cls, size:int) -> None:
        """Create the mp type"""
        mpdict = {
            'ng': 3,
            # from namelist.input
            'nx': size,
            'ny': size,
            # from Chtholly FMS wrapper
            'tile': cls._lib.chtholly_get_tile(),
            'pe': cls._lib.chtholly_get_mpp_pe(),
            'xsize': cls._lib.chtholly_get_xsize(),
            'ysize': cls._lib.chtholly_get_ysize(),
            'is_': cls._lib.chtholly_get_is(),
            'ie': cls._lib.chtholly_get_ie(),
            'js': cls._lib.chtholly_get_js(),
            'je': cls._lib.chtholly_get_je(),
            'isd': cls._lib.chtholly_get_isd(),
            'ied': cls._lib.chtholly_get_ied(),
            'jsd': cls._lib.chtholly_get_jsd(),
            'jed': cls._lib.chtholly_get_jed(),
        }
        # Initialize mp
        cls.mp = MpDate(**mpdict)

        return None


    @classmethod
    def _define_c_size(cls) -> None:
        """Define the c size"""
        # Define function signatures
        cls._lib.chtholly_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        cls._lib.chtholly_init.restype = None
        
        cls._lib.chtholly_end.argtypes = []
        cls._lib.chtholly_end.restype = None
        
        cls._lib.chtholly_get_tile.argtypes = []
        cls._lib.chtholly_get_tile.restype = int
        
        cls._lib.chtholly_get_xsize.argtypes = []
        cls._lib.chtholly_get_xsize.restype = int
        
        cls._lib.chtholly_get_ysize.argtypes = []
        cls._lib.chtholly_get_ysize.restype = int

        cls._lib.chtholly_get_mpp_pe.argtypes = []
        cls._lib.chtholly_get_mpp_pe.restype = int

        cls._lib.chtholly_get_is.argtypes = []
        cls._lib.chtholly_get_is.restype = int

        cls._lib.chtholly_get_ie.argtypes = []
        cls._lib.chtholly_get_ie.restype = int

        cls._lib.chtholly_get_js.argtypes = []
        cls._lib.chtholly_get_js.restype = int

        cls._lib.chtholly_get_je.argtypes = []
        cls._lib.chtholly_get_je.restype = int

        cls._lib.chtholly_get_isd.argtypes = []
        cls._lib.chtholly_get_isd.restype = int

        cls._lib.chtholly_get_ied.argtypes = []
        cls._lib.chtholly_get_ied.restype = int

        cls._lib.chtholly_get_jsd.argtypes = []
        cls._lib.chtholly_get_jsd.restype = int

        cls._lib.chtholly_get_jed.argtypes = []
        cls._lib.chtholly_get_jed.restype = int

        cls._lib.chtholly_ext_scalar_2d.argtypes = [ctypes.POINTER(ctypes.c_double)]
        cls._lib.chtholly_ext_scalar_2d.restype = None

        cls._lib.chtholly_communication2d.argtypes = [
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_double)
        ]
        cls._lib.chtholly_communication2d.restype = None

        return None