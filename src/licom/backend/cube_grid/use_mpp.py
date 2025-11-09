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
import functools
from typing import Callable

import jax.numpy as jnp
import jax

# Local application imports
from mymodule.timer import timed, Timer
from datatype import MpDate
from readnamelist import Namelist

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
    def init(cls, namelist: Namelist) -> None:

       cls._load_chtholly_library()
       cls._lib.chtholly_init(namelist.nx, namelist.ny, namelist.npes_x, namelist.npes_y)
       cls._mp_init(namelist)

    @classmethod
    def _load_chtholly_library(cls) -> None:
        """Load the Chtholly FMS library"""
        # Get the library path automatically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..", "..", "..")
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
    def _mp_init(cls, namelist: Namelist) -> None:
        """Create the mp type"""
        mpdict = {
            'ng': 3,
            # from namelist.input
            'nx': namelist.nx,
            'ny': namelist.ny,
            'npz': namelist.npz,
            'lib': namelist.lib,
            'platform': namelist.platform,
            'precision': namelist.precision,
            'dtb': namelist.barotropic_dt,
            'dtc': namelist.baroclinic_dt,
            'nbb': int(namelist.baroclinic_dt / namelist.barotropic_dt),
            'rk_barotr': namelist.rk_barotr,
            'case': namelist.case,
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
        cls._rmp_flag()

        # Initialize library functions for backend
        (lambda lib: cls._setup_jax() if lib == 'jax' 
         else cls._setup_numpy())(cls.mp.lib)

        return None
    
    @classmethod
    def _rmp_flag(cls) -> None:
        """Set the rmp flag"""
        # locate the domain
        cls.rmp_w = True if cls.mp.is_ == 1 else False
        cls.rmp_e = True if cls.mp.ie == cls.mp.nx else False
        cls.rmp_s = True if cls.mp.js == 1 else False
        cls.rmp_n = True if cls.mp.je == cls.mp.ny else False

        cls.rmp_sw = cls.rmp_s and cls.rmp_w
        cls.rmp_se = cls.rmp_s and cls.rmp_e
        cls.rmp_nw = cls.rmp_n and cls.rmp_w
        cls.rmp_ne = cls.rmp_n and cls.rmp_e

    @classmethod
    def _setup_jax(cls) -> None:
        """Setup JAX library functions for backend."""
        import jax.numpy as jnp
        import numpy as np
        import jax
        from backend import Field
        
        # Communication functions using MPP
        def _update_domain_jax(var: jnp.ndarray) -> jnp.ndarray:
            """Exchange scalar values for JAX arrays."""
            # Convert JAX array to NumPy for C function
            np_var = np.array(var)
            # Use C function to exchange scalar values
            c_data = np_var.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            cls._lib.chtholly_ext_scalar_2d(c_data)
            # Convert back to JAX array
            return Field.array(np_var)

        # Communication functions using MPP
        @timed('remap.communication')
        def _communication2d_jax(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Exchange vector values for JAX arrays."""
            # Convert JAX array to NumPy for C function
            np_u = np.array(u)
            np_v = np.array(v)
            # Use C function to exchange vector values
            c_data_u = np_u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_data_v = np_v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            cls._lib.chtholly_communication2d(c_data_u, c_data_v)
            # Convert back to JAX array
            return Field.array(np_u), Field.array(np_v)
        
        # jax.jit for the transform function (pure calculation)
        @jax.jit
        def _transform(u: jnp.ndarray, v: jnp.ndarray, k: jnp.ndarray,
                       range: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Transform the vector values"""
            ull = (k[..., 0, 0] * u + k[..., 0, 1] * v) * range
            vll = (k[..., 1, 0] * u + k[..., 1, 1] * v) * range
            return ull, vll
        
        # private functions for jax backend
        cls._update_domain = _update_domain_jax
        cls._transform = _transform
        # public functions for jax backend
        cls.communication2d = _communication2d_jax

        return None

    @classmethod
    def _setup_numpy(cls) -> None:
        """Setup NumPy library functions for backend."""
        import numpy as np
        
        # Communication functions using MPP
        def _update_domain_numpy(var: np.ndarray) -> np.ndarray:
            """Exchange scalar values for NumPy arrays."""
            # Convert NumPy array to C pointer
            c_data = var.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            cls._lib.chtholly_ext_scalar_2d(c_data)
            return var

        # Communication functions using MPP
        def _communication2d_numpy(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Exchange vector values for NumPy arrays."""
            # Convert NumPy array to C pointer
            c_data_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_data_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            cls._lib.chtholly_communication2d(c_data_u, c_data_v)
            return u, v

        # transform function (pure calculation)
        def _transform(u: np.ndarray, v: np.ndarray, 
                       k: np.ndarray, range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Transform the vector values"""
            ull = (k[..., 0, 0] * u + k[..., 0, 1] * v) * range
            vll = (k[..., 1, 0] * u + k[..., 1, 1] * v) * range
            return ull, vll
        
        # private functions for numpy backend
        cls._update_domain = _update_domain_numpy
        cls._transform = _transform
        # public functions for numpy backend
        cls.communication2d = _communication2d_numpy

        return None

    @classmethod
    def ext_scalar(cls, var):
        """
        Exchange scalar values across domain boundaries.
        """
        with Timer('remap.communication'):
            a = cls._update_domain(var)
        b = cls._cube_rmp(a)
        return b

    @classmethod
    def ext_vector(cls, u, v):
        """
        Exchange vector values across domain boundaries.
        """
        from duogrid.duogrid import Duogrid as Dg
        ull, vll = cls._transform(u, v, Dg.a_c2l, Dg.inner)
        ull = cls.ext_scalar(ull)
        vll = cls.ext_scalar(vll)
        ull, vll = cls._transform(ull, vll, Dg.a_l2c, Dg.outer)
        ull = ull + u * Dg.inner
        vll = vll + v * Dg.inner

        return ull, vll

    @classmethod
    def _cube_rmp(cls, var):
        """
        Port of Fortran subroutine cube_rmp(var, dg) to Python.

        Convert halo/edge values on a cubed-sphere tile using precomputed
        k2e mappings and coefficients stored in `dg`.

        Args:
            var: 2D array-like Field for which boundary remapping is applied.

        Returns:
            The updated `var` after remapping (for JAX it returns a new array).
        """
        from duogrid.duogrid import Duogrid
        
        # Extract parameters for JIT-compatible function
        mp = cls.mp
        isd, jsd, ied, jed = mp.isd, mp.jsd, mp.ied, mp.jed
        is_, js, ie, je = mp.is_, mp.js, mp.ie, mp.je
        
        # Extract remapping flags
        rmp_s, rmp_n, rmp_w, rmp_e = cls.rmp_s, cls.rmp_n, cls.rmp_w, cls.rmp_e
        
        # Extract k2e parameters
        coef = Duogrid.k2e_coef  # (ni, nj, nord)
        loc_arr = Duogrid.k2e_loc  # (ni, nj) Fortran physical indices
        
        # Call JIT-compiled function
        return cls._cube_rmp_jit(var, isd, jsd, ied, jed, is_, js, ie, je, 
                                rmp_s, rmp_n, rmp_w, rmp_e, coef, loc_arr)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    def _cube_rmp_jit(var, isd, jsd, ied, jed, is_, js, ie, je, 
                     rmp_s, rmp_n, rmp_w, rmp_e, coef, loc_arr):
        """
        JIT-compiled version of cube_rmp function.
        
        Args:
            var: 2D JAX array for which boundary remapping is applied
            isd, jsd, ied, jed: Domain indices
            is_, js, ie, je: Local domain indices  
            rmp_s, rmp_n, rmp_w, rmp_e: Remapping flags for each boundary
            coef: k2e coefficients array
            loc_arr: k2e location array
            
        Returns:
            The updated `var` after remapping
        """
        # Work array
        var_kik = var
        
        # Constants
        nord = 2
        offset = 1

        # South boundary
        if rmp_s:
            # ii = 1
            j = js - 1
            # Use array operations to remove loops
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i_indices-isd, j-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )
            # ii = 2
            j = js - 2
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            var = var.at[i_indices-isd, j-jsd].set(0)
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )
            # ii = 3
            j = js - 3
            # Use array operations to remove loops
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i_indices-isd, j-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )

        # North boundary
        if rmp_n:
            # Manually unroll ii loop, ng=3, ii=1,2,3
            # ii = 1
            j = je + 1
            # Use array operations to remove loops
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i_indices-isd, j-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )
            # ii = 2
            j = je + 2
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            var = var.at[i_indices-isd, j-jsd].set(0)
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )
            # ii = 3
            j = je + 3
            # Use array operations to remove loops
            i_indices = jnp.arange(is_, ie + 1)
            locs = loc_arr[i_indices-isd, j-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i_indices-isd, j-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+1-isd, j-jsd] * coef[i_indices-isd, j-jsd, 0]
            )
            var = var.at[i_indices-isd, j-jsd].set(
                var[i_indices-isd, j-jsd] +
                var_kik[los+2-isd, j-jsd] * coef[i_indices-isd, j-jsd, 1]
            )
                  
        # West boundary
        if rmp_w:
            # Manually unroll ii loop, ng=3, ii=1,2,3
            # ii = 1
            i = is_ - 1
            # Use array operations to remove loops
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i-isd, j_indices-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )
            # ii = 2
            i = is_ - 2
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            var = var.at[i-isd, j_indices-jsd].set(0)
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )
            # ii = 3
            i = is_ - 3
            # Use array operations to remove loops
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i-isd, j_indices-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )

        # East boundary
        if rmp_e:
            # Manually unroll ii loop, ng=3, ii=1,2,3
            # ii = 1
            i = ie + 1
            # Use array operations to remove loops
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i-isd, j_indices-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )
            # ii = 2
            i = ie + 2
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            var = var.at[i-isd, j_indices-jsd].set(0)
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )
            # ii = 3
            i = ie + 3
            # Use array operations to remove loops
            j_indices = jnp.arange(js, je + 1)
            locs = loc_arr[i-isd, j_indices-jsd].astype(int)
            los = locs - offset
            # First set target positions to zero
            var = var.at[i-isd, j_indices-jsd].set(0)
            # Calculate weighted sum
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+1-jsd] * coef[i-isd, j_indices-jsd, 0]
            )
            var = var.at[i-isd, j_indices-jsd].set(
                var[i-isd, j_indices-jsd] +
                var_kik[i-isd, los+2-jsd] * coef[i-isd, j_indices-jsd, 1]
            )

        # Copy corners (need to check if rmp)
        # sw
        var = var.at[0:is_-isd, 0:js-jsd].set(var[is_-isd, 0:js-jsd])

        # se
        var = var.at[ie+1-isd:ied+1-isd, 0:js-jsd].set(var[ie-isd, 0:js-jsd])

        # ne
        var = var.at[ie+1-isd:ied+1-isd, je+1-jsd:jed+1-jsd].set(var[ie-isd, je+1-jsd:jed+1-jsd])

        # nw
        var = var.at[0:is_-isd, je+1-jsd:jed+1-jsd].set(var[is_-isd, je+1-jsd:jed+1-jsd])

        return var


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