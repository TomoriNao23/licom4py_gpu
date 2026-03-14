"""
File: c_global_grid.py
Description: Python ctypes wrapper for c_global_grid getters, localized under licom.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-11
Updated: 2025-09-16
"""

# Standard library imports
import os
from ctypes import c_int, c_double, CDLL, POINTER
from typing import Tuple

# Third-party imports
import numpy as np


_lib = None
_defined = False


def _libc():
    global _lib
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    project_root = os.path.abspath(project_root)
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
    _lib = CDLL(lib_path)
    return _lib


def _define_once():
    global _defined
    if _defined:
        return
    lib = _libc()

    npptr = np.ctypeslib.ndpointer
    F = "F_CONTIGUOUS"

    lib.chtholly_global_grid_init.restype = None
    lib.chtholly_global_grid_init.argtypes = [
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int
    ]
    lib.chtholly_global_grid_end.restype = None
    lib.chtholly_global_grid_end.argtypes = None

    # 2D A-grid
    lib.chtholly_global_grid_get_a_x_dg.restype = None
    lib.chtholly_global_grid_get_a_x_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_y_dg.restype = None
    lib.chtholly_global_grid_get_a_y_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_kik_x_dg.restype = None
    lib.chtholly_global_grid_get_a_kik_x_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_kik_y_dg.restype = None
    lib.chtholly_global_grid_get_a_kik_y_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_sina_dg.restype = None
    lib.chtholly_global_grid_get_a_sina_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_cosa_dg.restype = None
    lib.chtholly_global_grid_get_a_cosa_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_dx_dg.restype = None
    lib.chtholly_global_grid_get_a_dx_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_dy_dg.restype = None
    lib.chtholly_global_grid_get_a_dy_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_da_dg.restype = None
    lib.chtholly_global_grid_get_a_da_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_rda_dg.restype = None
    lib.chtholly_global_grid_get_rda_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_rdx_dg.restype = None
    lib.chtholly_global_grid_get_rdx_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_rdy_dg.restype = None
    lib.chtholly_global_grid_get_rdy_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_k2e_loc_dg.restype = None
    lib.chtholly_global_grid_get_k2e_loc_dg.argtypes = [POINTER(c_int)]
    lib.chtholly_global_grid_get_a_f_dg.restype = None
    lib.chtholly_global_grid_get_a_f_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_ub.restype = None
    lib.chtholly_global_grid_get_ub.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_vb.restype = None
    lib.chtholly_global_grid_get_vb.argtypes = [POINTER(c_double)]

    # 3D A-grid
    lib.chtholly_global_grid_get_a_pt_ext.restype = None
    lib.chtholly_global_grid_get_a_pt_ext.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_k2e_coef_dg.restype = None
    lib.chtholly_global_grid_get_k2e_coef_dg.argtypes = [POINTER(c_double)]

    # 4D A-grid
    lib.chtholly_global_grid_get_a_gco_dg.restype = None
    lib.chtholly_global_grid_get_a_gco_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_gct_dg.restype = None
    lib.chtholly_global_grid_get_a_gct_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_c2l_dg.restype = None
    lib.chtholly_global_grid_get_a_c2l_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_a_l2c_dg.restype = None
    lib.chtholly_global_grid_get_a_l2c_dg.argtypes = [POINTER(c_double)]

    # 3D B-grid
    lib.chtholly_global_grid_get_b_pt_dg.restype = None
    lib.chtholly_global_grid_get_b_pt_dg.argtypes = [POINTER(c_double)]

    # 4D/2D C-grid
    lib.chtholly_global_grid_get_c_gco_dg.restype = None
    lib.chtholly_global_grid_get_c_gco_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_gct_dg.restype = None
    lib.chtholly_global_grid_get_c_gct_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_ct2ort_x_dg.restype = None
    lib.chtholly_global_grid_get_c_ct2ort_x_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_ort2ct_x_dg.restype = None
    lib.chtholly_global_grid_get_c_ort2ct_x_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_sina_dg.restype = None
    lib.chtholly_global_grid_get_c_sina_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_cosa_dg.restype = None
    lib.chtholly_global_grid_get_c_cosa_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_dy_dg.restype = None
    lib.chtholly_global_grid_get_c_dy_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_c_dx_dg.restype = None
    lib.chtholly_global_grid_get_c_dx_dg.argtypes = [POINTER(c_double)]

    # 4D/2D D-grid
    lib.chtholly_global_grid_get_d_gco_dg.restype = None
    lib.chtholly_global_grid_get_d_gco_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_gct_dg.restype = None
    lib.chtholly_global_grid_get_d_gct_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_ct2ort_y_dg.restype = None
    lib.chtholly_global_grid_get_d_ct2ort_y_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_ort2ct_y_dg.restype = None
    lib.chtholly_global_grid_get_d_ort2ct_y_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_sina_dg.restype = None
    lib.chtholly_global_grid_get_d_sina_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_cosa_dg.restype = None
    lib.chtholly_global_grid_get_d_cosa_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_dx_dg.restype = None
    lib.chtholly_global_grid_get_d_dx_dg.argtypes = [POINTER(c_double)]
    lib.chtholly_global_grid_get_d_dy_dg.restype = None
    lib.chtholly_global_grid_get_d_dy_dg.argtypes = [POINTER(c_double)]

    _defined = True


def init_with_mp(nx: int, ng: int, tile: int, isd: int, ied: int, jsd: int, jed: int, grid_type: int = 0) -> None:
    """Initialize c_global_grid with parameters from MpDate.

    res is approximated by nx here; ng and tile are passed through.
    """
    _define_once()
    _libc().chtholly_global_grid_init(int(nx), int(ng), int(grid_type), int(tile), int(isd), int(ied), int(jsd), int(jed))


def end() -> None:
    _define_once()
    _libc().chtholly_global_grid_end()


def _shape_2d(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int]:
    return (ied - isd + 1, jed - jsd + 1)


def _shape_3d_agrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int, int]:
    ni, nj = _shape_2d(isd, ied, jsd, jed)
    return (2, ni, nj)


def _shape_4d_agrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int, int, int]:
    ni, nj = _shape_2d(isd, ied, jsd, jed)
    return (2, 2, ni, nj)


def _shape_2d_cgrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int]:
    return (ied - isd + 2, jed - jsd + 1)


def _shape_2d_dgrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int]:
    return (ied - isd + 1, jed - jsd + 2)


def _shape_3d_bgrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int, int]:
    return (2, ied - isd + 2, jed - jsd + 2)


def _shape_4d_cgrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int, int, int]:
    return (2, 2, ied - isd + 2, jed - jsd + 1)


def _shape_4d_dgrid(isd: int, ied: int, jsd: int, jed: int) -> Tuple[int, int, int, int]:
    return (2, 2, ied - isd + 1, jed - jsd + 2)


def get_all_a_grid(isd: int, ied: int, jsd: int, jed: int):
    _define_once()
    ni_nj = _shape_2d(isd, ied, jsd, jed)
    arrs = {}
    for name in [
        ("a_x", _libc().chtholly_global_grid_get_a_x_dg),
        ("a_y", _libc().chtholly_global_grid_get_a_y_dg),
        ("a_kik_x", _libc().chtholly_global_grid_get_a_kik_x_dg),
        ("a_kik_y", _libc().chtholly_global_grid_get_a_kik_y_dg),
        ("a_sina", _libc().chtholly_global_grid_get_a_sina_dg),
        ("a_cosa", _libc().chtholly_global_grid_get_a_cosa_dg),
        ("a_dx", _libc().chtholly_global_grid_get_a_dx_dg),
        ("a_dy", _libc().chtholly_global_grid_get_a_dy_dg),
        ("a_da", _libc().chtholly_global_grid_get_a_da_dg),
        ("rda", _libc().chtholly_global_grid_get_rda_dg),
        ("rdx", _libc().chtholly_global_grid_get_rdx_dg),
        ("rdy", _libc().chtholly_global_grid_get_rdy_dg),
        ("a_f", _libc().chtholly_global_grid_get_a_f_dg),
        ("ub", _libc().chtholly_global_grid_get_ub),
        ("vb", _libc().chtholly_global_grid_get_vb),
    ]:
        arr = np.empty(ni_nj, dtype=np.float64, order="F")
        name[1](arr.ctypes.data_as(POINTER(c_double)))
        arrs[name[0]] = arr

    arr = np.empty(_shape_3d_agrid(isd, ied, jsd, jed), dtype=np.float64, order="F")
    _libc().chtholly_global_grid_get_a_pt_ext(arr.ctypes.data_as(POINTER(c_double)))
    arrs["a_pt"] = arr

    arr = np.empty(_shape_3d_agrid(isd, ied, jsd, jed), dtype=np.float64, order="F")
    _libc().chtholly_global_grid_get_k2e_coef_dg(arr.ctypes.data_as(POINTER(c_double)))
    arrs["k2e_coef"] = arr

    arr_i = np.empty(ni_nj, dtype=np.int32, order="F")
    _libc().chtholly_global_grid_get_k2e_loc_dg(arr_i.ctypes.data_as(POINTER(c_int)))
    arrs["k2e_loc"] = arr_i

    for name in [
        ("a_gco", _libc().chtholly_global_grid_get_a_gco_dg),
        ("a_gct", _libc().chtholly_global_grid_get_a_gct_dg),
        ("a_c2l", _libc().chtholly_global_grid_get_a_c2l_dg),
        ("a_l2c", _libc().chtholly_global_grid_get_a_l2c_dg),
    ]:
        arr4 = np.empty(_shape_4d_agrid(isd, ied, jsd, jed), dtype=np.float64, order="F")
        name[1](arr4.ctypes.data_as(POINTER(c_double)))
        arrs[name[0]] = arr4

    return arrs


def get_all_bc_d_grid(isd: int, ied: int, jsd: int, jed: int):
    _define_once()
    arrs = {}
    arr = np.empty(_shape_3d_bgrid(isd, ied, jsd, jed), dtype=np.float64, order="F")
    _libc().chtholly_global_grid_get_b_pt_dg(arr.ctypes.data_as(POINTER(c_double)))
    arrs["b_pt"] = arr

    for name, fn, shape in [
        ("c_gco", _libc().chtholly_global_grid_get_c_gco_dg, _shape_4d_cgrid),
        ("c_gct", _libc().chtholly_global_grid_get_c_gct_dg, _shape_4d_cgrid),
        ("c_ct2ort_x", _libc().chtholly_global_grid_get_c_ct2ort_x_dg, _shape_4d_cgrid),
        ("c_ort2ct_x", _libc().chtholly_global_grid_get_c_ort2ct_x_dg, _shape_4d_cgrid),
    ]:
        arr4 = np.empty(shape(isd, ied, jsd, jed), dtype=np.float64, order="F")
        fn(arr4.ctypes.data_as(POINTER(c_double)))
        arrs[name] = arr4

    for name, fn, shape in [
        ("c_sina", _libc().chtholly_global_grid_get_c_sina_dg, _shape_2d_cgrid),
        ("c_cosa", _libc().chtholly_global_grid_get_c_cosa_dg, _shape_2d_cgrid),
        ("c_dy", _libc().chtholly_global_grid_get_c_dy_dg, _shape_2d_cgrid),
        ("c_dx", _libc().chtholly_global_grid_get_c_dx_dg, _shape_2d_cgrid),
    ]:
        arr2 = np.empty(shape(isd, ied, jsd, jed), dtype=np.float64, order="F")
        fn(arr2.ctypes.data_as(POINTER(c_double)))
        arrs[name] = arr2

    for name, fn, shape in [
        ("d_gco", _libc().chtholly_global_grid_get_d_gco_dg, _shape_4d_dgrid),
        ("d_gct", _libc().chtholly_global_grid_get_d_gct_dg, _shape_4d_dgrid),
        ("d_ct2ort_y", _libc().chtholly_global_grid_get_d_ct2ort_y_dg, _shape_4d_dgrid),
        ("d_ort2ct_y", _libc().chtholly_global_grid_get_d_ort2ct_y_dg, _shape_4d_dgrid),
    ]:
        arr4 = np.empty(shape(isd, ied, jsd, jed), dtype=np.float64, order="F")
        fn(arr4.ctypes.data_as(POINTER(c_double)))
        arrs[name] = arr4

    for name, fn, shape in [
        ("d_sina", _libc().chtholly_global_grid_get_d_sina_dg, _shape_2d_dgrid),
        ("d_cosa", _libc().chtholly_global_grid_get_d_cosa_dg, _shape_2d_dgrid),
        ("d_dx", _libc().chtholly_global_grid_get_d_dx_dg, _shape_2d_dgrid),
        ("d_dy", _libc().chtholly_global_grid_get_d_dy_dg, _shape_2d_dgrid),
    ]:
        arr2 = np.empty(shape(isd, ied, jsd, jed), dtype=np.float64, order="F")
        fn(arr2.ctypes.data_as(POINTER(c_double)))
        arrs[name] = arr2

    return arrs


def duogrid_c_method(cls):
    """
    Class decorator that adds all c_duogrid functions as class methods to the decorated class.
    
    This decorator takes all the functions defined in this module and adds them as 
    class methods to the decorated class, allowing the class to use these C library functions
    as class methods.
    
    Args:
        cls: The class to be decorated (should be Duogrid)
        
    Returns:
        The decorated class with all c_duogrid functions as class methods
    """
    # Get all functions from this module that should be added as methods
    functions_to_add = [
        'init_with_mp',
        'end', 
        'get_all_a_grid',
        'get_all_bc_d_grid',
    ]
    
    # Add each function as a class method to the class
    for func_name in functions_to_add:
        if func_name in globals():
            func = globals()[func_name]
            
            # Create a wrapper method that ignores cls parameter
            def make_classmethod(original_func):
                def classmethod_wrapper(cls, *args, **kwargs):
                    return original_func(*args, **kwargs)
                return classmethod(classmethod_wrapper)
            
            # Bind the wrapped function as a class method to the class
            setattr(cls, func_name, make_classmethod(func))
    
    return cls