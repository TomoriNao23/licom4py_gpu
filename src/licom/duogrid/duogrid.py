"""
File: duogrid_data.py
Description: Duogrid data structure for grid management in LICOM ocean model.
    Ported from Fortran duogrid_data.F90 and duogrid_alloc.F90.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-04
Updated: 2025-09-19 (Chtholly add inner and outer masks)
"""

# Third-party imports
import numpy as np

# Local application imports
from backend.calculation.field import Field
from datatype import MpDate
from ._c_duogrid_function import duogrid_c_method
from .duogrid_cal import duogrid_cal

# Standard library imports
from typing import Any

@duogrid_c_method
@duogrid_cal
class Duogrid:
    """
    Duogrid data structure for grid management in LICOM ocean model.
    Use Fortran.Runtime.class gg to get all grid fields from cFMS global_grid.
    Use Python.backend Field to assign all grid fields to DuogridData.
    """
    
    # Class variables to store grid data
    # mp class
    mp : MpDate
    # calculation fields
    inner : Field.datatype
    outer : Field.datatype
    
    @classmethod
    def configure(cls, mp: MpDate) -> 'Duogrid':
        """
        Initialize Duogrid with MP domain configuration.
        
        Args:
            mp: MP domain configuration containing all necessary parameters
            
        Returns:
            Duogrid instance with initialized grid data
        """
        cls.mp = mp
        cls._map = {"nx": mp.nx, "ng": mp.ng, 
            "isd": mp.isd, "ied": mp.ied, 
            "jsd": mp.jsd, "jed": mp.jed, 
            "tile": mp.tile, "grid_type": 0}
        cls._init_from_cfms_global_grid()
        
        # Initialize calculation fields using decorator
        cls.init_calculations()
        
        return cls

    @classmethod
    def _init_from_cfms_global_grid(cls) -> None:
        """Populate duogrid fields using cFMS global_grid getters."""
          
        mp = cls.mp

        # set up global_grid (allocate Fortran.Runtime.class gg)
        cls.init_with_mp(**cls._map)

        # Fortran -> Numpy (Using Fortran.Runtime.class gg)
        a = cls.get_all_a_grid(mp.isd, mp.ied, mp.jsd, mp.jed)
        bcd = cls.get_all_bc_d_grid(mp.isd, mp.ied, mp.jsd, mp.jed)

        # Numpy -> Python (Using Python.backend Field)
        cls._assign_grid_fields(a, bcd) 
        
        # close global_grid (deallocate Fortran.Runtime.class gg)
        cls.end()

    @classmethod
    def _assign_grid_fields(cls, a: dict, bcd: dict) -> None:
        """
        Assign all grid fields from raw arrays to Field objects.
        
        Args:
            a: Dictionary containing A-grid arrays
            bcd: Dictionary containing B/C/D-grid arrays
        """
        # assign A-grid 2D (with Python.backend Field)
        cls.a_x = Field.array(a["a_x"])
        cls.a_y = Field.array(a["a_y"]) 
        cls.a_kik_x = Field.array(a["a_kik_x"]) 
        cls.a_kik_y = Field.array(a["a_kik_y"]) 
        cls.a_sina = Field.array(a["a_sina"]) 
        cls.a_cosa = Field.array(a["a_cosa"]) 
        cls.a_dx = Field.array(a["a_dx"])
        cls.a_dy = Field.array(a["a_dy"]) 
        cls.a_da = Field.array(a["a_da"]) 
        cls.rda = Field.array(a["rda"]) 
        cls.rdx = Field.array(a["rdx"]) 
        cls.rdy = Field.array(a["rdy"]) 
        cls.k2e_loc = Field.array(a["k2e_loc"])
        cls.a_f = Field.array(a["a_f"])
        cls.ub = Field.array(a["ub"])
        cls.vb = Field.array(a["vb"])
        
        # assign A-grid 3D (store in shape (ni, nj, 2))
        cls.k2e_coef = Field.array(np.transpose(a["k2e_coef"], (1, 2, 0)))
        cls.a_pt = Field.array(np.transpose(a["a_pt"], (1, 2, 0)))

        # assign 4D A-grid
        cls.a_gco = Field.array(np.transpose(a["a_gco"], (2, 3, 0, 1)))
        cls.a_gct = Field.array(np.transpose(a["a_gct"], (2, 3, 0, 1)))
        cls.a_c2l = Field.array(np.transpose(a["a_c2l"], (2, 3, 0, 1)))
        cls.a_l2c = Field.array(np.transpose(a["a_l2c"], (2, 3, 0, 1)))

        # B-grid / C-grid / D-grid
        cls.b_pt = Field.array(np.transpose(bcd["b_pt"], (1, 2, 0)))

        cls.c_gco = Field.array(np.transpose(bcd["c_gco"], (2, 3, 0, 1)))
        cls.c_gct = Field.array(np.transpose(bcd["c_gct"], (2, 3, 0, 1)))
        cls.c_ct2ort_x = Field.array(np.transpose(bcd["c_ct2ort_x"], (2, 3, 0, 1)))
        cls.c_ort2ct_x = Field.array(np.transpose(bcd["c_ort2ct_x"], (2, 3, 0, 1)))
        cls.c_sina = Field.array(bcd["c_sina"]) 
        cls.c_cosa = Field.array(bcd["c_cosa"]) 
        cls.c_dy = Field.array(bcd["c_dy"]) 
        cls.c_dx = Field.array(bcd["c_dx"])

        cls.d_gco = Field.array(np.transpose(bcd["d_gco"], (2, 3, 0, 1)))
        cls.d_gct = Field.array(np.transpose(bcd["d_gct"], (2, 3, 0, 1)))
        cls.d_ct2ort_y = Field.array(np.transpose(bcd["d_ct2ort_y"], (2, 3, 0, 1)))
        cls.d_ort2ct_y = Field.array(np.transpose(bcd["d_ort2ct_y"], (2, 3, 0, 1)))
        cls.d_sina = Field.array(bcd["d_sina"]) 
        cls.d_cosa = Field.array(bcd["d_cosa"]) 
        cls.d_dx = Field.array(bcd["d_dx"])
        cls.d_dy = Field.array(bcd["d_dy"])

        # test
        #Field.gather_tiles_with_id(a["a_x"], cls.mp.tile)
