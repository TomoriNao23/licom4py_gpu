"""
File: duogrid_data.py
Description: Data container for duogrid arrays and metadata used by LICOM.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-04
Updated: 2025-09-16
"""

# Standard library imports
from dataclasses import dataclass, field

# Local application imports
from .mp_data import MpDate

@dataclass
class DuogridData:
    """
    Duogrid data structure for grid management.
    
    The grid layout follows the C-grid convention:
    bpt(i,j+1) -- dpt(i,j+1) -- bpt(i+1,j+1)
         |             |             |
    cpt(i,j  ) -- apt(i,j  ) -- cpt(i+1,j  )
         |             |             |
    bpt(i,j  ) -- dpt(i,j  ) -- bpt(i+1,j  )
    """

    _field = {
        # 2D fields on A-grid (isd:ied, jsd:jed)
        '2d': [
            'a_x', 'a_y', 'a_kik_x', 'a_kik_y',
            'a_da', 'rda', 'a_sina', 'a_cosa',
            'a_dx', 'a_dy', 'rdx', 'rdy',
            'a_f', 'a_kik_f', 'k2e_loc'
        ],
        
        # 2D fields on B-grid (isd:ied+1, jsd:jed+1)
        '2d_bgrid': [],
        
        # 2D fields on C-grid (isd:ied+1, jsd:jed)
        '2d_cgrid': ['c_dy', 'c_sina', 'c_cosa'],
        
        # 2D fields on D-grid (isd:ied, jsd:jed+1)
        '2d_dgrid': ['d_dx', 'd_sina', 'd_cosa'],

        # 3D fields on A-grid (isd:ied, jsd:jed, 2)
        '3d_agrid': ['a_pt', 'k2e_coef'],
        
        # 3D fields on B-grid (isd:ied+1, jsd:jed+1, 2)
        '3d_bgrid': ['b_pt'],
        
        # 3D fields on C-grid (isd:ied+1, jsd:jed, 2)
        '3d_cgrid': ['c_pt'],
        
        # 3D fields on D-grid (isd:ied, jsd:jed+1, 2)
        '3d_dgrid': ['d_pt'],

        # 4D fields on A-grid (isd:ied, jsd:jed, 2, 2)
        '4d_agrid': [
            'a_gco', 'a_gct', 'a_c2l', 'a_l2c',
            'a_kik_gco', 'a_kik_gct', 'a_kik_c2l', 'a_kik_l2c'
        ],
        
        # 4D fields on B-grid (isd:ied+1, jsd:jed+1, 2, 2)
        '4d_bgrid': [],
        
        # 4D fields on C-grid (isd:ied+1, jsd:jed, 2, 2)
        '4d_cgrid': ['c_gco', 'c_gct', 'c_ct2ort_x', 'c_ort2ct_x'],
        
        # 4D fields on D-grid (isd:ied, jsd:jed+1, 2, 2)
        '4d_dgrid': ['d_gco', 'd_gct', 'd_ct2ort_y', 'd_ort2ct_y']
    }

    mp: MpDate

    k2e_nord: int = 2