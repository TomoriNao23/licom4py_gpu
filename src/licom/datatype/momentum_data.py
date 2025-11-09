"""
File: momentum_data.py
Description: Updated MomentumData class with all fields required for barotropic time stepping.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2025-09-24
REVISION HISTORY:
    2025-09-22 - Chtholly added missing fields for barotropic computation
    2025-09-24 - Chtholly added LMARS related fields
"""

from .data_abc import DataABC


class MomentumData(DataABC):
    """
    Momentum data container with all fields required for barotropic and baroclinic computations.
    """

    _field = {
        # 2D fields on A-grid (isd:ied, jsd:jed)
        '2d': [
            # Primary velocity and SSH fields
            'ub', 'vb',           # Barotropic velocities (current step)
            'ubp', 'vbp',         # Barotropic velocities (previous step) 
            'h0', 'h0p',          # Sea surface height (current and previous)
            
            # Advection terms
            'advx', 'advy',       # Advection terms for momentum equations
            
            # Contravariant and covariant velocity components
            'ub_ct', 'vb_ct',     # Contravariant velocities on A-grid
            'ub_cx', 'vb_cy',     # Contravariant velocities on C/D-grids (flux)
            'vb_cx', 'ub_cy',     # Contravariant velocities on D/C-grids (vorticity)
            
            # Time-averaged fields
            'h0f', 'h0bf',        # Time-averaged SSH fields
            
            # Pressure and force terms
            'pax', 'pay',         # Pressure gradient terms
            'pxb', 'pyb',         # Additional pressure terms
            'whx', 'why',         # Horizontal mixing terms
            
            # Viscosity and diffusion terms (if needed)
            'dlub', 'dlvb',       # Laplacian of velocity (for viscosity)
            
            # Work arrays
            'wgp',     

            # lmars related terms   
            'celerity_x', 'celerity_y',        
        ],
        
        # 3D fields on A-grid (npz, isd:ied, jsd:jed)  
        '3d': [
            'ua', 'va',           # 3D velocity components
            'uap', 'vap',         # Previous time step 3D velocities
        ],
        
        # Additional 3D fields if needed for baroclinic computations
        '3d1': [
            'w',                  # Vertical velocity (at layer interfaces)
        ]
    }

    def __init__(self):
        """Initialize MomentumData with all required fields."""
        super().__init__()