"""
File: __init__.py
Description: Initialization module for the mesh package.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-03-19

REVISION HISTORY:
    14/03/2026 - Formatting and header updates
    19/03/2026 - Expose Communication, Cube, Global2Local at package level
    07/04/2026 - Exposed make_spmd_jit wrapper
"""

from .communication import Communication
from .cube import Cube
from .g2l import Global2Local
from .gpu_mesh import GPU_Mesh
from .spmd import auto_pack, auto_unpack, make_spmd_jit
