"""
File: barotropic.py
Description: Barotropic momentum calculations for LICOM ocean model,
    handling depth-averaged currents and surface gravity waves.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""

def barotropic_step(bc_step, bt_sub):
    print(f"[BC {bc_step}] Execute barotropic substep {bt_sub}")