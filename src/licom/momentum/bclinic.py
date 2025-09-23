"""
File: baroclinic.py
Description: Baroclinic momentum calculations for LICOM ocean model,
    handling vertical shear and density-driven currents.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""

def baroclinic_step(bc_step):
    print(f"[BC {bc_step}] Execute baroclinic step")