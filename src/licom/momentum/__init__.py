"""
File: __init__.py
Description: Package initialization for momentum calculations in LICOM
    ocean model including barotropic and baroclinic processes.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
from .barotropic import barotropic_step
from .baroclinic import baroclinic_step
