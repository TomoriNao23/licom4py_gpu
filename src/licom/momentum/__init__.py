"""
File: __init__.py
Description: Package initialization for momentum calculations in LICOM
    ocean model including barotropic and baroclinic processes.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-03-19

REVISION HISTORY:
    03/09/2025 - Initial implementation
    19/03/2026 - Expose Momentum at package level
"""
# Local application imports
from .momentum import Momentum
