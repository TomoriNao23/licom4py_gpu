"""
File: data_abc.py
Description: Abstract base class for LICOM data containers,
    delegating field allocation to Global2Local.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-03-19

REVISION HISTORY:
    03/09/2025 - Initial implementation
    19/03/2026 - Refactor import to package-level path
"""
# Local application imports
from licom.mesh import Global2Local


class DataABC():
    def __init__(self) -> None:
        Global2Local.allocate(self, self._field)
