"""
File: exceptions.py
Description: Custom exception classes for LICOM model including namelist,
    initialization, and pyFMS integration errors.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""


class NamelistError(Exception):
    """Exception raised for errors in the namelist."""

    pass


class InitialError(Exception):
    """Exception raised for errors in the initial."""

    pass
