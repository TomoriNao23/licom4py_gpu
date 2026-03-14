"""
File: mp_data.py
Description: Data structures for MPP (Massively Parallel Processing) configuration
    and domain information used in the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-22
"""
# Standard library imports
from typing import NamedTuple

# MpDate
# 'ie' is the physical index of the domain
MpDate = NamedTuple(
    'MpDate',
    [
        ('tile', int),
        ('nx', int),
        ('ny', int),
        ('is_', int),
        ('ie', int),
        ('js', int),
        ('je', int),
        ('isd', int),
        ('ied', int),
        ('jsd', int),
        ('jed', int),
        ('ng', int),
        ('pe', int),
        ('xsize', int),
        ('ysize', int),
    ]
)