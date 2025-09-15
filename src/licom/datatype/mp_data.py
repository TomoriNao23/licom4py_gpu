"""
File: mp_data.py
Description: Data structures for MPP (Massively Parallel Processing) configuration
    and domain information used in the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-11
"""
# Standard library imports
from typing import NamedTuple, Tuple

# MpConfig
MpConfig = NamedTuple(
    'MpConfig',
    [
        ('nx', int),
        ('ny', int),
        ('npes_x', int),
        ('npes_y', int),
        ('io_layout', Tuple[int, int]),
        ('ntiles', int),
        ('halo', int),
        ('npz', int)
    ]
)

# MpDomain
# 'ie' is the physical index of the domain
MpDomain = NamedTuple(
    'MpDomain',
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
        ('npz', int), 
        ('ng', int),
    ]
)