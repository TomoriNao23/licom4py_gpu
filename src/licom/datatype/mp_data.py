"""
File: mp_data.py
Description: Data structures for MPP (Massively Parallel Processing) configuration
    and domain information used in the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-16
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
        ('npz', int), 
        ('ng', int),
        ('pe', int),
        ('xsize', int),
        ('ysize', int),
        ('platform', str), # Target platform: 'cpu' or 'gpu'
        ('lib', str),      # Backend library: 'jax' or 'numpy'
        ('precision', str) # Data precision for arrays: 'single' or 'double'
    ]
)