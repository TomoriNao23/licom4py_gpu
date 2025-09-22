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
        ('npz', int), 
        ('ng', int),
        ('pe', int),
        ('xsize', int),
        ('ysize', int),
        ('platform', str), # Target platform: 'cpu' or 'gpu'
        ('lib', str),      # Backend library: 'jax' or 'numpy'
        ('precision', str), # Data precision for arrays: 'single' or 'double'
        ('dtb', float),    # Barotropic time step
        ('dtc', float),    # Baroclinic time step
        ('nbb', int),      # Number of barotropic blocks
        ('rk_barotr', int), # Runge-Kutta order for barotropic time stepping
        ('case', str),     # Case name
    ]
)