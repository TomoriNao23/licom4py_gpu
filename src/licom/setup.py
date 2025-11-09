"""
File: setup.py
Description: Setup configuration for LICOM package installation,
    defining dependencies and package metadata.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Third-party libraries
from setuptools import setup, find_packages  # pyright: ignore[reportMissingModuleSource]

setup(
    name="licom",
    version="0.1.0",
    description="LICOM with Cubed-Sphere",
    author="Chtholly",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "netcdf4",
    ],
    python_requires=">=3.8",
)
