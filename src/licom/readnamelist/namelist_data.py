"""
File: namelist_data.py
Description: Main namelist configuration class for LICOM model, defining
    run parameters, grid settings, and GPU mesh configuration.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-05-14

REVISION HISTORY:
    03/09/2025 - Initial implementation of Namelist class
    16/09/2025 - Added validation
    14/05/2026 - Add async_io section
"""

# Standard library imports
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .loader import LoaderMixin
from .post_init import namelist_post_init
from .time_data import TimeConfig
from .validators import validate_namelist_class


@validate_namelist_class
@namelist_post_init
@dataclass(frozen=True, slots=True)
class Namelist(TimeConfig, LoaderMixin):
    """Main namelist configuration class inheriting time configuration."""

    namelist_local = ["run_params", "time_params", "grid", "gpu_mesh", "async_io"]

    # run_params section
    barotropic_dt: int = field(default=150, metadata={"section": namelist_local[0]})
    baroclinic_dt: int = field(default=3000, metadata={"section": namelist_local[0]})
    tracer_interval: Optional[int] = field(
        default=None, metadata={"section": namelist_local[0]}
    )
    rk_barotr: int = field(default=2, metadata={"section": namelist_local[0]})
    case: str = field(default="w92case2", metadata={"section": namelist_local[0]})

    # grid section
    nx: int = field(default=96, metadata={"section": namelist_local[2]})
    ny: int = field(default=96, metadata={"section": namelist_local[2]})
    npz: int = field(default=1, metadata={"section": namelist_local[2]})
    halo: int = field(default=1, metadata={"section": namelist_local[2]})

    # mesh section
    pdev: int = field(default=1, metadata={"section": namelist_local[3]})
    px: int = field(default=1, metadata={"section": namelist_local[3]})
    py: int = field(default=1, metadata={"section": namelist_local[3]})

    # computed once after init
    _io_layout: Tuple[int, int] = field(init=False, repr=False)
    _total_baroclinic_steps: int = field(init=False, repr=True)
    _total_barotropic_steps: int = field(init=False, repr=True)

    # async_io section
    diag_freq: Optional[int] = field(
        default=None, metadata={"section": namelist_local[4]}
    )
    async_threads: int = field(
        default=2, metadata={"section": namelist_local[4]}
    )
