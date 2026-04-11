"""
File: momentum.py
Description: Momentum class combining barotropic and diagnostic methods,
    built on MomentumData via decorator pattern.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-03-19

REVISION HISTORY:
    22/09/2025 - Initial implementation of Momentum class
    19/03/2026 - Refactor imports to package-level paths
"""

# Third-party imports
import jax.numpy as jnp

# Local application imports
from licom.datatype import MomentumData
from licom.duogrid import Dg
from licom.initial.w92_field import initialize_test_velocity_field
from licom.mymodule.diag import add_diag_methods

from .barotr import add_barotropic_methods


@add_barotropic_methods
@add_diag_methods
class Momentum(MomentumData):

    def __init__(self, namelist):
        # Call the parent class MomentumData's __init__ method to initialize attributes.
        super().__init__()

        # Initialize some commonly used scalars/parameters
        ## Barotropic step counter
        self.isb = jnp.int32(0)
        ## Number of barotropic blocks(time.baroclinic/time.barotropic)
        self.nbb = namelist.baroclinic_dt // namelist.barotropic_dt
        ## Barotropic time step
        self.dtb = float(namelist.barotropic_dt)
        # initialize the fields
        initialize_test_velocity_field(momentum=self, test_case=namelist.case)
        # Execute only once during initialization: State packing and JIT graph compilation
        self.setup_barotropic_jit(namelist.rk_barotr)
