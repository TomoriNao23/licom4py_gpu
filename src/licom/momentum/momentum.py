"""
File: momentum.py
Description: Momentum class combining diagnostic methods,
    built on MomentumData via decorator pattern.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-04-13

REVISION HISTORY:
    22/09/2025 - Initial implementation of Momentum class
    19/03/2026 - Refactor imports to package-level paths
    13/04/2026 - Removed barotropic JIT setup; Schedule now manages SPMD compilation
"""

# Third-party imports
import jax.numpy as jnp

# Local application imports
from licom.datatype import MomentumData
from licom.duogrid import Dg
from licom.initial.w92_field import initialize_test_velocity_field
from licom.mymodule.diag import add_diag_methods


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
        ## RK type for schedule-level core selection
        self.rk_barotr = namelist.rk_barotr
        # initialize the fields
        initialize_test_velocity_field(momentum=self, test_case=namelist.case)
