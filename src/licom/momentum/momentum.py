from datatype import MomentumData
from duogrid import Dg
from .barotr import add_barotropic_methods
from initial.w92_field import initialize_test_velocity_field
from mymodule.diag import add_diag_methods
import jax.numpy as jnp

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

        # barotropic method selection
        self.barotr = self.barotr_rk2 if namelist.rk_barotr == 2 else self.barotr_rk3

        # initialize the fields
        initialize_test_velocity_field(momentum=self, test_case=namelist.case)


    