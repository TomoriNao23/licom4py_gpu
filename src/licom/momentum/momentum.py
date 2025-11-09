from datatype import MomentumData
from duogrid.duogrid import Duogrid as Dg
from .barotr import add_barotropic_methods
from initial.w92_field import initialize_test_velocity_field
from mymodule.diag import add_diag_methods

@add_barotropic_methods
@add_diag_methods
class Momentum(MomentumData):

    def __init__(self):
        # Call the parent class MomentumData's __init__ method to initialize attributes.
        super().__init__()

        # Initialize some commonly used scalars/parameters
        ## Barotropic step counter
        self.isb = 0         
        ## Number of barotropic blocks(time.baroclinic/time.barotropic)     
        self.nbb = Dg.mp.nbb
        ## Barotropic time step
        self.dtb = Dg.mp.dtb

        # barotropic method selection
        self.barotr = self.barotr_rk2 if Dg.mp.rk_barotr == 2 else self.barotr_rk3

        # initialize the fields
        initialize_test_velocity_field(momentum=self, test_case=Dg.mp.case)


    