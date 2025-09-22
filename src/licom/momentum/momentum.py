from datatype import MomentumData
from duogrid.duogrid import Duogrid as Dg
from .barotropic import add_barotropic_methods
from initial.w92_field import initialize_test_velocity_field

@add_barotropic_methods
class Momentum(MomentumData):

    def __init__(self):
        # Call the parent class MomentumData's __init__ method to initialize attributes.
        super().__init__()

        # Initialize some commonly used scalars/parameters
        self.isb = 0              # Barotropic step counter
        self.nbb = Dg.mp.nbb      # Number of barotropic blocks
        self.rk_barotr = Dg.mp.rk_barotr # Runge-Kutta order for barotropic time stepping
        self.dtb = Dg.mp.dtb      # Barotropic time step
        
        # Physical constants (these might be moved to a constants module)
        self.grav = 9.8           # Gravitational acceleration

        # Initialize LMARS
        from momentum.lmars import LMARS
        self.lmars = LMARS()

        # barotropic method selection
        self.barotr = self.barotr_rk2 if self.rk_barotr == 2 else self.barotr_rk3

        # initialize the fields
        initialize_test_velocity_field(test_case=Dg.mp.case, momentum=self)


    