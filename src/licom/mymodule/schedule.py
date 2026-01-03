"""
File: schedule.py
Description: Main scheduling system for LICOM model execution, coordinating
    barotropic, baroclinic, and tracer steps with proper timing.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
from typing import Union

# Local application imports
from mymodule.timer import stop_timer
from readnamelist import Namelist, Timer
from momentum.momentum import Momentum
from mymodule.timer import timed, print_all_time
from duogrid.duogrid import Duogrid as Dg

class Schedule:

    @classmethod
    def init(cls, namelist: Namelist, routines: list = None):
        """
        Parameters:
            tracer_interval: Frequency of tracer execution (units: baroclinic steps)
            total_baroclinic_steps: Total number of baroclinic steps
            routines: List of routines to execute
            current_time: Timer object
        """
        cls.current_time: Timer = Timer(namelist._start_datetime, namelist.baroclinic_dt)
        cls.tracer_interval: Union[int, None] = namelist.tracer_interval
        cls.total_baroclinic_steps: int = namelist._total_baroclinic_steps
        cls.routines: list = ["barotropic", "baroclinic", "tracer"] \
            if routines is None else routines
        cls.diag_interval: int = 3600 / namelist.baroclinic_dt * namelist.diag_freq \
            if namelist.diag_freq is not None and namelist.diag_freq > 0 else None

    @classmethod
    @timed(name="simulation", enabled=True)
    def run(cls, momentum: Momentum) -> None:
        """
        Total number of baroclinic steps (outer loop)
        """

        for bc_step in range(cls.total_baroclinic_steps):

            # Diagnostics
            if cls.diag_interval is not None and bc_step % cls.diag_interval == 0:
                if Dg.mp.pe == 0:
                    print(f"Time: {cls.current_time.prev_dt.strftime('%Y-%m-%d-%H')}")
                momentum.print_global_diag()

            # Multiple barotropic sub-steps
            if "barotropic" in cls.routines:
                momentum.barotr()

            # Baroclinic step
            if "baroclinic" in cls.routines:
                return

            # Tracer process
            if ("tracer" in cls.routines) and \
                (cls.tracer_interval is not None) and \
                (bc_step % cls.tracer_interval == 0):
                    return

            # Advance current time after each baroclinic step
            cls.current_time.time_now()

