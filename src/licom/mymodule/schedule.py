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
from duogrid.duogrid import Duogrid as Dg
from readnamelist import Namelist, Timer
from momentum.momentum import Momentum

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
        cls.tracer_interval: Union[int, None] = namelist.tracer_interval
        cls.total_baroclinic_steps: int = namelist._total_baroclinic_steps
        cls.routines: list = ["barotropic", "baroclinic", "tracer"] if routines is None else routines
        cls.current_time: Timer = Timer(namelist._start_datetime, namelist.baroclinic_dt)
        cls.diag_interval: int = 3600 / namelist.baroclinic_dt * namelist.diag_freq

    @classmethod
    def run(cls, momentum: Momentum) -> None:
        """
        Total number of baroclinic steps (outer loop)
        """
        for bc_step in range(cls.total_baroclinic_steps):

            # Diagnostics
            if bc_step % cls.diag_interval == 0:
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
            if cls.tracer_interval is not None:
                if bc_step % cls.tracer_interval == 0:
                    if "tracer" in cls.routines:
                        return

            # Advance current time after each baroclinic step
            cls.current_time.time_now()
