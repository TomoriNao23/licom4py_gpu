"""
File: schedule.py
Description: Main scheduling system for LICOM model execution, coordinating
    barotropic, baroclinic, and tracer steps with proper timing.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
from typing import Callable, Optional, Union, Dict

# Local application imports
from readnamelist import Namelist, Timer
from momentum import barotropic_step, baroclinic_step
from tracer import tracer_step

class Schedule:
    def __init__(self, namelist: Namelist, routines: Optional[Dict[str, Callable]] = None):
        """
        Parameters:
            ratio: Number of barotropic sub-steps
            tracer_interval: Frequency of tracer execution (units: baroclinic steps)
            total_baroclinic_steps: Total number of baroclinic steps
            routines: Dictionary of callback functions
            current_time: Timer object
        """
        self.ratio: int = int(namelist.baroclinic_dt / namelist.barotropic_dt)
        self.tracer_interval: Union[int, None] = namelist.tracer_interval
        self.total_baroclinic_steps: int = namelist._total_baroclinic_steps
        self.routines: Dict[str, Callable] = {
            "barotropic": barotropic_step,
            "baroclinic": baroclinic_step,
            "tracer": tracer_step
        } if routines is None else routines
        self.current_time: Timer = Timer(namelist._start_datetime, namelist.baroclinic_dt)

    def run(self) -> None:
        """
        Total number of baroclinic steps (outer loop)
        """
        for bc_step in range(1, self.total_baroclinic_steps + 1):

            # Multiple barotropic sub-steps
            for bt_sub in range(1, self.ratio + 1):
                if "barotropic" in self.routines:
                    self.routines["barotropic"](bc_step, bt_sub)

            # Baroclinic step
            if "baroclinic" in self.routines:
                self.routines["baroclinic"](bc_step)

            # Tracer process
            if self.tracer_interval is not None:
                if bc_step % self.tracer_interval == 0:
                    if "tracer" in self.routines:
                        self.routines["tracer"](bc_step)

            # Advance current time after each baroclinic step
            self.current_time.time_now()
