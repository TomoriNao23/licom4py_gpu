"""
File: schedule.py
Description: Main scheduling system for LICOM model execution, coordinating
    barotropic, baroclinic, and tracer steps with proper timing.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-01-07

REVISION HISTORY:
    03/09/2025 - Initial implementation of Schedule class
    01/07/2026 - Refactored execution strategy for performance optimization
"""
# Standard library imports
from typing import Union

# Local application imports
from readnamelist import Namelist, Timer
from momentum.momentum import Momentum
from mymodule.timer import timed
from mymodule import InitialError
from duogrid import Dg


class Schedule:

    @classmethod
    def configure(cls, namelist: Namelist, routines: list = None):
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
        cls._build_execution_strategy()
        
    @classmethod
    def _build_execution_strategy(cls):
        """Construct branch-free execution functions to accelerate runtime"""

        # Initialize all function variables to None
        diag_fn = None
        barotropic_fn = None
        baroclinic_fn = None
        tracer_fn = None
            
        # Diagnostics
        if cls.diag_interval is not None:
            diag_interval = cls.diag_interval
            import jax
            def diag_step(m):
                if jax.process_index() == 0:
                    print(f"Time: {cls.current_time.prev_dt.strftime('%Y-%m-%d-%H')}")
                m.print_global_diag()
            diag_fn = diag_step
        
        # Barotropic
        if "barotropic" in cls.routines:
            barotropic_fn = lambda m: m.barotr()
        
        # Baroclinic
        if "baroclinic" in cls.routines:
            baroclinic_fn = lambda m: None  # TODO: baroclinic logic
        
        # Tracer
        if "tracer" in cls.routines and cls.tracer_interval is not None:
            tracer_interval = cls.tracer_interval
            tracer_fn = lambda m: None  # TODO: tracer logic
        
        # Most common case: barotropic + baroclinic executed every step; tracer conditional; no diagnostics
        if barotropic_fn and baroclinic_fn and tracer_fn and not diag_fn:
            def execute_step(momentum, step):
                barotropic_fn(momentum)
                baroclinic_fn(momentum)
                if step % tracer_interval == 0:
                    tracer_fn(momentum)
            cls._execute_step = execute_step
        # Common case 2: barotropic + baroclinic every step, tracer conditional, with diagnostics
        elif barotropic_fn and baroclinic_fn and tracer_fn and diag_fn:
            def execute_step(momentum, step):
                if step % diag_interval == 0:
                    diag_fn(momentum)
                barotropic_fn(momentum)
                baroclinic_fn(momentum)
                if step % tracer_interval == 0:
                    tracer_fn(momentum)
            cls._execute_step = execute_step
        # Only barotropic with diagnostics
        elif barotropic_fn and not baroclinic_fn and not tracer_fn and diag_fn:
            def execute_step(momentum, step):
                if step % diag_interval == 0:
                    diag_fn(momentum)
                barotropic_fn(momentum)
            cls._execute_step = execute_step
        # Only barotropic without diagnostics
        elif barotropic_fn and not baroclinic_fn and not tracer_fn and not diag_fn:
            def execute_step(momentum, step):
                barotropic_fn(momentum)
            cls._execute_step = execute_step
        else:
            raise InitialError("Execution strategy not implemented for the given routine combination.")

    @classmethod
    @timed(name="simulation", enabled=True)
    def run(cls, momentum: Momentum) -> None:
        """
        Total number of baroclinic steps (outer loop)
        """
        for bc_step in range(cls.total_baroclinic_steps):
            cls._execute_step(momentum, bc_step)
            cls.current_time.time_now()