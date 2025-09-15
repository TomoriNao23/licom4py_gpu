"""
File: post_init.py
Description: Decorators for handling __post_init__ logic in configuration classes,
    keeping the classes clean and focused on data definition.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-04
"""
# Standard library imports
from datetime import datetime

# Local application imports
from datatype import MpConfig
from backend.calculation.field import BackendConfig
from .timemanager import TimeManager

def time_config_post_init(cls):
    """Decorator: Handle all __post_init__ logic for TimeConfig class"""
    original_post_init = getattr(cls, '__post_init__', None)
    
    def enhanced_post_init(self):
        # 1. Execute validation first (does not rely on cached properties)
        self.check_time_params()

        # 2. Compute start datetime with defaults applied
        start_dt = datetime(
            self.start_year or 2000,
            self.start_month or 1,
            self.start_day or 1,
            self.start_hour or 0,
            self.start_minute or 0,
            self.start_second or 0,
        )

        # 3. Compute end datetime using TimeManager helper
        end_dt = TimeManager.compute_end_datetime(
            start_dt,
            self.integration_years,
            self.integration_months,
            self.integration_days,
            self.integration_hours,
            self.integration_minutes,
            self.integration_seconds,
        )

        # 4. Cache computed values
        object.__setattr__(self, "_start_datetime", start_dt)
        object.__setattr__(self, "_end_datetime", end_dt)
        object.__setattr__(self, "_total_integration_seconds",
                            int((end_dt - start_dt).total_seconds()))
        
        # 5. If there's an original __post_init__, execute it too
        if original_post_init is not None and original_post_init != enhanced_post_init:
            original_post_init(self)
    
    cls.__post_init__ = enhanced_post_init
    return cls

def namelist_post_init(cls):
    """Decorator: Handle all __post_init__ logic for Namelist class"""
    original_post_init = getattr(cls, '__post_init__', None)
    
    def enhanced_post_init(self):
        # 1. First execute parent class __post_init__ (validate time params etc.)
        super(cls, self).__post_init__()
        
        # 2. Execute validation
        self.check_namelist()
        
        # 3. Compute immutable, cached values
        object.__setattr__(self, "_total_baroclinic_steps",
                           int(self._total_integration_seconds / self.baroclinic_dt))
        object.__setattr__(self, "_total_barotropic_steps",
                           int(self._total_integration_seconds / self.barotropic_dt))
        object.__setattr__(self, "_io_layout", (self.io_x, self.io_y))
        object.__setattr__(self, "_mp_cfg", MpConfig(
            nx=self.nx,
            ny=self.ny,
            halo=self.halo,
            npes_x=self.npes_x,
            npes_y=self.npes_y,
            io_layout=self._io_layout,
            ntiles=6,
            npz=self.npz,
        ))
        object.__setattr__(self, "_backend_cfg", BackendConfig(
            platform=self.platform,
            precision=self.precision,
            lib=self.lib,
        ))
        
        # 4. If there's an original __post_init__, execute it too
        if original_post_init is not None and original_post_init != enhanced_post_init:
            original_post_init(self)
    
    cls.__post_init__ = enhanced_post_init
    return cls