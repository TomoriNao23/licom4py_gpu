"""
File: time_data.py
Description: Time configuration class for LICOM model, managing start time,
    integration periods, and computed datetime properties.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# Local application imports
from .post_init import time_config_post_init
from .validators import validate_time_params

@validate_time_params
@time_config_post_init
@dataclass(frozen=True, slots=True)
class TimeConfig:
    """Base time configuration for start time and integration period."""

    # Start time parameters - all optional with default values (2000-01-01 00:00:00)
    start_year: Optional[int] = field(default=None, metadata={"section": "time_params"})
    start_month: Optional[int] = field(default=None, metadata={"section": "time_params"})
    start_day: Optional[int] = field(default=None, metadata={"section": "time_params"})
    start_hour: Optional[int] = field(default=None, metadata={"section": "time_params"})
    start_minute: Optional[int] = field(default=None, metadata={"section": "time_params"})
    start_second: Optional[int] = field(default=None, metadata={"section": "time_params"})

    # Integration time parameters (all optional, but at least one must be specified)
    integration_years: Optional[int] = field(default=None, metadata={"section": "time_params"})
    integration_months: Optional[int] = field(default=None, metadata={"section": "time_params"})
    integration_days: Optional[int] = field(default=None, metadata={"section": "time_params"})
    integration_hours: Optional[int] = field(default=None, metadata={"section": "time_params"})
    integration_minutes: Optional[int] = field(default=None, metadata={"section": "time_params"})
    integration_seconds: Optional[int] = field(default=None, metadata={"section": "time_params"})

    # cached, computed once after init
    _start_datetime: datetime = field(init=False, repr=False)
    _end_datetime: datetime = field(init=False, repr=False)
    _total_integration_seconds: int = field(init=False, repr=True)