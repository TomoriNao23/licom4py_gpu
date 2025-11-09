"""
File: timemanager.py
Description: Time management utilities for LICOM model including datetime
    calculations and timer functionality for simulation scheduling.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
from datetime import datetime, timedelta
from typing import Optional


class TimeManager:
    """Time-related utilities used by scheduling and config classes."""

    @classmethod
    def compute_end_datetime(
        cls,
        start_dt: datetime,
        integration_years: Optional[int],
        integration_months: Optional[int],
        integration_days: Optional[int],
        integration_hours: Optional[int],
        integration_minutes: Optional[int],
        integration_seconds: Optional[int],
    ) -> datetime:
        """Compute end datetime from start datetime and integration components.

        - Applies year and month increments with proper overflow handling
        - Applies day/hour/minute/second increments via timedelta
        """
        current_dt = start_dt
        if integration_years:
            current_dt = current_dt.replace(year=current_dt.year + integration_years)
        if integration_months:
            new_month = current_dt.month + integration_months
            new_year = current_dt.year + (new_month - 1) // 12
            new_month = (new_month - 1) % 12 + 1
            try:
                current_dt = current_dt.replace(year=new_year, month=new_month)
            except ValueError:
                import calendar
                last_day = calendar.monthrange(new_year, new_month)[1]
                current_dt = current_dt.replace(year=new_year, month=new_month, day=last_day)

        delta = timedelta(
            days=integration_days or 0,
            hours=integration_hours or 0,
            minutes=integration_minutes or 0,
            seconds=integration_seconds or 0,
        )
        return current_dt + delta


class Timer:
    def __init__(self, prev_dt: datetime, delta_dt: int):
        self.prev_dt = prev_dt
        self.delta_dt = delta_dt

    def time_now(self) -> None:
        self.prev_dt += timedelta(seconds=self.delta_dt)
        return None


