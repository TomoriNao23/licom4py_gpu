"""
File: validators.py
Description: Validation mixin for LICOM namelist configuration, providing
    time parameter and namelist validation utilities.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
from datetime import datetime

def validate_time_params(cls):
    """Decorator: Validate time parameters for configuration classes"""
    
    # Add validation method to the class
    def check_time_params(self) -> None:
        if getattr(self, "start_month") is not None:
            self._require(1 <= self.start_month <= 12, "Start month must be between 1 and 12")
        if getattr(self, "start_day") is not None:
            self._require(1 <= self.start_day <= 31, "Start day must be between 1 and 31")
        if getattr(self, "start_hour") is not None:
            self._require(0 <= self.start_hour <= 23, "Start hour must be between 0 and 23")
        if getattr(self, "start_minute") is not None:
            self._require(0 <= self.start_minute <= 59, "Start minute must be between 0 and 59")
        if getattr(self, "start_second") is not None:
            self._require(0 <= self.start_second <= 59, "Start second must be between 0 and 59")
        if getattr(self, "start_year") is not None:
            self._require(self.start_year > 0, "Start year must be positive")

        try:
            datetime(
                getattr(self, "start_year") or 2000,
                getattr(self, "start_month") or 1,
                getattr(self, "start_day") or 1,
                getattr(self, "start_hour") or 0,
                getattr(self, "start_minute") or 0,
                getattr(self, "start_second") or 0,
            )
        except ValueError as e:
            self._require(False, f"Invalid start date: {e}")

        integration_specified = any([
            getattr(self, "integration_years"),
            getattr(self, "integration_months"),
            getattr(self, "integration_days"),
            getattr(self, "integration_hours"),
            getattr(self, "integration_minutes"),
            getattr(self, "integration_seconds"),
        ])
        self._require(integration_specified, "At least one integration time component must be specified")

        time_fields = [
            "integration_years",
            "integration_months",
            "integration_days",
            "integration_hours",
            "integration_minutes",
            "integration_seconds",
        ]
        for field_name in time_fields:
            value = getattr(self, field_name)
            if value is not None:
                self._require(value > 0, f"{field_name} must be positive if specified")
    
    # Add helper method to the class
    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            from licom import NamelistError
            raise NamelistError(message)
    
    # Attach methods to the class
    cls.check_time_params = check_time_params
    cls._require = _require
    
    return cls
    
def validate_namelist_class(cls):
    """Decorator: Validate Namelist class parameters and add validation methods"""
    
    # Add validation method to the class
    def check_namelist(self) -> None:
        # run_params validations
        self._require(self.baroclinic_dt > 0, "baroclinic_dt must be greater than zero.")
        self._require(self._total_integration_seconds >= self.baroclinic_dt,
                "total_integration_seconds must be at least as large as baroclinic_dt.")
        self._require(self._total_integration_seconds % self.baroclinic_dt == 0,
                "total_integration_seconds must be divisible by baroclinic_dt.")
        self._require(self.barotropic_dt > 0, "barotropic_dt must be greater than zero.")
        self._require(self.baroclinic_dt % self.barotropic_dt == 0,
                "baroclinic_dt must be an integer multiple of barotropic_dt.")
        self._require(self.tracer_interval is None or self.tracer_interval > 0,
                "Tracer frequency must be a positive integer or None")
        self._require(self.barotropic_dt <= self.baroclinic_dt,
                "Barotropic time step must be less than baroclinic time step")

        # grid validations
        self._require(self.nx > 0 and self.ny > 0, "nx, ny must be positive")
        self._require(self.halo >= 0, "halo must be non-negative")
        self._require(self.npz > 0, "npz must be positive")

        # mpi validations
        self._require(self.px > 0 and self.py > 0, "px, py must be positive")
        self._require(self.pdev > 0, "pdev must be positive")
    
    # Attach method to the class
    cls.check_namelist = check_namelist
    
    return cls
