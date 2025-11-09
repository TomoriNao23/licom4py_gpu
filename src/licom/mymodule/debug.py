"""
File: debug.py
Description: Debug functionality for LICOM model including pyFMS integration
    verification and simulation step information display.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-03
"""
# Standard library imports
import sys
import os

# Local application imports
from mymodule.exceptions import PyFMSError

# Add src directory to Python path for pyFMS access
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

class Debug:
    """
    LICOM Debug class, integrating all debug functionalities.
    """

    def __init__(self, namelist=None):
        self.namelist = namelist
        print("                 debug mode output")
        self.print_pyfms_integration_info()
        self.print_debug_steps_info()

    def print_debug_steps_info(self):
        """
        Print debug step information for the LICOM simulation.
        """
        if not self.namelist:
            print(" No namelist data, unable to display debug information")
            return

        try:
            start_time = self.namelist._start_datetime.strftime("%Y-%m-%d %H:%M:%S")
            end_time = self.namelist._end_datetime.strftime("%Y-%m-%d %H:%M:%S")
            total_seconds = self.namelist._total_integration_seconds
            total_baroclinic_steps = self.namelist._total_baroclinic_steps
            total_barotropic_steps = self.namelist._total_barotropic_steps

            print("\n=======================================================")
            print("         LICOM Debug Steps Info     ")
            print("=======================================================")
            print(f"Start Time: {start_time}")
            print(f"End Time:   {end_time}")
            print(f"Total Integration Seconds: {total_seconds} s")
            print(f"Total Barotropic Steps:   {total_barotropic_steps} steps")
            print(f"Total Baroclinic Steps:   {total_baroclinic_steps} steps")
            print("=======================================================")

        except (ValueError, TypeError) as e:
            print(f" Error printing debug info: {e}")

    def print_pyfms_integration_info(self):
        """
        Print pyFMS integration information.
        """
        print("\n=======================================================")
        print("         pyFMS Integration Info     ")
        print("=======================================================")

        try:
            # Try importing pyFMS and its submodules
            import pyfms
            from pyfms.py_mpp import mpp, mpp_domains as md
            import pyfms.cfms as _cfms
            from pyfms.py_fms import fms as _fms
            print("✓ pyFMS successfully imported")
            pyfms_path = pyfms.__file__
            if len(pyfms_path) > 50:
                # If path is too long, print in multiple lines
                print("  pyFMS path:")
                for i in range(0, len(pyfms_path), 50):
                    print(f"    {pyfms_path[i:i+50]}")
            else:
                print(f"  pyFMS path: {pyfms_path}")
            # Check available pyFMS modules
            available_modules = []

            # Test each submodule - using correct module paths
            modules_to_test = [
                ('pyfms.cfms', 'cFMS Interface'),
                ('pyfms.py_fms', 'FMS Core'),
                ('pyfms.py_mpp', 'MPP (Message Passing)'),
                ('pyfms.py_diag_manager', 'Diagnostic Manager'),
                ('pyfms.py_field_manager', 'Field Manager'),
                ('pyfms.py_horiz_interp', 'Horizontal Interpolation'),
                ('pyfms.py_data_override', 'Data Override')
            ]

            for module_name, description in modules_to_test:
                try:
                    __import__(module_name)
                    available_modules.append(f"  ✓ {module_name}: {description}")
                except ImportError:
                    available_modules.append(f"  ⚠ {module_name}: {description} (not available)")

            print("\nAvailable pyFMS modules:")
            for module_info in available_modules:
                print(module_info)
            print("=======================================================")
        except Exception:
            from licom.mymodule.exceptions import PyFMSError
            raise PyFMSError("[ERROR] pyFMS mpp/domains not available")
