"""
File: main.py
Description: Main entry point for the LICOM Ocean Model. Initializes the model
    and runs either normal simulation or debug mode based on command line arguments.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-01-04

REVISION HISTORY:
    03/09/2025 - Initial implementation of main program
    04/01/2026 - Added timer
"""
# Standard library imports
import sys
import os
import argparse
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Local application imports
from initial.initial_all import Initial
from mymodule import Schedule, print_all_time

def main(debug_mode=False):
    """LICOM main program entry"""
    # Initialize LICOM only once to avoid FMS variable reallocation
    licom = None

    # Always initialize LICOM first
    licom = Initial()
    Schedule.run(licom.momentum)
    print_all_time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LICOM Ocean Model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    main(debug_mode=args.debug)


