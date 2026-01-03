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

    try:
        # Always initialize LICOM first
        licom = Initial()
        if not debug_mode:
            # Normal mode: run LICOM simulation
            Schedule.run(licom.momentum)
            print_all_time()
            pass
        else:
            # Debug mode: run debug functionality with existing initialization
            print("Running in debug mode...")
            try:
                from mymodule.debug import Debug

                # Redirect output to logs/debug.output
                debug_output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'debug.output')
                os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)

                # Capture debug output and save to file
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                    debug_instance = Debug(licom.namelist)

                # Save debug output to file
                with open(debug_output_path, 'w', encoding='utf-8') as f:
                    f.write(output_buffer.getvalue())

                print(f"Debug output saved to: {debug_output_path}")
                print("Debug mode completed.")

            except Exception as debug_error:
                print(f"Debug mode failed: {debug_error}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LICOM Ocean Model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    main(debug_mode=args.debug)


