"""
File: main.py
Description: Main entry point for the LICOM Ocean Model. Initializes the model
    and runs either normal simulation or debug mode based on command line arguments.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-03-19

REVISION HISTORY:
    03/09/2026 - Initial implementation of main program
    04/01/2026 - Added timer
    19/03/2026 - Update timer; refactor imports to package-level paths
"""
import time
import contextlib
import jax

# Local application imports
from licom.initial import Initial
from licom.mymodule import Schedule

@contextlib.contextmanager
def jax_timer(name: str):
    """Context manager to time a block of code with JAX synchronization."""
    t_start = time.perf_counter()
    yield
    jax.block_until_ready(jax.numpy.array(0))
    t_end = time.perf_counter()
    print(f"{name} time: {t_end - t_start:.4f} s")

def main():
    """LICOM main program entry"""

    with jax_timer("Initial"):
        licom = Initial()

    with jax_timer("Schedule.run"):
        Schedule.run(licom.momentum)

if __name__ == "__main__":
    main()


