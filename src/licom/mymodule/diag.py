"""
File: diag.py
Description: Diagnostic methods for LICOM model using native JAX sharding.
             Correctly aggregates per-shard statistics to avoid double-counting
             halo regions at internal partition boundaries.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15
Updated: 2026-05-14

REVISION HISTORY:
    15/03/2026 - Refactor from MPI to JAX
    14/05/2026 - diagnostics using async

"""

# Third-party imports
import numpy as np

# Local application imports
from licom.kernel import GPU_Mesh


def add_diag_methods(cls):
    """
    Add diagnostic methods to the class.
    
    This diagnostic module now strictly handles pure diagnostic logic.
    It expects that the underlying fields (e.g., self.h0) are already
    NumPy arrays residing in CPU memory.
    """

    def _extract_interior(arr, h, py, px, nx_local, ny_local):
        """
        Extract the interior (non-halo) region from the concatenated GPU array.
        The array from device_get has shape (ntile, py * nx_h, px * ny_h).
        Reshaping allows slicing the halo off every single shard simultaneously.
        """
        ntile = arr.shape[0]
        nx_h = nx_local + 2 * h
        ny_h = ny_local + 2 * h
        arr_blocks = arr.reshape(ntile, py, nx_h, px, ny_h)
        return arr_blocks[:, :, h:-h, :, h:-h]

    def print_global_diag(self, time_str: str):
        """
        Print global diagnostic information.
        Computes the max, min, and mean of the interior regions (excluding halo).
        """
        h = GPU_Mesh.halo
        py = GPU_Mesh.py
        px = GPU_Mesh.px
        nx_local = GPU_Mesh.nx_local
        ny_local = GPU_Mesh.ny_local
        
        # Correctly slice out the halo for all shards (including internal boundaries)
        h0_int = _extract_interior(self.h0, h, py, px, nx_local, ny_local)
        ub_int = _extract_interior(self.ub, h, py, px, nx_local, ny_local)
        vb_int = _extract_interior(self.vb, h, py, px, nx_local, ny_local)
        
        # Local per-shard statistics computed entirely on CPU
        h0_max, h0_min, h0_mean = float(np.max(h0_int)), float(np.min(h0_int)), float(np.mean(h0_int))
        ub_max, ub_min, ub_mean = float(np.max(ub_int)), float(np.min(ub_int)), float(np.mean(ub_int))
        vb_max, vb_min, vb_mean = float(np.max(vb_int)), float(np.min(vb_int)), float(np.mean(vb_int))

        # Format and group the output into a single print to prevent thread interleaving
        out = [
            f"{time_str}",
            "                  max                  min                 mean",
            f"Global h0: {h0_max:20.16f}  {h0_min:20.16f}  {h0_mean:20.16f}",
            f"Global ub: {ub_max:20.16f}  {ub_min:20.16f}  {ub_mean:20.16f}",
            f"Global vb: {vb_max:20.16f}  {vb_min:20.16f}  {vb_mean:20.16f}",
            "------------------------------------------------"
        ]
        print("\n".join(out))

    cls.print_global_diag = print_global_diag
    return cls
