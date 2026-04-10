"""
File: main.py
Description: Script to extract and generate offline Duogrid fields sequentially for JAX native single-process usage.
    Resolution is specified via --nx argument (read from namelist by Makefile).
    Output file is named duogrid_C{nx}.npz and written to the current working directory.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
"""

import os
import sys
import argparse
from typing import Dict, Optional, Set
import numpy as np

# Setup paths so that local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from _c_duogrid_function import init_with_mp, end, get_all_a_grid, get_all_bc_d_grid


class DuogridFieldGenerator:
    """
    Generator for extracting and assembling duogrid spatial fields sequentially.
    """

    ESSENTIAL_FIELDS: Set[str] = {
        # A-grid metrics & operators
        'a_sina', 'a_f', 'rdx', 'rdy', 'rda', 'a_gct',
        # Mapping / Transformation
        'k2e_coef', 'k2e_loc_i', 'k2e_loc_j', 'a_c2l', 'a_l2c',
        # Cross-grid metrics (used in A-grid vorticity/div)
        'c_dy', 'd_dx',
        # Initial field generation (lon/lat)
        'a_pt',
    }

    def __init__(self, nx: int, full_export: bool = False):
        """
        Initialize the generator with grid resolution and export flag.
        
        Args:
            nx (int): Resolution of the cubed-sphere grid (e.g., 96 for C96).
            full_export (bool): If True, export all fields instead of only essential ones.
        """
        self.nx = nx
        self.full_export = full_export
        self.target_dim = nx + 6
        
        # Grid boundaries and indices
        self.isd, self.jsd = -2, -2
        self.ied, self.jed = self.nx + 3, self.nx + 3

    def _process_arrays(self, local_data: Dict[str, np.ndarray], transpose_rules: Dict[tuple, tuple]) -> Dict[str, np.ndarray]:
        """
        Process, transpose based on rules, and ensure memory contiguity.
        """
        processed_results = {}

        for key, data in local_data.items():
            # Apply any axes transposition rules if matched for consistent layout
            for rule_keys, axes in transpose_rules.items():
                if key in rule_keys:
                    data = np.transpose(data, axes)
                    break
            
            # Truncate dimensions to target_dim if necessary (to remove excessive halo layers)
            if any(s > self.target_dim for s in data.shape):
                slices = tuple(slice(0, self.target_dim) if s > self.target_dim else slice(None) for s in data.shape)
                data = np.ascontiguousarray(data[slices])
            else:
                data = np.ascontiguousarray(data)
                
            processed_results[key] = data

        return processed_results

    def _save_data(self, all_data: Dict[str, np.ndarray]) -> None:
        """
        Filter, report, and serialize the gathered arrays into a .npz file.
        """
        if not self.full_export:
            filtered_data = {k: v for k, v in all_data.items() if k in self.ESSENTIAL_FIELDS}
            missing = self.ESSENTIAL_FIELDS - set(all_data.keys())
            if missing:
                print(f"Warning: Expected essential fields not found: {missing}")
            all_data = filtered_data

        print("Gathered arrays dimension summary:")
        for name, arr in all_data.items():
            print(f"  {name:15}: {arr.shape}")

        out_file = os.path.join(os.getcwd(), f"duogrid_C{self.nx}.npz")
        np.savez(out_file, **all_data)
        print(f"Duogrid data generated successfully: {out_file}")

    def run(self) -> None:
        """Execute the full duogrid field extraction workflow across 6 faces sequentially."""
        all_gathered_data = {}
        
        # We loop 6 times to generate the 6 faces without MPI
        for tile in range(1, 7):
            init_with_mp(
                nx=self.nx, ng=3, tile=tile,
                isd=self.isd, ied=self.ied, jsd=self.jsd, jed=self.jed, grid_type=0
            )

            # Retrieve raw local partitions from C library mappings
            a_grid = get_all_a_grid(self.isd, self.ied, self.jsd, self.jed)
            bcd_grid = get_all_bc_d_grid(self.isd, self.ied, self.jsd, self.jed)

            # Specify transpose rules specific to structural layouts mapped from Fortran
            a_rules = {
                ("k2e_coef", "a_pt"): (1, 2, 0),
                ("a_gco", "a_gct", "a_c2l", "a_l2c"): (2, 3, 0, 1)
            }
            
            bcd_rules = {
                ("b_pt",): (1, 2, 0),
                ("c_gco", "c_gct", "c_ct2ort_x", "c_ort2ct_x", "d_gco", "d_gct", "d_ct2ort_y", "d_ort2ct_y"): (2, 3, 0, 1)
            }

            processed_a = self._process_arrays(a_grid, a_rules)
            processed_bcd = self._process_arrays(bcd_grid, bcd_rules)

            merged_tile = {**processed_a, **processed_bcd}
            
            for key, arr in merged_tile.items():
                if key not in all_gathered_data:
                    all_gathered_data[key] = []
                all_gathered_data[key].append(arr)
                
            # End C library routines safely before next tile
            end()

        # Stack over tile axis (axis=0) to form the final multi-tile arrays
        stacked_data = {k: np.stack(v, axis=0) for k, v in all_gathered_data.items()}
        
        self._save_data(stacked_data)


def main():
    """CLI Entrypoint for invoking the generator."""
    parser = argparse.ArgumentParser(description="Generate offline duogrid initial field")
    parser.add_argument(
        "--nx", type=int,
        default=int(os.environ.get("FIELD_NX", 96)),
        help="Grid resolution (default: FIELD_NX env var, then 96)"
    )
    parser.add_argument(
        "--full", action="store_true", 
        help="Export all fields instead of just essential ones"
    )
    args = parser.parse_args()
    
    generator = DuogridFieldGenerator(nx=args.nx, full_export=args.full)
    generator.run()


if __name__ == "__main__":
    main()
