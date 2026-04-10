"""
File: main.py
Description: Script to extract and generate offline Duogrid fields over MPI for JAX native single-process usage.
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
from mpi4py import MPI

# Setup paths so that local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from _c_duogrid_function import init_with_mp, end, get_all_a_grid, get_all_bc_d_grid


class DuogridFieldGenerator:
    """
    Generator for extracting and assembling duogrid spatial fields across MPI tiles.
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
        
        # MPI Setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Grid boundaries and indices
        self.tile = self.rank + 1
        self.isd, self.jsd = -2, -2
        self.ied, self.jed = self.nx + 3, self.nx + 3

    def _init_c_library(self) -> None:
        """Initialize the underlying C grid arrays via the specific duogrid bindings."""
        init_with_mp(
            nx=self.nx, ng=3, tile=self.tile,
            isd=self.isd, ied=self.ied, jsd=self.jsd, jed=self.jed, grid_type=0
        )

    def _gather_tiles_with_id(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Gather partitioned data from all MPI processes into a single unified array on the root process.
        
        Args:
            data (np.ndarray): Local partition data array for the current tile.
            
        Returns:
            Optional[np.ndarray]: The combined multi-tile array on root (rank 0); None on other ranks.
        """
        # Truncate dimensions to target_dim if necessary (to remove excessive halo layers)
        if any(s > self.target_dim for s in data.shape):
            slices = tuple(slice(0, self.target_dim) if s > self.target_dim else slice(None) for s in data.shape)
            data = np.ascontiguousarray(data[slices])

        # 1. Gather tile IDs to determine the final stacking sequence
        tile_ids = None
        if self.rank == 0:
            tile_ids = np.empty(self.size, dtype=np.int32)

        self.comm.Gather(
            sendbuf=np.array(self.tile, dtype=np.int32),
            recvbuf=tile_ids,
            root=0
        )

        # 2. Gather the actual multidimensional data payload
        recvbuf = None
        if self.rank == 0:
            recvbuf = np.empty((self.size, *data.shape), dtype=data.dtype)

        self.comm.Gather(
            sendbuf=data,
            recvbuf=recvbuf,
            root=0
        )

        # 3. Reorder tiles sequentially on the root process
        if self.rank == 0:
            tiles_dict = {tile_ids[r]: recvbuf[r] for r in range(self.size)}
            # Stack the individual tiles along the first dimension strictly sorted by tile ID
            tiles_all = np.stack(
                [tiles_dict[t] for t in sorted(tiles_dict.keys())],
                axis=0
            )
            return tiles_all

        return None

    def _process_and_gather_arrays(self, local_data: Dict[str, np.ndarray], transpose_rules: Dict[tuple, tuple]) -> Dict[str, np.ndarray]:
        """
        Process, transpose based on rules, ensure memory contiguity, and gather arrays across MPI tasks.
        
        Args:
            local_data (Dict[str, np.ndarray]): Dictionary of raw local arrays fetched from C core.
            transpose_rules (Dict[tuple, tuple]): Mapping pointing field names to specific np.transpose tuple operations.
            
        Returns:
            Dict[str, np.ndarray]: Gathered array dictionary mapped by field names (only populated on rank 0).
        """
        gathered_results = {}

        for key, data in local_data.items():
            # Apply any axes transposition rules if matched for consistent layout
            for rule_keys, axes in transpose_rules.items():
                if key in rule_keys:
                    data = np.transpose(data, axes)
                    break
            
            # Data must be strictly contiguous in memory before the block memory transfer in MPI
            data = np.ascontiguousarray(data)
                
            gathered = self._gather_tiles_with_id(data)
            if self.rank == 0 and gathered is not None:
                gathered_results[key] = gathered

        return gathered_results

    def _save_data(self, all_data: Dict[str, np.ndarray]) -> None:
        """
        Filter, report, and serialize the gathered arrays into a .npz file on the root process.
        
        Args:
            all_data (Dict[str, np.ndarray]): Consolidated dictionary containing global data arrays.
        """
        if self.rank != 0:
            return

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
        """Execute the full duogrid field extraction workflow."""
        self._init_c_library()

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

        # Gather distributed dictionaries
        gathered_a = self._process_and_gather_arrays(a_grid, a_rules)
        gathered_bcd = self._process_and_gather_arrays(bcd_grid, bcd_rules)

        # Merge them (only populated on rank 0)
        all_gathered_data = {**gathered_a, **gathered_bcd}
        
        self._save_data(all_gathered_data)

        # End C library routines safely
        end()


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
