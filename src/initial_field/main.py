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
import numpy as np

# Setup paths so that local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from use_mpp import FMS_chtholly
from _c_duogrid_function import init_with_mp, end, get_all_a_grid, get_all_bc_d_grid

def gather_tiles_with_id(data, tile):

        from mpi4py import MPI
        import numpy as np

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # 1. 收 tile id
        tile_ids = None
        if rank == 0:
            tile_ids = np.empty(size, dtype=np.int32)

        comm.Gather(
            sendbuf=np.array(tile, dtype=np.int32),
            recvbuf=tile_ids,
            root=0
        )

        # 2. 收数据
        recvbuf = None
        if rank == 0:
            recvbuf = np.empty((size, *data.shape), dtype=data.dtype)

        comm.Gather(
            sendbuf=data,
            recvbuf=recvbuf,
            root=0
        )
        # 3. rank 0 按 tile 重排
        if rank == 0:
            tiles = {}
            # We can detect nx from the local geometry in python FMS wrapper
            # It's exposed by FMS_chtholly.mp.nx
            from use_mpp import FMS_chtholly
            nx = FMS_chtholly.mp.nx
            target_dim = nx + 6

            for r in range(size):
                arr = recvbuf[r]
                # Slice logic: check each dimension of the array.
                # If its size is nx + 6, apply slice [3:-3]
                slices = []
                for dim_size in arr.shape:
                    if dim_size == target_dim:
                        slices.append(slice(3, -3))
                    else:
                        slices.append(slice(None))
                tiles[tile_ids[r]] = arr[tuple(slices)]

            # 保证 tile 顺序
            tiles_all = np.stack(
                [tiles[t] for t in sorted(tiles.keys())],
                axis=0
            )
            return tiles_all

        return None

def generate(nx: int):
    # NOTE: Do NOT import mpi4py before chtholly_init!
    # mpi4py calls MPI_Init() at import time, which causes FMS mpp_init to
    # abort because MPI is already initialized but no localcomm was provided.
    # Instead, use FMS library functions to get rank after FMS is initialized.
    FMS_chtholly.configure(nx)
    rank = FMS_chtholly._lib.chtholly_get_mpp_pe()
    mp = FMS_chtholly.mp

    # Initialize c_global_grid
    init_with_mp(
        nx=mp.nx, ng=3, tile=mp.tile,
        isd=mp.isd, ied=mp.ied, jsd=mp.jsd, jed=mp.jed, grid_type=0
    )

    # Get local data
    a = get_all_a_grid(mp.isd, mp.ied, mp.jsd, mp.jed)
    bcd = get_all_bc_d_grid(mp.isd, mp.ied, mp.jsd, mp.jed)

    # Gather data across processes
    all_data = {}

    for key, data in a.items():
        if key in ["k2e_coef", "a_pt"]:
            data = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        elif key in ["a_gco", "a_gct", "a_c2l", "a_l2c"]:
            data = np.ascontiguousarray(np.transpose(data, (2, 3, 0, 1)))
            
        gathered = gather_tiles_with_id(data, mp.tile)
        if rank == 0 and gathered is not None:
            all_data[key] = gathered

    for key, data in bcd.items():
        if key in ["b_pt"]:
            data = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        elif key in ["c_gco", "c_gct", "c_ct2ort_x", "c_ort2ct_x", "d_gco", "d_gct", "d_ct2ort_y", "d_ort2ct_y"]:
            data = np.ascontiguousarray(np.transpose(data, (2, 3, 0, 1)))

        gathered = gather_tiles_with_id(data, mp.tile)
        if rank == 0 and gathered is not None:
            all_data[key] = gathered

    # Save to file named by resolution: duogrid_C{nx}.npz in CWD (field/)
    if rank == 0:
        out_file = os.path.join(os.getcwd(), f"duogrid_C{nx}.npz")
        np.savez(out_file, **all_data)
        print(f"Duogrid data generated successfully: {out_file}")

    # End
    end()
    FMS_chtholly.end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline duogrid initial field")
    parser.add_argument(
        "--nx", type=int,
        default=int(os.environ.get("FIELD_NX", 96)),
        help="Grid resolution (default: FIELD_NX env var, then 96)"
    )
    args = parser.parse_args()
    generate(nx=args.nx)
