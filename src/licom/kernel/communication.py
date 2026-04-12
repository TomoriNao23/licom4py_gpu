"""
File: communication.py
Description: Defines communication topology, scheduling, and core halo exchange routines for meshes.
             All direction-dependent branching is pre-compiled into routing tables at configure() time.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-04-11

REVISION HISTORY:
    14/03/2026 - Formatting and header updates
    11/04/2026 - Pre-compiled routing tables, eliminated all hot-path if/elif branching
"""

# Third-party imports
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.sharding import PartitionSpec as P


# ==========================================
# Topology Class
# ==========================================
class Topology:
    """
    Responsible for handling the topology connection rules of the grid,
    with built-in direction and rotation constant configurations.
    """

    # Direction constants (East, West, North, South)
    E, W, N, S = 0, 1, 2, 3

    # Transformation rule constants
    ID = 0  # Identity (no transformation)
    FLIP_I_TRANS = 1  # Flip and transpose

    NTILE = 6  # Number of tiles (e.g., for a cubed-sphere grid)

    @classmethod
    def get_transform(cls, dir_a, dir_b):
        """
        Determines the data transformation rule when joining two adjacent faces.
        """
        table = {
            (cls.E, cls.W): cls.ID,
            (cls.W, cls.E): cls.ID,
            (cls.N, cls.S): cls.ID,
            (cls.S, cls.N): cls.ID,
            (cls.E, cls.S): cls.FLIP_I_TRANS,
            (cls.W, cls.N): cls.FLIP_I_TRANS,
            (cls.S, cls.E): cls.FLIP_I_TRANS,
            (cls.N, cls.W): cls.FLIP_I_TRANS,
        }
        return table[(dir_a, dir_b)]

    @classmethod
    def generate_edges(cls):
        """
        Generates topological edges: records the geometric rules of interconnected faces.
        Returns a list of tuples: (tile_A, dir_A, tile_B, dir_B, transform_id).
        """
        edges = []
        for tile in range(cls.NTILE):
            # Abstract adjacency rules for even and odd faces on a cubed sphere
            if tile % 2 == 0:
                rules = {
                    cls.S: ((tile + 4) % 6, cls.E),
                    cls.E: ((tile + 2) % 6, cls.S),
                    cls.N: ((tile + 1) % 6, cls.S),
                    cls.W: ((tile + 5) % 6, cls.E),
                }
            else:
                rules = {
                    cls.E: ((tile + 1) % 6, cls.W),
                    cls.N: ((tile + 2) % 6, cls.W),
                    cls.W: ((tile + 4) % 6, cls.N),
                    cls.S: ((tile + 5) % 6, cls.N),
                }

            # Generate NSEW rotation rules and topological connection edges
            for d_a, (nb, d_b) in rules.items():
                # Only add the edge if tile < nb to avoid duplicating bidirectional edges
                if tile < nb:
                    tid = cls.get_transform(d_a, d_b)
                    edges.append((tile, d_a, nb, d_b, tid))
        return edges


# ==========================================
# Direction Routing Tables (module-level, compiled once at import)
# ==========================================
# Aliases for readability
E, W, N, S = Topology.E, Topology.W, Topology.N, Topology.S

# Mirror direction: the opposite direction for internal neighbor resolution
_MIRROR = {E: W, W: E, N: S, S: N}

# Whether the direction is along the EW axis
_IS_EW = {E: True, W: True, N: False, S: False}

# Coordinate selector: EW directions use y-index, NS directions use x-index
_MY_IDX = {
    E: lambda x, y: y,
    W: lambda x, y: y,
    N: lambda x, y: x,
    S: lambda x, y: x,
}

# Internal neighbor coordinate offset functions
_INT_NEIGHBOR = {
    E: lambda x, y: (x + 1, y),
    W: lambda x, y: (x - 1, y),
    N: lambda x, y: (x, y + 1),
    S: lambda x, y: (x, y - 1),
}

# Internal boundary check functions (is neighbor within this tile?)
_IS_INTERNAL = {
    E: lambda x, y, px, py: x < px - 1,
    W: lambda x, y, px, py: x > 0,
    N: lambda x, y, px, py: y < py - 1,
    S: lambda x, y, px, py: y > 0,
}

# External boundary check functions (is this shard on the tile edge?)
_IS_BOUNDARY = {
    E: lambda x, y, px, py: x == px - 1,
    W: lambda x, y, px, py: x == 0,
    N: lambda x, y, px, py: y == py - 1,
    S: lambda x, y, px, py: y == 0,
}


# ==========================================
# Transform Functions (branch-free, pre-indexed by direction)
# ==========================================
def _transform_ew_scalar(data, tid, n):
    """EW transform for scalar (0-dim) tid."""
    raw = data[..., :, :n]
    swapped = jnp.swapaxes(raw, -1, -2)
    return lax.cond(
        tid == Topology.FLIP_I_TRANS,
        lambda x: jnp.flip(x, axis=-2),
        lambda x: x,
        swapped,
    )


def _transform_ew_batched(data, tid, n):
    """EW transform for batched (>0-dim) tid."""
    raw = data[..., :, :n]
    swapped = jnp.swapaxes(raw, -1, -2)
    tid_mask = jnp.expand_dims(tid, axis=tuple(range(tid.ndim, swapped.ndim)))
    return jnp.where(
        tid_mask == Topology.FLIP_I_TRANS, jnp.flip(swapped, axis=-2), swapped
    )


def _transform_ns_scalar(data, tid, n):
    """NS transform for scalar (0-dim) tid."""
    raw = data[..., :, :n]
    return lax.cond(
        tid == Topology.FLIP_I_TRANS, lambda x: jnp.flip(x, axis=-1), lambda x: x, raw
    )


def _transform_ns_batched(data, tid, n):
    """NS transform for batched (>0-dim) tid."""
    raw = data[..., :, :n]
    tid_mask = jnp.expand_dims(tid, axis=tuple(range(tid.ndim, raw.ndim)))
    return jnp.where(tid_mask == Topology.FLIP_I_TRANS, jnp.flip(raw, axis=-1), raw)


# Pre-indexed transform dispatch tables (direction -> function)
_TRANSFORM_SCALAR = {
    E: _transform_ew_scalar,
    W: _transform_ew_scalar,
    N: _transform_ns_scalar,
    S: _transform_ns_scalar,
}
_TRANSFORM_BATCHED = {
    E: _transform_ew_batched,
    W: _transform_ew_batched,
    N: _transform_ns_batched,
    S: _transform_ns_batched,
}


# ==========================================
# Configure-time Table Builders
# ==========================================
def _build_pack_fn_table(h):
    """
    Build dispatch table for edge packing (8 entries: 4 directions × 2 full modes).
    All if/elif branches are resolved here ONCE at configure() time.
    """
    return {
        (E, False): lambda u: jnp.swapaxes(u[..., h:-h, -2 * h : -h], -1, -2),
        (E, True): lambda u: jnp.swapaxes(u[..., :, -2 * h : -h], -1, -2),
        (W, False): lambda u: jnp.swapaxes(u[..., h:-h, h : 2 * h], -1, -2),
        (W, True): lambda u: jnp.swapaxes(u[..., :, h : 2 * h], -1, -2),
        (N, False): lambda u: u[..., -2 * h : -h, h:-h],
        (N, True): lambda u: u[..., -2 * h : -h, :],
        (S, False): lambda u: u[..., h : 2 * h, h:-h],
        (S, True): lambda u: u[..., h : 2 * h, :],
    }


def _build_tgt_slice_table(h):
    """
    Build dispatch table for halo write-back target slices.
    """
    return {
        (E, False): jnp.s_[..., h:-h, -h:],
        (E, True): jnp.s_[..., :, -h:],
        (W, False): jnp.s_[..., h:-h, :h],
        (W, True): jnp.s_[..., :, :h],
        (N, False): jnp.s_[..., -h:, h:-h],
        (N, True): jnp.s_[..., -h:, :],
        (S, False): jnp.s_[..., :h, h:-h],
        (S, True): jnp.s_[..., :h, :],
    }


def _build_boundary_pack_fn_table(h, IDX_E, IDX_W, IDX_N, IDX_S):
    """
    Build dispatch table for boundary edge packing (C-grid vector field synchronization).
    """
    return {
        E: lambda u, v: v[..., h:-h, IDX_E],
        W: lambda u, v: v[..., h:-h, IDX_W],
        N: lambda u, v: u[..., IDX_N, h:-h],
        S: lambda u, v: u[..., IDX_S, h:-h],
    }


# ==========================================
# Communication Class
# ==========================================
class Communication:
    """
    Communication core responsible for domain updates and boundary mean synchronization.
    All direction-dependent logic is pre-compiled into routing tables at configure() time.
    """

    # Physical edge communication index configuration specific to C-grids
    IDX_W = 3
    IDX_E = -3
    IDX_S = 3
    IDX_N = -3

    halo = 0
    nx_local = 0
    ny_local = 0
    _n_pad = 0
    mesh = None

    _routing = None

    # Pre-compiled dispatch tables (populated by configure())
    _pack_fn = None
    _tgt_slice = None
    _boundary_pack_fn = None

    _update_domain_jit = None
    _boundary_communication_jit = None

    @classmethod
    def configure(cls, halo, nx_local, ny_local, mesh):
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.mesh = mesh
        cls._n_pad = max(nx_local, ny_local)

        edges = Topology.generate_edges()

        # Build static routing map for single-round communication
        routing_list = [[[0, 0, 0] for _ in range(4)] for _ in range(Topology.NTILE)]
        for tA, dA, tB, dB, tid in edges:
            routing_list[tA][dA] = [tB, dB, tid]
            routing_list[tB][dB] = [tA, dA, tid]
        cls._routing = jnp.array(routing_list, dtype=jnp.int32)

        # Pre-compile all dispatch tables (all if/elif resolved here, once)
        cls._pack_fn = _build_pack_fn_table(halo)
        cls._tgt_slice = _build_tgt_slice_table(halo)
        cls._boundary_pack_fn = _build_boundary_pack_fn_table(
            halo, cls.IDX_E, cls.IDX_W, cls.IDX_N, cls.IDX_S
        )

    # -------------------------------------------------
    # Shared Logic (branch-free via routing tables)
    # -------------------------------------------------
    @classmethod
    def _resolve_neighbor(cls, t, x, y, px, py, d):
        """
        Resolve the neighbor shard coordinates for direction d.
        Fully branch-free: all direction logic resolved via pre-compiled lookup tables.
        """
        nb = cls._routing[t, d, 0]
        dB = cls._routing[t, d, 1]
        tid_ext = cls._routing[t, d, 2]

        my_idx = _MY_IDX[d](x, y)

        idx_b_flip = jnp.where(dB <= 1, (py - 1) - my_idx, (px - 1) - my_idx)
        idx_b = jnp.where(tid_ext == Topology.FLIP_I_TRANS, idx_b_flip, my_idx)

        x_b_ext = jnp.where(dB == W, 0, jnp.where(dB == E, px - 1, idx_b))
        y_b_ext = jnp.where(dB == S, 0, jnp.where(dB == N, py - 1, idx_b))

        int_x_b, int_y_b = _INT_NEIGHBOR[d](x, y)
        int_d_b = _MIRROR[d]
        is_internal = _IS_INTERNAL[d](x, y, px, py)

        t_b = jnp.where(is_internal, t, nb)
        x_b = jnp.where(is_internal, int_x_b, x_b_ext)
        y_b = jnp.where(is_internal, int_y_b, y_b_ext)
        d_b = jnp.where(is_internal, int_d_b, dB)
        tid = jnp.where(is_internal, Topology.ID, tid_ext)

        return t_b, x_b, y_b, d_b, tid

    @classmethod
    def _apply_transform(cls, data, recv_dir, tid, nx_eval, ny_eval, full=False):
        """
        Unified data rotation/flip application layer.
        Uses pre-compiled transform dispatch tables — no if/elif at call time.
        """
        n = nx_eval if _IS_EW[recv_dir] else ny_eval
        if full:
            n += 2 * cls.halo

        if hasattr(tid, "ndim") and tid.ndim > 0:
            return _TRANSFORM_BATCHED[recv_dir](data, tid, n)
        return _TRANSFORM_SCALAR[recv_dir](data, tid, n)

    # -------------------------------------------------
    # Domain Update (Halo Exchange)
    # -------------------------------------------------
    @classmethod
    def _pack_edge(cls, u, d, full=False):
        """Extracts boundary data using pre-compiled pack function table."""
        n = cls._n_pad + (2 * cls.halo if full else 0)
        raw = cls._pack_fn[(d, full)](u)
        cur = raw.shape[-1]
        if cur < n:
            pad_width = [(0, 0)] * raw.ndim
            pad_width[-1] = (0, n - cur)
            raw = jnp.pad(raw, pad_width)
        return raw

    @classmethod
    def _gather_halo(cls, send):
        """Gather halos in SPMD distributed mode (valid even on single GPU due to JAX Mesh)."""
        recv_x = lax.all_gather(send, "x", tiled=False)
        recv_xy = lax.all_gather(recv_x, "y", tiled=False)
        recv_global = lax.all_gather(recv_xy, "tile", tiled=False)

        py = recv_global.shape[1]
        px = recv_global.shape[2]

        if send.ndim > 3:
            # recv_global shape is (pdev, py, px, 4, vtile, *batch_dims, h_size, n_size)
            # Dynamically push vtile (axis 4) to be next to pdev (axis 0)
            ndim_rg = recv_global.ndim
            perm = [0, 4, 1, 2, 3] + list(range(5, ndim_rg))
            recv = jnp.transpose(recv_global, tuple(perm))

            # Collapse pdev and vtile into Topology.NTILE (flattening the tile dimension)
            new_shape = (Topology.NTILE,) + recv.shape[2:]
            recv = recv.reshape(new_shape)

            mesh_t = lax.axis_index("tile")
            vtile = send.shape[1]
            t = mesh_t * vtile + jnp.arange(vtile)
        else:
            recv = recv_global
            t = lax.axis_index("tile")

        x = lax.axis_index("x")
        y = lax.axis_index("y")
        return recv, t, x, y, px, py

    @classmethod
    def _update_domain_single(cls, u):
        """Applies boundary exchange using all_gather and ppermute (to correctly sync diagonal corners)."""
        h = cls.halo
        px = cls.mesh.shape["x"]
        py = cls.mesh.shape["y"]
        x = lax.axis_index("x")
        y = lax.axis_index("y")

        def process_dir(u_curr, recv, t, x_, y_, px_, py_, d, full):
            t_b, x_b, y_b, d_b, tid = cls._resolve_neighbor(t, x_, y_, px_, py_, d)
            nx_eval = u_curr.shape[-2] - 2 * h
            ny_eval = u_curr.shape[-1] - 2 * h
            return cls._apply_transform(
                recv[t_b, y_b, x_b, d_b], d, tid, nx_eval, ny_eval, full
            )

        def run_pass_global(u_curr):
            send = jnp.stack(
                [
                    cls._pack_edge(u_curr, E, False),
                    cls._pack_edge(u_curr, W, False),
                    cls._pack_edge(u_curr, N, False),
                    cls._pack_edge(u_curr, S, False),
                ]
            )
            recv, t_idx, x_idx, y_idx, px_idx, py_idx = cls._gather_halo(send)

            res = u_curr
            for d in (E, W, N, S):
                tgt_slice = cls._tgt_slice[(d, False)]
                is_boundary = _IS_BOUNDARY[d](x_idx, y_idx, px_idx, py_idx)

                # Use a closure factory to properly isolate 'd' and 'tgt_slice' during trace loop
                def make_branch(d_val, slice_val):
                    return lambda r: r.at[slice_val].set(
                        process_dir(u_curr, recv, t_idx, x_idx, y_idx, px_idx, py_idx, d_val, False)
                    )

                res = lax.cond(is_boundary, make_branch(d, tgt_slice), lambda r: r, res)
            return res

        # Pass 1: Global boundaries (updates cross-tile edge halos, omitting diagonal corners)
        u = run_pass_global(u)

        # Pass 2: Local EW (updates X-axis halos for sub-tiles within a mesh tile, using full wide edge to sweep new global data)
        if px > 1:
            send_ew = jnp.stack([u[..., :, h : 2 * h], u[..., :, -2 * h : -h]], axis=-1)
            recv_ew = lax.all_gather(send_ew, "x", tiled=False)

            recv_from_E = recv_ew[(x + 1) % px, ..., 0]
            u = lax.cond(
                x < px - 1,
                lambda val: val.at[..., :, -h:].set(recv_from_E),
                lambda val: val,
                u,
            )

            recv_from_W = recv_ew[(x - 1 + px) % px, ..., 1]
            u = lax.cond(
                x > 0,
                lambda val: val.at[..., :, :h].set(recv_from_W),
                lambda val: val,
                u,
            )

        # Pass 3: Local NS (final sweep updating Y-axis halos including newly populated corners)
        if py > 1:
            send_ns = jnp.stack([u[..., h : 2 * h, :], u[..., -2 * h : -h, :]], axis=-1)
            recv_ns = lax.all_gather(send_ns, "y", tiled=False)

            recv_from_N = recv_ns[(y + 1) % py, ..., 0]
            u = lax.cond(
                y < py - 1,
                lambda val: val.at[..., -h:, :].set(recv_from_N),
                lambda val: val,
                u,
            )

            recv_from_S = recv_ns[(y - 1 + py) % py, ..., 1]
            u = lax.cond(
                y > 0,
                lambda val: val.at[..., :h, :].set(recv_from_S),
                lambda val: val,
                u,
            )

        return u

    @classmethod
    def update_domain(cls, u):
        """Public API to trigger domain update within the defined Mesh."""
        with cls.mesh:
            return cls._update_domain_single(u)

    # -------------------------------------------------
    # C-Grid Boundary Edge Synchronization
    # -------------------------------------------------
    @classmethod
    def _pack_boundary_edge(cls, u, v, d):
        """Extracts boundary lines using pre-compiled dispatch table."""
        h = cls.halo
        n = cls._n_pad

        line = cls._boundary_pack_fn[d](u, v)
        line = jnp.expand_dims(line, -2)
        pad_h = [(0, 0)] * line.ndim
        pad_h[-2] = (0, h - 1)
        raw = jnp.pad(line, pad_h)

        cur = raw.shape[-1]
        if cur < n:
            pad_width = [(0, 0)] * raw.ndim
            pad_width[-1] = (0, n - cur)
            raw = jnp.pad(raw, pad_width)
        return raw

    @classmethod
    def _boundary_communication_single(cls, u, v):
        """Applies single-round boundary synchronization for vector fields."""
        send = jnp.stack(
            [
                cls._pack_boundary_edge(u, v, E),
                cls._pack_boundary_edge(u, v, W),
                cls._pack_boundary_edge(u, v, N),
                cls._pack_boundary_edge(u, v, S),
            ]
        )

        recv, t, x, y, px, py = cls._gather_halo(send)

        def process_dir(d):
            t_b, x_b, y_b, d_b, tid = cls._resolve_neighbor(t, x, y, px, py, d)
            nx_eval = u.shape[-2] - 2 * cls.halo
            ny_eval = u.shape[-1] - 2 * cls.halo
            return cls._apply_transform(
                recv[t_b, y_b, x_b, d_b], d, tid, nx_eval, ny_eval, full=False
            )

        h = cls.halo

        arr_E = process_dir(E)
        val_E = ((v[..., h:-h, cls.IDX_E] + arr_E[..., :, 0]) / 2.0).astype(v.dtype)
        v = lax.cond(
            x == px - 1,
            lambda val: val.at[..., h:-h, cls.IDX_E].set(val_E),
            lambda val: val,
            v,
        )

        arr_W = process_dir(W)
        val_W = ((v[..., h:-h, cls.IDX_W] + arr_W[..., :, 0]) / 2.0).astype(v.dtype)
        v = lax.cond(
            x == 0,
            lambda val: val.at[..., h:-h, cls.IDX_W].set(val_W),
            lambda val: val,
            v,
        )

        arr_N = process_dir(N)
        val_N = ((u[..., cls.IDX_N, h:-h] + arr_N[..., 0, :]) / 2.0).astype(u.dtype)
        u = lax.cond(
            y == py - 1,
            lambda val: val.at[..., cls.IDX_N, h:-h].set(val_N),
            lambda val: val,
            u,
        )

        arr_S = process_dir(S)
        val_S = ((u[..., cls.IDX_S, h:-h] + arr_S[..., 0, :]) / 2.0).astype(u.dtype)
        u = lax.cond(
            y == 0,
            lambda val: val.at[..., cls.IDX_S, h:-h].set(val_S),
            lambda val: val,
            u,
        )

        return u, v

    @classmethod
    def boundary_communication(cls, u, v):
        """Public API to trigger boundary synchronization within the defined Mesh."""
        with cls.mesh:
            return cls._boundary_communication_single(u, v)

    # -------------------------------------------------
    # Print Scheduling Info
    # -------------------------------------------------
    @classmethod
    def print_schedule_info(cls):
        """Prints the calculated topology routing."""
        if cls._routing is None:
            print(
                "Error: Communication is not yet configured. Please call configure() first."
            )
            return

        print("Communication routing (Single Round All-Gather):")
        dn = {E: "E", W: "W", N: "N", S: "S"}
        for t in range(Topology.NTILE):
            routes = []
            for d in (E, W, N, S):
                nb, dB, _ = cls._routing[t, d]
                routes.append(f"{dn[d]}<-T{nb+1}:{dn[int(dB)]}")
            print(f"  Tile {t+1}: " + ", ".join(routes))
